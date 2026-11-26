import type OnnxGraph from "../../ir/OnnxGraph.js";
import OperationNode from "../../ir/OperationNode.js";
import TensorNode from "../../ir/TensorNode.js";
import RegionArgumentNode from "../../ir/RegionArgumentNode.js";
import type { ConcreteValueNode, KnownShape, ValueNode } from "../../ir/OnnxTypes.js";
import { GraphBuilder } from "../../ir/GraphBuilder.js";
import {
    int64Vec,
    readScalarFromTensorNode,
    scalarInt64,
    topologicalSortOperationNodes,
    toStaticShape,
} from "../../utils/index.js";
import type { GraphPass } from "../GraphPass.js";
import { TrackedGraphBuilder } from "../tracking/TrackedGraphBuilder.js";
import type { HistoryManager } from "../tracking/HistoryManager.js";
import { matchesScheduleTarget } from "../Schedule.js";

export interface LoopUnrollSpec {
    target: string;
    maxTripCount?: number;
}

/**
 * Statically unrolls innermost ONNX `Loop` nodes whose constant trip count does
 * not exceed `maxTripCount`, cloning the body per iteration and stacking any
 * `scan_outputs` via `Unsqueeze` + `Concat`.
 */
export class LoopUnrollingPass implements GraphPass {
    public readonly name = "ScheduleLoopUnroll";

    constructor(private readonly specs: readonly LoopUnrollSpec[]) {}

    public run(graph: OnnxGraph.Class, historyManager?: HistoryManager): boolean {
        let changed = false;
        for (const spec of this.specs) {
            const maxTrips = spec.maxTripCount ?? 16;
            if (this.unrollInGraph(graph, spec.target, maxTrips, historyManager)) {
                changed = true;
            }
        }
        return changed;
    }

    private unrollInGraph(
        g: OnnxGraph.Class,
        target: string,
        maxTrips: number,
        hm?: HistoryManager,
    ): boolean {
        let localChanged = false;
        for (const op of topologicalSortOperationNodes(g)) {
            if (!g.hasNode(op.id)) continue;
            for (const sub of op.regions) {
                if (this.unrollInGraph(sub, target, maxTrips)) localChanged = true;
            }
            if (op.type !== "Loop") continue;
            if (!matchesScheduleTarget(target, op)) continue;

            const inputs = op.getInputs() ?? [];
            if (inputs.length < 2) continue;
            const tripCount = readScalarFromTensorNode(inputs[0]);
            if (
                tripCount === undefined ||
                !Number.isInteger(tripCount) ||
                tripCount < 1 ||
                tripCount > maxTrips
            ) {
                continue;
            }
            if (op.regions.length === 0) continue;
            const loopBody = op.regions[0];
            const hasInnerLoop = loopBody
                .getOperationNodes()
                .toArray()
                .some((o) => o.type === "Loop");
            if (hasInnerLoop) continue;

            const tb: GraphBuilder = hm
                ? new TrackedGraphBuilder(
                      g,
                      `ScheduleUnroll_${op.id}_${tripCount}`,
                      `Unroll Loop '${op.id}' (${tripCount} iterations)`,
                      "schedule_unroll",
                  )
                : new GraphBuilder(g, `unroll_${op.id}`);

            LoopUnrollingPass.unrollStaticLoop(op, tripCount, tb);
            if (hm) {
                hm.pushPatch((tb as TrackedGraphBuilder).commitPatch());
            }
            localChanged = true;
        }
        return localChanged;
    }

    /** Rewrites a single static-trip-count `Loop` into `tripCount` inlined iterations. */
    public static unrollStaticLoop(
        loopOp: OperationNode.Class,
        tripCount: number,
        builder: GraphBuilder,
    ): void {
        const graph = builder.graph;
        const loopBody = loopOp.regions[0];
        const loopInputs = loopOp.getInputs() ?? [];
        const condInit = loopInputs[1] as ConcreteValueNode;
        const carryInits = loopInputs.slice(2) as ConcreteValueNode[];
        const outerOutputs = loopOp.getOutputs();

        const bodyInputs = loopBody.getInputTensorNodes().toArray();
        const bodyIter = bodyInputs[0];
        const bodyCondIn = bodyInputs[1];
        const bodyCarryIns = bodyInputs.slice(2);
        const bodyOutputs = loopBody.getOutputTensorNodes().toArray();
        const numCarries = carryInits.length;
        const numScans = Math.max(0, bodyOutputs.length - 1 - numCarries);

        const resolveInnerOutputSource = (innerOutTensor: TensorNode.Class): ValueNode => {
            const prod = innerOutTensor.incomers.sources.filterIs(OperationNode).first();
            if (prod && prod.type === "Identity") {
                return (prod.getInputs()?.[0] ?? innerOutTensor) as ValueNode;
            }
            return innerOutTensor;
        };

        const condOutSource = resolveInnerOutputSource(bodyOutputs[0]);
        const carryOutSources = bodyOutputs.slice(1, 1 + numCarries).map(resolveInnerOutputSource);
        const scanOutSources = bodyOutputs.slice(1 + numCarries).map(resolveInnerOutputSource);

        let curCond: ValueNode = condInit;
        let curCarries: ValueNode[] = [...carryInits];
        const scanSteps: ConcreteValueNode[][] = Array.from({ length: numScans }, () => []);
        const ax0 = builder.createConstant(`unroll_ax0_${loopOp.id}`, int64Vec([0]));
        const bodyOps = topologicalSortOperationNodes(loopBody);
        const bodyConstants = loopBody.getConstantNodes().toArray();
        const captures = loopBody.getNodes().filterIs(RegionArgumentNode).toArray();

        for (let it = 0; it < tripCount; it++) {
            const valMap = new Map<string, ValueNode>();
            valMap.set(
                bodyIter.id,
                builder.createConstant(`unroll_${loopOp.id}_i${it}`, scalarInt64(it)),
            );
            valMap.set(bodyCondIn.id, curCond);
            bodyCarryIns.forEach((bIn, k) => {
                valMap.set(bIn.id, curCarries[k]);
            });

            for (const cap of captures) {
                const resolved = graph.hasNode(cap.originalName)
                    ? (graph.getNodeById(cap.originalName) as ValueNode)
                    : graph.hasNode(cap.id)
                      ? (graph.getNodeById(cap.id) as ValueNode)
                      : undefined;
                if (resolved) valMap.set(cap.id, resolved);
            }

            for (const cNode of bodyConstants) {
                const proto = {
                    ...cNode.constantValue,
                    dims: [...(cNode.constantValue.dims ?? [])],
                };
                const clonedC = builder.createConstant(`u_${loopOp.id}_i${it}_${cNode.id}`, proto);
                valMap.set(cNode.id, clonedC);
            }

            for (const innerOp of bodyOps) {
                const origOuts = innerOp.getOutputs();
                if (
                    innerOp.type === "Identity" &&
                    origOuts.length === 1 &&
                    origOuts[0].is(TensorNode) &&
                    origOuts[0].as(TensorNode).type === "output"
                ) {
                    const src = innerOp.getInputs()?.[0];
                    if (src) valMap.set(origOuts[0].id, valMap.get(src.id) ?? src);
                    continue;
                }

                const mappedIns = (innerOp.getInputs() ?? []).map((inp) => {
                    const m = valMap.get(inp.id);
                    if (m) return m;
                    if (graph.hasNode(inp.id)) return graph.getNodeById(inp.id) as ValueNode;
                    return inp;
                });

                const expectedOuts = origOuts.map((o) => ({
                    type: o.literalType,
                    shape: Array.isArray(o.shape) ? toStaticShape(o.shape) : [],
                }));
                const clonedOuts = builder.createOp(
                    innerOp.type,
                    mappedIns,
                    { ...innerOp.attributes },
                    expectedOuts,
                );
                origOuts.forEach((origO, idx) => {
                    if (idx < clonedOuts.length) {
                        valMap.set(origO.id, clonedOuts[idx]);
                    }
                });
            }

            curCond = valMap.get(condOutSource.id) ?? condOutSource;
            curCarries = carryOutSources.map((src, k) => valMap.get(src.id) ?? curCarries[k]);
            scanOutSources.forEach((src, j) => {
                const stepVal = (valMap.get(src.id) ?? src) as ConcreteValueNode;
                const stepShape = toStaticShape(stepVal.shape);
                const [unsq] = builder.createOp("Unsqueeze", [stepVal, ax0], {}, [
                    { type: stepVal.literalType, shape: [1, ...stepShape] },
                ]);
                scanSteps[j].push(unsq);
            });
        }

        const finalOuterOutputs: ValueNode[] = [...curCarries];
        for (let j = 0; j < numScans; j++) {
            const steps = scanSteps[j];
            if (steps.length === 1) {
                finalOuterOutputs.push(steps[0]);
            } else {
                const elemTy = outerOutputs[numCarries + j]?.literalType ?? steps[0].literalType;
                const rawTargetShape =
                    outerOutputs.length > numCarries + j
                        ? outerOutputs[numCarries + j]?.shape
                        : undefined;
                const targetShape: KnownShape =
                    rawTargetShape !== undefined
                        ? toStaticShape(rawTargetShape)
                        : [tripCount, ...toStaticShape(steps[0].shape).slice(1)];
                const [stacked] = builder.createOp("Concat", steps, { axis: 0 }, [
                    { type: elemTy, shape: targetShape },
                ]);
                finalOuterOutputs.push(stacked);
            }
        }

        for (let idx = 0; idx < outerOutputs.length; idx++) {
            const oldOut = outerOutputs[idx];
            const newOut = finalOuterOutputs.length > idx ? finalOuterOutputs[idx] : undefined;
            if (newOut !== undefined && oldOut.id !== newOut.id) {
                builder.replaceAllUsesWith(oldOut, newOut);
            }
        }
        builder.removeNode(loopOp);
    }
}
