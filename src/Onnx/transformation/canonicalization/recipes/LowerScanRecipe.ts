import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../../OnnxGraph.js";
import OperationNode from "../../../OperationNode.js";
import TensorNode from "../../../TensorNode.js";
import ConstantNode from "../../../ConstantNode.js";
import OnnxEdge from "../../../OnnxEdge.js";
import { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape, ValueNode } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import {
    asConcreteValueNode,
    bool,
    getIntAttr,
    getIntsAttr,
    int64Vec,
    normalizeAxis,
    scalarInt64,
    toStaticShape,
    topologicalSortOperationNodes,
    uniq,
} from "../../../Utils.js";
import { createCapturedInput } from "../../loop-lowering/RecipeUtils.js";

export class LowerScanRecipe implements DecompositionRecipe {
    public readonly name = "LowerScan";
    public readonly targetOp = "Scan";
    public readonly exposesControlFlow = true;
    public readonly exposesDataAccess = true;
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["loop"];
    public readonly producedOps = [
        "Loop",
        "Shape",
        "Gather",
        "Sub",
        "Identity",
        "Transpose",
        "Slice",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "Scan") return false;
        const numScanInputs = getIntAttr(op, "num_scan_inputs", 0);
        const ins = op.getInputs();
        return numScanInputs > 0 && op.regions.length > 0 && !!ins && ins.length >= numScanInputs;
    }

    apply(op: OperationNode.Class, outerBuilder: GraphBuilder): void {
        const outerGraph = outerBuilder.graph;
        const ins = op.getInputs() as ConcreteValueNode[];
        const outs = op.getOutputs();
        const scanBody = op.regions[0];

        const numScanInputs = getIntAttr(op, "num_scan_inputs", 1);
        const numStates = ins.length - numScanInputs;
        const numScanOutputs = outs.length - numStates;

        const inAxesRaw = getIntsAttr(
            op,
            "scan_input_axes",
            new Array<number>(numScanInputs).fill(0),
        );
        const inDirsRaw = getIntsAttr(
            op,
            "scan_input_directions",
            new Array<number>(numScanInputs).fill(0),
        );
        const outAxesRaw = getIntsAttr(
            op,
            "scan_output_axes",
            new Array<number>(numScanOutputs).fill(0),
        );
        const outDirsRaw = getIntsAttr(
            op,
            "scan_output_directions",
            new Array<number>(numScanOutputs).fill(0),
        );

        const initialStates = ins.slice(0, numStates);
        const scanInputs = ins.slice(numStates);

        // 1. Determine trip count from the first scan input's scan axis
        const firstScan = scanInputs[0];
        const firstShape = toStaticShape(firstScan.shape);
        const firstAxis = normalizeAxis(inAxesRaw[0] ?? 0, firstShape.length);
        const staticSeqLen = firstShape[firstAxis];

        let tripCountNode: ValueNode;
        if (staticSeqLen > 0) {
            tripCountNode = outerBuilder.createConstant(
                `scan_trip_${op.id}`,
                scalarInt64(staticSeqLen),
            );
        } else {
            const [sNode] = outerBuilder.createOp("Shape", [firstScan]);
            const axConst = outerBuilder.createConstant(
                `scan_trip_ax_${op.id}`,
                scalarInt64(firstAxis),
            );
            [tripCountNode] = outerBuilder.createOp("Gather", [sNode, axConst], { axis: 0 }, [
                { type: DataType.INT64, shape: [] },
            ]);
        }

        const condInit = outerBuilder.createConstant(`scan_cond_${op.id}`, bool(true));

        // 2. Build the ONNX Loop body graph
        const loopBody = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const innerBuilder = new GraphBuilder(loopBody, `scan_body_${op.id}`);

        const iterNode = loopBody
            .addNode(uniq(loopBody, "iter"))
            .init(new TensorNode.Builder(DataType.INT64, [], "input"))
            .as(TensorNode);
        const condIn = loopBody
            .addNode(uniq(loopBody, "cond_in"))
            .init(new TensorNode.Builder(DataType.BOOL, [], "input"))
            .as(TensorNode);

        // Map original Scan body formal inputs -> Loop body nodes
        const scanBodyInputs = scanBody
            .getTensorNodes()
            .toArray()
            .filter((t) => t.type === "input");
        const scanBodyOutputs = scanBody
            .getTensorNodes()
            .toArray()
            .filter((t) => t.type === "output");

        const valueMap = new Map<string, ValueNode>();

        // 2a. Loop-carried state inputs
        for (let i = 0; i < numStates; i++) {
            const initSt = initialStates[i];
            const loopStateIn = loopBody
                .addNode(uniq(loopBody, `state_in_${i}`))
                .init(
                    new TensorNode.Builder(initSt.literalType, initSt.shape as KnownShape, "input"),
                )
                .as(TensorNode);
            if (scanBodyInputs.length > i) {
                valueMap.set(scanBodyInputs[i].id, loopStateIn);
            }
        }

        // 2b. Slice each scan input at `iter` (or `tripCount - 1 - iter` if reverse)
        let tripInner: ValueNode | undefined;
        let oneInner: ValueNode | undefined;

        for (let j = 0; j < numScanInputs; j++) {
            const outerScanIn = scanInputs[j];
            const captured = createCapturedInput(loopBody, outerScanIn);
            const scanRank = outerScanIn.shape.length;
            const axis = normalizeAxis(inAxesRaw[j] ?? 0, scanRank);
            const isReverse = (inDirsRaw[j] ?? 0) === 1;

            let idx: ConcreteValueNode = iterNode;
            if (isReverse) {
                if (!tripInner) {
                    tripInner =
                        staticSeqLen > 0
                            ? innerBuilder.createConstant(
                                  `scan_len_${op.id}`,
                                  scalarInt64(staticSeqLen),
                              )
                            : createCapturedInput(loopBody, tripCountNode);
                }
                if (!oneInner) {
                    oneInner = innerBuilder.createConstant(`scan_one_${op.id}`, scalarInt64(1));
                }
                const [lenMinus1] = innerBuilder.createOp("Sub", [tripInner, oneInner], {}, [
                    { type: DataType.INT64, shape: [] },
                ]);
                [idx] = innerBuilder.createOp("Sub", [lenMinus1, iterNode], {}, [
                    { type: DataType.INT64, shape: [] },
                ]);
            }

            const elemShape = [...toStaticShape(outerScanIn.shape)];
            elemShape.splice(axis, 1);

            const [gathered] = innerBuilder.createOp("Gather", [captured, idx], { axis }, [
                { type: outerScanIn.literalType, shape: elemShape as KnownShape },
            ]);

            if (scanBodyInputs.length > numStates + j) {
                valueMap.set(scanBodyInputs[numStates + j].id, gathered);
            }
        }

        // 2c. Clone constants and operations from Scan body into Loop body
        for (const cNode of scanBody.nodes.filterIs(ConstantNode).toArray()) {
            const clonedC = innerBuilder.createConstant(cNode.id, cNode.constantValue);
            valueMap.set(cNode.id, clonedC);
        }

        for (const bodyOp of topologicalSortOperationNodes(scanBody)) {
            const mappedIns = (bodyOp.getInputs() ?? []).map((inp) => {
                const existing = valueMap.get(inp.id);
                if (existing) return existing;
                return createCapturedInput(loopBody, inp as ValueNode);
            });
            const bodyOuts = bodyOp.getOutputs();
            const expectedOuts = bodyOuts.map((o) => ({
                type: o.literalType,
                shape: o.shape as KnownShape,
            }));

            const createdOuts = innerBuilder.createOp(
                bodyOp.type,
                mappedIns,
                bodyOp.getAttributes(),
                expectedOuts,
            );
            bodyOuts.forEach((origOut, idx) => {
                if (createdOuts.length > idx) {
                    valueMap.set(origOut.id, createdOuts[idx]);
                }
            });
        }

        // 2d. Wire Loop body outputs: [cond_out, ...state_outs, ...scan_outs]
        const emitBodyOutput = (src: ValueNode, name: string) => {
            const idOp = loopBody
                .addNode(uniq(loopBody, `id_${name}`))
                .init(new OperationNode.Builder("Identity", [src]))
                .as(OperationNode);
            const outT = loopBody
                .addNode(uniq(loopBody, name))
                .init(new TensorNode.Builder(src.literalType, src.shape as KnownShape, "output"))
                .as(TensorNode);
            loopBody.addEdge(src, idOp).init(new OnnxEdge.Builder(src.literalType, src.shape));
            loopBody.addEdge(idOp, outT).init(new OnnxEdge.Builder(outT.literalType, outT.shape));
        };

        emitBodyOutput(condIn, "cond_out");
        for (let i = 0; i < scanBodyOutputs.length; i++) {
            const mapped = valueMap.get(scanBodyOutputs[i].id) ?? scanBodyOutputs[i];
            emitBodyOutput(mapped, i < numStates ? `state_out_${i}` : `scan_out_${i - numStates}`);
        }

        // 3. Emit outer Loop node
        const loopInputs: ValueNode[] = [tripCountNode, condInit, ...initialStates];
        const loopOp = outerBuilder
            .addNode(
                uniq(outerGraph, `ScanLoop_${op.id}`),
                new OperationNode.Builder("Loop", loopInputs, {}, [loopBody]),
            )
            .as(OperationNode);

        for (const inp of loopInputs) {
            if (outerGraph.hasNode(inp.id)) {
                outerBuilder.addEdge(inp, loopOp, inp.literalType, inp.shape);
            }
        }

        // 4. Wire Loop outputs back to Scan outputs (with optional axis transpose / reverse)
        for (let i = 0; i < outs.length; i++) {
            const origOut = asConcreteValueNode(outs[i]);
            const origShape = Array.isArray(origOut.shape) ? origOut.shape : [];
            const isScanOut = i >= numStates;
            const scanIdx = i - numStates;
            const outAxis = isScanOut
                ? normalizeAxis(outAxesRaw[scanIdx] ?? 0, origShape.length)
                : 0;
            const isRevOut = isScanOut && (outDirsRaw[scanIdx] ?? 0) === 1;

            // Loop always stacks scan outputs along axis 0
            const rawShape = [...toStaticShape(origShape)];
            if (isScanOut && outAxis !== 0 && rawShape.length > outAxis) {
                const seqDim = rawShape.splice(outAxis, 1)[0];
                rawShape.unshift(seqDim);
            }

            const loopOutTensor = outerBuilder
                .addNode(
                    uniq(outerGraph, `scan_loop_out_${op.id}_${i}`),
                    new TensorNode.Builder(
                        origOut.literalType,
                        rawShape as KnownShape,
                        "intermediate",
                    ),
                )
                .as(TensorNode);
            outerBuilder.addEdge(
                loopOp,
                loopOutTensor,
                origOut.literalType,
                rawShape as KnownShape,
            );

            let finalOut: ConcreteValueNode = loopOutTensor;

            if (isRevOut && staticSeqLen > 0) {
                const revStart = outerBuilder.createConstant(
                    `scan_rev_s_${op.id}_${i}`,
                    int64Vec([staticSeqLen - 1]),
                );
                const revEnd = outerBuilder.createConstant(
                    `scan_rev_e_${op.id}_${i}`,
                    int64Vec([-staticSeqLen - 1]),
                );
                const revAx = outerBuilder.createConstant(
                    `scan_rev_a_${op.id}_${i}`,
                    int64Vec([0]),
                );
                const revStep = outerBuilder.createConstant(
                    `scan_rev_st_${op.id}_${i}`,
                    int64Vec([-1]),
                );
                [finalOut] = outerBuilder.createOp(
                    "Slice",
                    [finalOut, revStart, revEnd, revAx, revStep],
                    {},
                    [{ type: origOut.literalType, shape: rawShape as KnownShape }],
                );
            }

            if (isScanOut && outAxis !== 0 && rawShape.length > 1) {
                // Move axis 0 to `outAxis`
                const perm: number[] = [];
                for (let k = 1; k <= outAxis; k++) perm.push(k);
                perm.push(0);
                for (let k = outAxis + 1; k < rawShape.length; k++) perm.push(k);
                [finalOut] = outerBuilder.createOp("Transpose", [finalOut], { perm }, [
                    { type: origOut.literalType, shape: origOut.shape as KnownShape },
                ]);
            }

            outerBuilder.replaceAllUsesWith(origOut, finalOut);
        }

        outerBuilder.removeNode(op);
    }
}
