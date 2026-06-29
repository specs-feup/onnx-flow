import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, ValueNode } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { readConstIntegerVectorFromTensorNode, makeTensorProto } from "../../../Utils.js";
import ConstantNode from "../../../ConstantNode.js";
import TensorNode from "../../../TensorNode.js";
import { TransformationOpportunity } from "../../TransformationOpportunity.js";

export class LowerSliceRecipe implements DecompositionRecipe {
    public readonly name = "LowerSlice";
    public readonly targetOp = "Slice";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = true;
    public readonly producedOps = ["Range", "Gather", "Identity"];

    match(op: OperationNode.Class): TransformationOpportunity | null {
        if (op.type !== "Slice") return null;
        const ins = op.getInputs() ?? [];
        if (ins.length < 3) return null;

        // We only lower if starts and ends are statically known constants
        const readVec = (t: ValueNode) => {
            if (t.is(ConstantNode)) return readConstIntegerVectorFromTensorNode(t.as(ConstantNode));
            if (t.is(TensorNode)) return readConstIntegerVectorFromTensorNode(t.as(TensorNode));
            return undefined;
        };
        const starts = readVec(ins[1]);
        const ends = readVec(ins[2]);
        if (starts === undefined || ends === undefined) return null;
        return new TransformationOpportunity(
            this.name,
            op.id,
            "Lower Slice",
            (builder: GraphBuilder) => this.apply(op, builder),
        );
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const X = ins[0];
        const Y = op.getOutputs()[0];

        const inShape = X.shape.map((d) => (typeof d === "number" ? d : 1));
        const rank = inShape.length;

        const readVec = (idx: number) => {
            if (!(idx in ins)) return undefined;
            const t: ConcreteValueNode = ins[idx];
            if (t.is(ConstantNode)) return readConstIntegerVectorFromTensorNode(t.as(ConstantNode));
            if (t.is(TensorNode)) return readConstIntegerVectorFromTensorNode(t.as(TensorNode));
            return undefined;
        };

        const starts = readVec(1)!;
        const ends = readVec(2)!;
        const axes = readVec(3) ?? Array.from({ length: starts.length }, (_, i) => i);
        const steps = readVec(4) ?? new Array(axes.length).fill(1);

        const fullStarts = new Array(rank).fill(0);
        const fullEnds = inShape.slice();
        const fullSteps = new Array(rank).fill(1);

        for (let i = 0; i < axes.length; i++) {
            const ax = axes[i];
            if (ax < 0 || ax >= rank) continue;

            const dimVal = inShape[ax];
            let s = Number(starts[i]);
            let e = Number(ends[i]);
            const stp = Number(steps[i]);

            if (s < 0) s += dimVal;
            if (e < 0) e += dimVal;

            if (stp > 0) {
                s = Math.max(0, Math.min(s, dimVal));
                e = Math.max(0, Math.min(e, dimVal));
            } else {
                s = Math.min(dimVal - 1, Math.max(s, 0));
                e = Math.min(dimVal - 1, Math.max(e, -1));
            }

            fullStarts[ax] = s;
            fullEnds[ax] = e;
            fullSteps[ax] = stp;
        }

        const changingAxes: number[] = [];
        for (let ax = 0; ax < rank; ax++) {
            if (!(fullStarts[ax] === 0 && fullEnds[ax] === inShape[ax] && fullSteps[ax] === 1)) {
                changingAxes.push(ax);
            }
        }

        if (changingAxes.length === 0) {
            const id = builder.createOp("Identity", [X])[0];
            builder.replaceAllUsesWith(Y, id);
        } else {
            let curT = X;
            for (let i = 0; i < changingAxes.length; i++) {
                const ax = changingAxes[i];
                const cS = builder.createConstant(
                    `Slice_S_${op.id}_${ax}`,
                    makeTensorProto(DataType.INT64, [], [fullStarts[ax]]),
                );
                const cE = builder.createConstant(
                    `Slice_E_${op.id}_${ax}`,
                    makeTensorProto(DataType.INT64, [], [fullEnds[ax]]),
                );
                const cStep = builder.createConstant(
                    `Slice_Step_${op.id}_${ax}`,
                    makeTensorProto(DataType.INT64, [], [fullSteps[ax]]),
                );

                const range = builder.createOp("Range", [cS, cE, cStep])[0];
                curT = builder.createOp("Gather", [curT, range], { axis: ax })[0];
            }
            builder.replaceAllUsesWith(Y, curT);
        }

        op.remove();
    }
}
