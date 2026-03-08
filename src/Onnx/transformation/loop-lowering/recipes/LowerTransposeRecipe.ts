import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, ValueNode } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { toStaticShape, getIntsAttr, makeTensorProto } from "../../../Utils.js";

export class LowerTransposeRecipe implements DecompositionRecipe {
    public readonly name = "LowerTranspose";
    public readonly targetOp = "Transpose";
    public readonly exposesControlFlow = true;
    public readonly exposesDataAccess = true;
    public readonly producedOps = ["Loop", "Gather", "ScatterElements"];

    canApply(op: OperationNode.Class): boolean {
        return op.type === "Transpose";
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const input = op.getInputs()![0] as ConcreteValueNode;
        const output = op.getOutputs()[0];

        const inShape = toStaticShape(input.shape);
        const rank = inShape.length;

        const defaultPerm = Array.from({ length: rank }, (_, i) => rank - 1 - i);
        const perm = getIntsAttr(op, "perm", defaultPerm);

        let outShape = toStaticShape(output.shape);
        if (outShape.length === 0) {
            outShape = new Array(rank);
            for (let i = 0; i < rank; i++) outShape[i] = inShape[perm[i]];
        }

        let dtype = (output.literalType as unknown as DataType | undefined) ?? DataType.UNDEFINED;
        if (dtype === DataType.UNDEFINED)
            dtype = (input.literalType as unknown as DataType | undefined) ?? DataType.UNDEFINED;
        if (dtype === DataType.UNDEFINED) dtype = DataType.FLOAT;

        const totalElements = outShape.reduce((a, b) => a * b, 1);
        const shapeConst = builder.createConstant(
            `shape_${op.id}`,
            makeTensorProto(DataType.INT64, [rank], outShape),
        );

        const { innerBuilder, trip, vInitial, loopOutput, finalize } = builder.createForLoopRegion(
            builder,
            totalElements,
            dtype,
            [totalElements],
            `TransposeLoop_${op.id}`,
        );

        const flatAxes = innerBuilder.createConstant(
            `axes_${op.id}`,
            makeTensorProto(DataType.INT64, [1], [0]),
        );

        let rem: ValueNode = trip;
        const outCoords: ConcreteValueNode[] = [];
        for (let i = rank - 1; i >= 0; i--) {
            const dimConst = innerBuilder.createConstant(
                `dim_${op.id}_${i}`,
                makeTensorProto(DataType.INT64, [], [outShape[i]]),
            );
            const coord = innerBuilder.createOp("Mod", [rem, dimConst])[0];
            outCoords.unshift(coord);
            rem = innerBuilder.createOp("Div", [rem, dimConst])[0];
        }

        const inCoords: ConcreteValueNode[] = new Array(rank);
        for (let i = 0; i < rank; i++) {
            inCoords[perm[i]] = outCoords[i];
        }

        let currentGather = input;
        for (let i = 0; i < rank; i++) {
            // By passing the scalar coordinate directly, ONNX completely drops the axis.
            currentGather = innerBuilder.createOp("Gather", [currentGather, inCoords[i]], {
                axis: 0,
            })[0];
        }

        // currentGather is now guaranteed to be a pure scalar (rank 0).
        const iterUnsq = innerBuilder.createOp("Unsqueeze", [trip, flatAxes])[0];
        const updateVal = innerBuilder.createOp("Unsqueeze", [currentGather, flatAxes])[0];

        const scatterOut = innerBuilder.createOp(
            "ScatterElements",
            [vInitial, iterUnsq, updateVal],
            { axis: 0 },
        )[0];

        finalize([scatterOut]);

        // Reshape happens on the OUTER graph
        const finalReshape = builder.createOp("Reshape", [loopOutput, shapeConst], {}, [
            { type: dtype, shape: outShape },
        ])[0];
        builder.replaceAllUsesWith(output, finalReshape);
        op.remove();
    }
}
