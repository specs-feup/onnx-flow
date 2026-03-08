import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import { OpCategory } from "../../../Schema/OpSchema.js";
import { OpRegistry } from "../../../Schema/OpRegistry.js";
import type { ConcreteValueNode } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { broadcastShapes, getIntAttr, makeTensorProto, toStaticShape } from "../../../Utils.js";
import { buildBroadcastGather } from "./RecipeUtils.js";

export class LowerElementWiseRecipe implements DecompositionRecipe {
    public readonly name = "LowerElementWise";
    public readonly targetOp = OpCategory.ElementWise;
    public readonly exposesControlFlow = true;
    public readonly exposesDataAccess = true;
    public readonly producedOps = ["Loop", "Gather", "ScatterElements"];

    canApply(op: OperationNode.Class): boolean {
        const schema = OpRegistry.getInstance().get(op.type, 19);
        if (schema?.category !== OpCategory.ElementWise) return false;

        const inputs = op.getInputs() ?? [];
        if (!inputs.some((input) => input.shape.length > 0)) return false;

        // Strict guard against dynamic shapes
        const outShape = toStaticShape(op.getOutputs()[0]?.shape);
        if (!Array.isArray(outShape)) return false;

        return outShape.every((d) => typeof d === "number" && !isNaN(d) && d > 0);
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const inputs = op.getInputs() as ConcreteValueNode[];
        const output = op.getOutputs()[0];

        const inShapes = inputs.map((i) => toStaticShape(i.shape));
        const outShape = broadcastShapes(...inShapes);

        let dtype = (output.literalType as unknown as DataType | undefined) ?? DataType.UNDEFINED;

        if (dtype === DataType.UNDEFINED && op.type === "Cast") {
            dtype = getIntAttr(op, "to", DataType.UNDEFINED);
        }

        if (dtype === DataType.UNDEFINED)
            dtype =
                (inputs[0].literalType as unknown as DataType | undefined) ?? DataType.UNDEFINED;
        if (dtype === DataType.UNDEFINED) dtype = DataType.FLOAT;

        // 1. Calculate trip count
        const totalElements = outShape.reduce((a, b) => a * b, 1);

        // 2. Generate Loop
        const { innerBuilder, trip, vInitial, loopOutput, finalize } = builder.createForLoopRegion(
            builder,
            totalElements,
            dtype,
            [totalElements],
            `Loop_${op.id}`,
        );

        // 3. INSIDE LOOP: Gather inputs
        const gatheredInputs = inputs.map((input, idx) => {
            return buildBroadcastGather(
                innerBuilder,
                input,
                trip,
                outShape,
                toStaticShape(input.shape),
                `gath_${op.id}_${idx}`,
            );
        });

        // 4. INSIDE LOOP: Perform Math
        const innerMathResult = innerBuilder.createOp(
            op.type,
            gatheredInputs,
            op.getAttributes(),
        )[0];

        // 5. INSIDE LOOP: ScatterElements into the Carry State
        const flatAxes = innerBuilder.createConstant(
            `flat_axes_${op.id}`,
            makeTensorProto(DataType.INT64, [1], [0]),
        );

        const iterIdx = innerBuilder.createOp("Unsqueeze", [trip, flatAxes])[0];
        const updateVal = innerBuilder.createOp("Unsqueeze", [innerMathResult, flatAxes])[0];

        const scatterOut = innerBuilder.createOp(
            "ScatterElements",
            [vInitial, iterIdx, updateVal],
            { axis: 0 },
        )[0];

        // 6. Finalize
        finalize([scatterOut]);

        // OUTER GRAPH: Reshape back to the original multi-dimensional tensor
        const origShapeConst = builder.createConstant(
            `origShape_${op.id}`,
            makeTensorProto(DataType.INT64, [outShape.length], outShape),
        );
        const finalReshape = builder.createOp("Reshape", [loopOutput, origShapeConst], {}, [
            { type: dtype, shape: outShape },
        ])[0];

        builder.replaceAllUsesWith(output, finalReshape);
        op.remove();
    }
}
