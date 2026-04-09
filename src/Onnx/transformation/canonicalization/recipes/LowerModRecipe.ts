import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { formatId } from "@specs-feup/onnx-flow/Onnx/Utils";
import { getIntAttr } from "../../../Utils.js";

export class LowerModRecipe implements DecompositionRecipe {
    public readonly name = "LowerMod";
    public readonly targetOp = "Mod";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = ["Add", "Div", "Floor", "Neg", "Mul"];

    canApply(op: OperationNode.Class): boolean {
        return op.type === "Mod";
    }

    apply(node: OperationNode.Class, builder: GraphBuilder): void {
        const ins = node.getInputs() as ConcreteValueNode[];
        const A = ins[0];
        const B = ins[1];
        const Y = node.getOutputs()[0];

        const outShape = (Y.shape as KnownShape | undefined) ?? (A.shape as KnownShape);
        const originalDtype = (A.literalType as DataType | undefined) ?? DataType.FLOAT;

        const fmod = getIntAttr(node, "fmod", 0);

        const isInt =
            originalDtype === DataType.INT64 ||
            originalDtype === DataType.INT32 ||
            originalDtype === DataType.INT16 ||
            originalDtype === DataType.INT8 ||
            originalDtype === DataType.UINT64 ||
            originalDtype === DataType.UINT32 ||
            originalDtype === DataType.UINT16 ||
            originalDtype === DataType.UINT8;

        const workDtype = isInt ? DataType.FLOAT : originalDtype;
        const workExpectedOut = [{ type: workDtype, shape: outShape }];

        const A_work = isInt ? builder.createOp("Cast", [A], { to: DataType.FLOAT }, workExpectedOut)[0] : A;
        const B_work = isInt ? builder.createOp("Cast", [B], { to: DataType.FLOAT }, workExpectedOut)[0] : B;

        const div = builder.createOp("Div", [A_work, B_work], {}, workExpectedOut)[0];

        let multiplier: ConcreteValueNode;

        if (fmod === 0) {
            // Módulo Matemático: A - B * floor(A / B)
            multiplier = builder.createOp("Floor", [div], {}, workExpectedOut)[0];
        } else {
            // Módulo C/C++: A - B * trunc(A / B)
            // Em ONNX fazemos: trunc(x) = sign(x) * floor(abs(x))
            const absDiv = builder.createOp("Abs", [div], {}, workExpectedOut)[0];
            const floorAbs = builder.createOp("Floor", [absDiv], {}, workExpectedOut)[0];
            const signDiv = builder.createOp("Sign", [div], {}, workExpectedOut)[0];
            
            multiplier = builder.createOp("Mul", [signDiv, floorAbs], {}, workExpectedOut)[0];
        }

        const mul = builder.createOp("Mul", [B_work, multiplier], {}, workExpectedOut)[0];

        const sub = builder.createOp("Sub", [A_work, mul], {}, workExpectedOut)[0];

        const expectedFinalOut = [{ type: originalDtype, shape: outShape }];
        const finalResult = isInt ? builder.createOp("Cast", [sub], { to: originalDtype }, expectedFinalOut)[0] : sub;

        builder.replaceAllUsesWith(Y, finalResult);
        node.remove();
    }
}