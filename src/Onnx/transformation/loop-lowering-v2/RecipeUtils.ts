import type OnnxGraph from "../../OnnxGraph.js";
import type { ValueNode, ConcreteValueNode, KnownShape, Shape } from "../../OnnxTypes.js";
import type BaseNode from "@specs-feup/flow/graph/BaseNode";
import ConstantNode from "../../ConstantNode.js";
import TensorNode from "../../TensorNode.js";
import RegionArgumentNode from "../../RegionArgumentNode.js";
import { ensureFlatInput, createCapturedInput, squeezeIfLen1, decodeMixedRadix, buildLinearIndex, unsqueezeIdx, gatherFrom } from "../loop-lowering/BuildLoop.js";
import { toStaticShape, makeTensorConst, scalarInt64, computeStrides } from "../../Utils.js";

export function resolveRecipeInput(
    body: OnnxGraph.Class,
    input: BaseNode.Class,
    valueMap: Map<string, ValueNode>,
    iter: ConcreteValueNode,
    axes: ConcreteValueNode,
    outShape: KnownShape,
    flatten: boolean = true,
    returnGather: boolean = true
): ValueNode {
    // 1. Deforestation Check! (This now works because valueMap keys are fixed)
    if (valueMap.has(input.id)) {
        return valueMap.get(input.id)!; 
    }

    // 2. Identify and Capture the Outer Node
    let tOuter: ValueNode;
    if (input.is(TensorNode)) tOuter = input.as(TensorNode);
    else if (input.is(ConstantNode)) tOuter = input.as(ConstantNode);
    else if (input.is(RegionArgumentNode)) tOuter = input.as(RegionArgumentNode);
    else throw new Error(`Unhandled input case for ${input.id}`);

    let tInner: ValueNode;
    if (body.hasNode(tOuter.id)) {
        tInner = body.getNodeById(tOuter.id) as ValueNode;
    } else {
        if (tOuter.is(ConstantNode)) {
            const c = tOuter.as(ConstantNode);
            tInner = body.addNode(c.id).init(new ConstantNode.Builder(c.constantValue, c.isInput)).as(ConstantNode);
        } else {
            tInner = createCapturedInput(body, tOuter);
        }
    }

    if (!returnGather || tInner.shape.length === 0) {
        return flatten ? ensureFlatInput(body, tInner) : tInner;
    }

    // 3. Smart Caching Gather
    const inDimsStatic = toStaticShape(tInner.shape as Shape);
    const outDimsStatic = toStaticShape(outShape as Shape);
    
    // Create a unique key based on the shapes involved in this broadcast
    const shapeKey = `__idx_cache_${inDimsStatic.join(',')}_to_${outDimsStatic.join(',')}`;
    let linU: ConcreteValueNode;

    if (valueMap.has(shapeKey)) {
        // We already did the math! Reuse the linear index tensor.
        linU = valueMap.get(shapeKey) as ConcreteValueNode;
    } else {
        // Do the complex index decoding once and cache it!
        const rO = outDimsStatic.length;
        const rI = inDimsStatic.length;
        const outRadix = outDimsStatic.map(d => d > 0 ? d : 1);
        const inRadix = inDimsStatic.map(d => d > 0 ? d : 1);
        
        const oDigits = decodeMixedRadix(body, iter, outRadix, `gb_out_${shapeKey}`);
        const iDigits: any[] = [];
        
        for (let k = 0; k < rI; k++) {
            const inDim = inRadix[k];
            const outPos = rO - rI + k;
            if (outPos < 0 || inDim === 1) {
                iDigits.push(makeTensorConst(body, `gb_zero_${shapeKey}_${k}`, scalarInt64(0)));
            } else {
                iDigits.push(oDigits[outPos]);
            }
        }

        const strides = computeStrides(inRadix);
        const linScalar = buildLinearIndex(body, iDigits, strides, `gb_lin_${shapeKey}`);
        linU = unsqueezeIdx(body, linScalar, axes, `gb_linU_${shapeKey}`);
        
        valueMap.set(shapeKey, linU);
    }

    // 4. Perform the cheap 1D gather using the shared index
    const flatT = ensureFlatInput(body, tInner);
    const [, gathered] = gatherFrom(body, flatT, `gb_g_${tInner.id}`, linU, 0);
    return squeezeIfLen1(body, gathered, axes, `gb_sq_${tInner.id}`);
}