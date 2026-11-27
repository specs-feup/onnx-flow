import type OnnxGraph from "../../ir/OnnxGraph.js";
import OperationNode from "../../ir/OperationNode.js";
import ConstantNode from "../../ir/ConstantNode.js";
import TensorNode from "../../ir/TensorNode.js";
import { GraphBuilder } from "../../ir/GraphBuilder.js";
import { DataType } from "../../ir/OnnxTypes.js";
import {
    getFloatAttr,
    makeTensorProto,
    readTensorData,
    topologicalSortOperationNodes,
    toStaticShape,
} from "../../utils/index.js";
import { BaseGraphPass } from "../GraphPass.js";
import {
    asRewritableIntermediate,
    cleanupOrphanConstant,
    collectCapturedIds,
} from "./OptimizationUtils.js";

/**
 * Folds inference `BatchNormalization` into an immediately preceding `Conv` node
 * when `Conv`'s weights (and optional bias) and all 4 BN parameters (`scale`, `B`,
 * `mean`, `var`) are constant initializers.
 *
 * Eliminates `BatchNormalization` (and its `Sqrt` approximation under strict scalar
 * CGRA profiles) with exact mathematical equivalence:
 * `W'[c] = W[c] * scale[c] / sqrt(var[c] + eps)`
 * `B'[c] = (B[c] - mean[c]) * scale[c] / sqrt(var[c] + eps) + bnBias[c]`
 */
export class ConvBnFoldingPass extends BaseGraphPass {
    public override readonly name = "ConvBnFolding";

    public override run(graph: OnnxGraph.Class): boolean {
        let changed = this.runOnRegions(graph);
        const capturedIds = collectCapturedIds(graph);

        for (const bnOp of topologicalSortOperationNodes(graph)) {
            if (!graph.hasNode(bnOp.id) || bnOp.type !== "BatchNormalization") continue;
            if (this.tryFoldConvBn(bnOp, graph, capturedIds)) {
                changed = true;
            }
        }

        return changed;
    }

    private tryFoldConvBn(
        bnOp: OperationNode.Class,
        graph: OnnxGraph.Class,
        capturedIds: Set<string>,
    ): boolean {
        // Inference-mode BN only (single output Y, 5 inputs: X, scale, B, mean, var)
        const bnOutputs = bnOp.getOutputs();
        const bnInputs = bnOp.getInputs();
        if (bnOutputs.length !== 1 || !bnInputs || bnInputs.length !== 5) return false;

        const [bnX, scaleNode, bnBiasNode, meanNode, varNode] = bnInputs;
        const convOut = asRewritableIntermediate(bnX, capturedIds);
        if (convOut === undefined) return false;

        // Conv output must feed ONLY this BatchNormalization node
        const convConsumers = convOut.outgoers.targets.filterIs(OperationNode).toArray();
        if (convConsumers.length !== 1 || convConsumers[0].id !== bnOp.id) return false;

        const convOp = convOut.incomers.sources.filterIs(OperationNode).first();
        if (convOp === undefined || convOp.type !== "Conv") return false;

        const convInputs = convOp.getInputs();
        if (!convInputs || convInputs.length < 2) return false;
        const convData = convInputs[0];
        const weightNode = convInputs[1];
        const convBiasNode = convInputs.length >= 3 ? convInputs[2] : undefined;

        if (
            !weightNode.is(ConstantNode) ||
            !scaleNode.is(ConstantNode) ||
            !bnBiasNode.is(ConstantNode) ||
            !meanNode.is(ConstantNode) ||
            !varNode.is(ConstantNode)
        ) {
            return false;
        }
        if (convBiasNode !== undefined && !convBiasNode.is(ConstantNode)) {
            return false;
        }

        const dtype = weightNode.literalType;
        if (dtype !== DataType.FLOAT && dtype !== DataType.DOUBLE) return false;

        const wShape = toStaticShape(weightNode.shape);
        if (wShape.length < 3 || wShape.includes(-1)) return false;
        const outChannels = wShape[0];
        const perChannelElems = wShape.slice(1).reduce((a, b) => a * b, 1);

        const wVals = readTensorData(weightNode);
        const scaleVals = readTensorData(scaleNode);
        const bnBiasVals = readTensorData(bnBiasNode);
        const meanVals = readTensorData(meanNode);
        const varVals = readTensorData(varNode);
        const convBiasVals = convBiasNode !== undefined ? readTensorData(convBiasNode) : undefined;

        if (
            wVals === undefined ||
            scaleVals === undefined ||
            bnBiasVals === undefined ||
            meanVals === undefined ||
            varVals === undefined ||
            wVals.length !== outChannels * perChannelElems ||
            scaleVals.length !== outChannels ||
            bnBiasVals.length !== outChannels ||
            meanVals.length !== outChannels ||
            varVals.length !== outChannels
        ) {
            return false;
        }
        if (
            convBiasNode !== undefined &&
            (convBiasVals === undefined || convBiasVals.length !== outChannels)
        ) {
            return false;
        }

        const epsilon = getFloatAttr(bnOp, "epsilon", 1e-5);
        const foldedW = new Array<number>(wVals.length);
        const foldedB = new Array<number>(outChannels);

        for (let c = 0; c < outChannels; c++) {
            const denomSq = varVals[c] + epsilon;
            if (!(denomSq > 0) || !Number.isFinite(denomSq)) return false;
            const factor = scaleVals[c] / Math.sqrt(denomSq);
            const origB = convBiasVals !== undefined ? convBiasVals[c] : 0;
            foldedB[c] = (origB - meanVals[c]) * factor + bnBiasVals[c];

            const base = c * perChannelElems;
            for (let k = 0; k < perChannelElems; k++) {
                foldedW[base + k] = wVals[base + k] * factor;
            }
        }

        const builder = new GraphBuilder(graph, `conv_bn_${convOp.id}`);
        const newW = builder.createConstant(
            `w_folded_${convOp.id}`,
            makeTensorProto(dtype, wShape, foldedW),
        );
        const newB = builder.createConstant(
            `b_folded_${convOp.id}`,
            makeTensorProto(dtype, [outChannels], foldedB),
        );

        // Detach old W / B edges from Conv and attach folded W / B
        for (const e of weightNode.outgoers.toArray()) {
            if (e.target.id === convOp.id) e.remove();
        }
        if (convBiasNode !== undefined) {
            for (const e of convBiasNode.outgoers.toArray()) {
                if (e.target.id === convOp.id) e.remove();
            }
        }

        convOp.setInputs([convData, newW, newB]);
        builder.addEdge(newW, convOp, dtype, wShape);
        builder.addEdge(newB, convOp, dtype, [outChannels]);

        // Rewire BN output uses to Conv's output tensor and remove BN
        const bnOut = bnOutputs[0];
        const edgeConvToBn = graph.getEdge(convOut.id, bnOp.id);
        if (edgeConvToBn) edgeConvToBn.remove();

        if (bnOut.is(TensorNode) && bnOut.as(TensorNode).type === "output") {
            const bnOutTensor = bnOut.as(TensorNode);
            const bnOutId = bnOutTensor.id;
            const convToOutEdge = graph.getEdge(convOp.id, convOut.id);
            if (convToOutEdge) convToOutEdge.remove();
            const bnToOutEdge = graph.getEdge(bnOp.id, bnOutId);
            if (bnToOutEdge) bnToOutEdge.remove();
            bnOp.remove();
            if (graph.hasNode(convOut.id)) convOut.remove();
            builder.addEdge(convOp, bnOutTensor, bnOutTensor.literalType, bnOutTensor.shape);
        } else {
            builder.replaceAllUsesWith(bnOut, convOut);
            bnOp.remove();
            if (graph.hasNode(bnOut.id) && bnOut.outgoers.length === 0) {
                bnOut.remove();
            }
        }

        // Clean up orphaned original Conv/BN constant parameters
        cleanupOrphanConstant(weightNode, graph, capturedIds);
        cleanupOrphanConstant(convBiasNode, graph, capturedIds);
        cleanupOrphanConstant(scaleNode, graph, capturedIds);
        cleanupOrphanConstant(bnBiasNode, graph, capturedIds);
        cleanupOrphanConstant(meanNode, graph, capturedIds);
        cleanupOrphanConstant(varNode, graph, capturedIds);

        return true;
    }
}
