import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../../OnnxGraph.js";
import OperationNode from "../../../OperationNode.js";
import TensorNode from "../../../TensorNode.js";
import { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape, ValueNode } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import {
    bool,
    getIntAttr,
    getIntsAttr,
    getStringAttr,
    int64Vec,
    makeTensorProto,
    scalarInt64,
    toStaticShape,
    uniq,
    zeroTensor,
} from "../../../Utils.js";
import {
    buildLinearIndex,
    buildMixedStrides,
    createCapturedInput,
    decodeMixedRadix,
    ensureFlatInput,
    extractMixedDims,
    mulMixed,
    packShapeVector,
    resolvePoolShapeAndPadsMixed,
    squeezeIfLen1,
} from "../../loop-lowering/RecipeUtils.js";
import { emitBodyOutput } from "../RnnLoopUtils.js";

export class LowerMaxPoolWithIndicesRecipe implements DecompositionRecipe {
    public readonly name = "LowerMaxPoolWithIndices";
    public readonly targetOp = "MaxPool";
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 10;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["pooling", "multi-output", "loop"] as const;
    public readonly exposesControlFlow = true;
    public readonly exposesDataAccess = true;
    public readonly producedOps = [
        "Loop",
        "Shape",
        "Concat",
        "ReduceProd",
        "Max",
        "Expand",
        "Reshape",
        "Gather",
        "ScatterElements",
        "Unsqueeze",
        "Squeeze",
        "Add",
        "Sub",
        "Mul",
        "Div",
        "Mod",
        "Equal",
        "Less",
        "Greater",
        "And",
        "Or",
        "Not",
        "Where",
        "Identity",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "MaxPool") return false;
        if (op.getOutputs().length < 2) return false;
        const X = op.getInputs()?.[0];
        if (!X || !Array.isArray(X.shape) || X.shape.length < 3) return false;
        const dims = toStaticShape(X.shape);

        const kernel = getIntsAttr(op, "kernel_shape", []);
        if (kernel.length !== dims.length - 2) return false;

        const storageOrder = getIntAttr(op, "storage_order", 0);
        if (storageOrder === 1 && dims.length !== 4) return false;
        return storageOrder === 0 || storageOrder === 1;
    }

    private resolveMixedPadsAndOutSpatial(
        op: OperationNode.Class,
        builder: GraphBuilder,
        inDims: (number | ConcreteValueNode)[],
        tag: string,
    ): { pHeads: (number | ConcreteValueNode)[]; outSpatial: (number | ConcreteValueNode)[] } {
        const spatialRank = inDims.length - 2;
        const kernel = getIntsAttr(op, "kernel_shape", []);
        const strides = getIntsAttr(op, "strides", new Array<number>(spatialRank).fill(1));
        const dilations = getIntsAttr(op, "dilations", new Array<number>(spatialRank).fill(1));
        const autoPad = getStringAttr(op, "auto_pad", "NOTSET");
        const ceilMode = getIntAttr(op, "ceil_mode", 0);
        const explicitPads = getIntsAttr(op, "pads", []);

        return resolvePoolShapeAndPadsMixed(
            autoPad,
            inDims.slice(2),
            kernel,
            strides,
            dilations,
            explicitPads,
            ceilMode,
            builder,
            tag,
        );
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const outerGraph = builder.graph;
        const X = op.getInputs()![0] as ConcreteValueNode;
        const outputs = op.getOutputs();
        const YOut = outputs[0];
        const IdxOut = outputs[1];

        const rawInDims = toStaticShape(X.shape);
        const isDynamic = rawInDims.includes(-1);
        const dtype = X.literalType as DataType;
        const spatialRank = rawInDims.length - 2;
        const storageOrder = getIntAttr(op, "storage_order", 0);

        const kernel = getIntsAttr(op, "kernel_shape", []);
        const strides = getIntsAttr(op, "strides", new Array<number>(spatialRank).fill(1));
        const dilations = getIntsAttr(op, "dilations", new Array<number>(spatialRank).fill(1));
        const kTotal = kernel.reduce((a, b) => a * b, 1);

        const outerInDims = extractMixedDims(builder, X, rawInDims, `mpib_${op.id}`);
        const { outSpatial: outerOutSpatial } = this.resolveMixedPadsAndOutSpatial(
            op,
            builder,
            outerInDims,
            `mpib_${op.id}`,
        );
        const outerOutDims = [outerInDims[0], outerInDims[1], ...outerOutSpatial];
        const knownOutShape = outerOutDims.map((d) =>
            typeof d === "number" ? d : -1,
        ) as KnownShape;

        let tripCount: ConcreteValueNode;
        let valCarry0: ConcreteValueNode;
        let idxCarry0: ConcreteValueNode;
        let carryKnownShape: KnownShape;
        let outShapeNode: ConcreteValueNode;

        if (!isDynamic) {
            const staticOut = outerOutDims as number[];
            const outTotal = staticOut.reduce((a, b) => a * b, 1);
            tripCount = builder.createConstant(`mpi_trip_${op.id}`, scalarInt64(outTotal * kTotal));
            valCarry0 = builder.createConstant(`mpi_val0_${op.id}`, zeroTensor(dtype, [outTotal]));
            idxCarry0 = builder.createConstant(
                `mpi_idx0_${op.id}`,
                zeroTensor(DataType.INT64, [outTotal]),
            );
            carryKnownShape = [outTotal];
            outShapeNode = builder.createConstant(`mpi_oshape_${op.id}`, int64Vec(staticOut));
        } else {
            const ax0 = builder.createConstant(`mpib_ax0_${op.id}`, int64Vec([0]));
            outShapeNode = packShapeVector(builder, outerOutDims, `mpib_osh_${op.id}`);
            const [outTotal] = builder.createOp("ReduceProd", [outShapeNode, ax0], { keepdims: 0 });
            const [carryShapeVec] = builder.createOp("Unsqueeze", [outTotal, ax0]);
            const totalIters = mulMixed(builder, outTotal, kTotal, `mpib_it_${op.id}`);
            tripCount =
                typeof totalIters === "number"
                    ? builder.createConstant(`mpi_trip_${op.id}`, scalarInt64(totalIters))
                    : totalIters;
            const zeroV = builder.createConstant(
                `mpi_zv_${op.id}`,
                makeTensorProto(dtype, [], [0]),
            );
            const zeroI = builder.createConstant(`mpi_zi_${op.id}`, scalarInt64(0));
            [valCarry0] = builder.createOp("Expand", [zeroV, carryShapeVec], {}, [
                { type: dtype, shape: [-1] },
            ]);
            [idxCarry0] = builder.createOp("Expand", [zeroI, carryShapeVec], {}, [
                { type: DataType.INT64, shape: [-1] },
            ]);
            carryKnownShape = [-1];
        }

        const condInit = builder.createConstant(`mpi_cond_${op.id}`, bool(true));

        // 2. Build Loop body
        const loopBody = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const innerBuilder = new GraphBuilder(loopBody, `mpi_body_${op.id}`);

        const iterNode = loopBody
            .addNode(uniq(loopBody, "iter"))
            .init(new TensorNode.Builder(DataType.INT64, [], "input"))
            .as(TensorNode);
        const condIn = loopBody
            .addNode(uniq(loopBody, "cond_in"))
            .init(new TensorNode.Builder(DataType.BOOL, [], "input"))
            .as(TensorNode);
        const valCarryIn = loopBody
            .addNode(uniq(loopBody, "val_carry_in"))
            .init(new TensorNode.Builder(dtype, carryKnownShape, "input"))
            .as(TensorNode);
        const idxCarryIn = loopBody
            .addNode(uniq(loopBody, "idx_carry_in"))
            .init(new TensorNode.Builder(DataType.INT64, carryKnownShape, "input"))
            .as(TensorNode);

        const axes = innerBuilder.createConstant(`mpi_ax0_${op.id}`, int64Vec([0]));
        const zeroI64 = innerBuilder.createConstant(`mpi_z64_${op.id}`, scalarInt64(0));
        const negInfVal =
            dtype === DataType.FLOAT || dtype === DataType.DOUBLE
                ? -3.4028234663852886e38
                : -2147483648;
        const negInf = innerBuilder.createConstant(
            `mpi_neginf_${op.id}`,
            makeTensorProto(dtype, [], [negInfVal]),
        );

        const xCap = createCapturedInput(loopBody, X);
        const innerInDims = extractMixedDims(innerBuilder, xCap, rawInDims, `mpi_id_${op.id}`);
        const { pHeads, outSpatial } = this.resolveMixedPadsAndOutSpatial(
            op,
            innerBuilder,
            innerInDims,
            `mpi_sh_${op.id}`,
        );
        const innerOutDims = [innerInDims[0], innerInDims[1], ...outSpatial];
        const iterDims = [...innerOutDims, ...kernel];
        const inStrides = buildMixedStrides(innerBuilder, innerInDims);
        const outStrides = buildMixedStrides(innerBuilder, innerOutDims);
        const flatX = ensureFlatInput(innerBuilder, xCap);

        const coords = decodeMixedRadix(innerBuilder, iterNode, iterDims, `mpi_dec_${op.id}`);
        const outPtrStrides = [...outStrides, ...new Array<number>(spatialRank).fill(0)];
        const ncPtrStrides = [
            inStrides[0],
            inStrides[1],
            ...new Array<number>(spatialRank * 2).fill(0),
        ];

        const flatOutIdx = buildLinearIndex(
            innerBuilder,
            coords,
            outPtrStrides,
            `mpi_olin_${op.id}`,
        );
        const ncBasePtr = buildLinearIndex(
            innerBuilder,
            coords,
            ncPtrStrides,
            `mpi_nclin_${op.id}`,
        );

        let flatInIdx: ConcreteValueNode = ncBasePtr;
        let isValid: ConcreteValueNode | null = null;
        let isFirstTap: ConcreteValueNode | null = null;
        const safeSpatialCoords: ConcreteValueNode[] = [];

        for (let d = 0; d < spatialRank; d++) {
            const oCoord = coords[2 + d];
            const kCoord = coords[2 + spatialRank + d];

            const [kIsZero] = innerBuilder.createOp("Equal", [kCoord, zeroI64]);
            isFirstTap = isFirstTap
                ? innerBuilder.createOp("And", [isFirstTap, kIsZero])[0]
                : kIsZero;

            const sVal = strides[d] ?? 1;
            const pVal = pHeads[d] ?? 0;
            const dilVal = dilations[d] ?? 1;

            let inCoord: ConcreteValueNode =
                sVal === 1
                    ? oCoord
                    : innerBuilder.createOp("Mul", [
                          oCoord,
                          innerBuilder.createConstant(`mpi_st_${d}_${op.id}`, scalarInt64(sVal)),
                      ])[0];

            const kOff: ConcreteValueNode =
                dilVal === 1
                    ? kCoord
                    : innerBuilder.createOp("Mul", [
                          kCoord,
                          innerBuilder.createConstant(`mpi_dil_${d}_${op.id}`, scalarInt64(dilVal)),
                      ])[0];

            [inCoord] = innerBuilder.createOp("Add", [inCoord, kOff]);
            if (typeof pVal !== "number") {
                [inCoord] = innerBuilder.createOp("Sub", [inCoord, pVal]);
            } else if (pVal !== 0) {
                [inCoord] = innerBuilder.createOp("Sub", [
                    inCoord,
                    innerBuilder.createConstant(`mpi_pad_${d}_${op.id}`, scalarInt64(pVal)),
                ]);
            }

            const dimNode =
                typeof innerInDims[2 + d] === "number"
                    ? innerBuilder.createConstant(
                          `mpi_dim_${d}_${op.id}`,
                          scalarInt64(innerInDims[2 + d] as number),
                      )
                    : (innerInDims[2 + d] as ConcreteValueNode);
            const [ltZero] = innerBuilder.createOp("Less", [inCoord, zeroI64]);
            const [geZero] = innerBuilder.createOp("Not", [ltZero]);
            const [ltDim] = innerBuilder.createOp("Less", [inCoord, dimNode]);
            const [dimValid] = innerBuilder.createOp("And", [geZero, ltDim]);
            isValid = isValid ? innerBuilder.createOp("And", [isValid, dimValid])[0] : dimValid;

            const [safeCoord] = innerBuilder.createOp("Where", [dimValid, inCoord, zeroI64]);
            safeSpatialCoords.push(safeCoord);

            const str = inStrides[2 + d];
            const term =
                typeof str !== "number"
                    ? innerBuilder.createOp("Mul", [safeCoord, str])[0]
                    : str === 1
                      ? safeCoord
                      : innerBuilder.createOp("Mul", [
                            safeCoord,
                            innerBuilder.createConstant(`mpi_istr_${d}_${op.id}`, scalarInt64(str)),
                        ])[0];
            [flatInIdx] = innerBuilder.createOp("Add", [flatInIdx, term]);
        }

        // Reported index: row-major flatInIdx (storage_order=0) or column-major (storage_order=1, 2D spatial)
        let reportedIdx: ConcreteValueNode = flatInIdx;
        if (storageOrder === 1 && spatialRank === 2) {
            const hNode =
                typeof innerInDims[2] === "number"
                    ? innerBuilder.createConstant(
                          `mpi_h_${op.id}`,
                          scalarInt64(innerInDims[2] as number),
                      )
                    : (innerInDims[2] as ConcreteValueNode);
            const [wTimesH] = innerBuilder.createOp("Mul", [safeSpatialCoords[1], hNode]);
            const [colMajorSpatial] = innerBuilder.createOp("Add", [wTimesH, safeSpatialCoords[0]]);
            [reportedIdx] = innerBuilder.createOp("Add", [ncBasePtr, colMajorSpatial]);
        }

        const [inUnsq] = innerBuilder.createOp("Unsqueeze", [flatInIdx, axes]);
        const [rawVal] = innerBuilder.createOp("Gather", [flatX, inUnsq], { axis: 0 });
        const xVal = squeezeIfLen1(
            innerBuilder,
            rawVal,
            axes,
            `mpi_sqx_${op.id}`,
        ) as ConcreteValueNode;
        const [effVal] = innerBuilder.createOp("Where", [isValid!, xVal, negInf]);

        const [outUnsq] = innerBuilder.createOp("Unsqueeze", [flatOutIdx, axes]);
        const [prevValRaw] = innerBuilder.createOp("Gather", [valCarryIn, outUnsq], { axis: 0 });
        const prevVal = squeezeIfLen1(
            innerBuilder,
            prevValRaw,
            axes,
            `mpi_sqv_${op.id}`,
        ) as ConcreteValueNode;

        const [prevIdxRaw] = innerBuilder.createOp("Gather", [idxCarryIn, outUnsq], { axis: 0 });
        const prevIdx = squeezeIfLen1(
            innerBuilder,
            prevIdxRaw,
            axes,
            `mpi_sqi_${op.id}`,
        ) as ConcreteValueNode;

        const [baseVal] = innerBuilder.createOp("Where", [isFirstTap!, negInf, prevVal]);
        const [isBetter] = innerBuilder.createOp("Greater", [effVal, baseVal]);
        const [firstValid] = innerBuilder.createOp("And", [isFirstTap!, isValid!]);
        const [betterValid] = innerBuilder.createOp("And", [isValid!, isBetter]);
        const [takeNew] = innerBuilder.createOp("Or", [firstValid, betterValid]);

        const [newVal] = innerBuilder.createOp("Where", [takeNew, effVal, baseVal]);
        const [newIdx] = innerBuilder.createOp("Where", [takeNew, reportedIdx, prevIdx]);

        const [newValUnsq] = innerBuilder.createOp("Unsqueeze", [newVal, axes]);
        const [nextValCarry] = innerBuilder.createOp(
            "ScatterElements",
            [valCarryIn, outUnsq, newValUnsq],
            { axis: 0 },
        );

        const [newIdxUnsq] = innerBuilder.createOp("Unsqueeze", [newIdx, axes]);
        const [nextIdxCarry] = innerBuilder.createOp(
            "ScatterElements",
            [idxCarryIn, outUnsq, newIdxUnsq],
            { axis: 0 },
        );

        emitBodyOutput(loopBody, condIn, "cond_out");
        emitBodyOutput(loopBody, nextValCarry, "val_carry_out");
        emitBodyOutput(loopBody, nextIdxCarry, "idx_carry_out");

        // 3. Emit outer Loop node
        const loopInputs: ValueNode[] = [tripCount, condInit, valCarry0, idxCarry0];
        const loopOp = builder
            .addNode(
                uniq(outerGraph, `MaxPoolIdxLoop_${op.id}`),
                new OperationNode.Builder("Loop", loopInputs, {}, [loopBody]),
            )
            .as(OperationNode);

        for (const inp of loopInputs) {
            if (outerGraph.hasNode(inp.id)) {
                builder.addEdge(inp, loopOp, inp.literalType, inp.shape);
            }
        }

        const flatValOut = builder
            .addNode(
                uniq(outerGraph, `mpi_flat_val_${op.id}`),
                new TensorNode.Builder(dtype, carryKnownShape, "intermediate"),
            )
            .as(TensorNode);
        const flatIdxOut = builder
            .addNode(
                uniq(outerGraph, `mpi_flat_idx_${op.id}`),
                new TensorNode.Builder(DataType.INT64, carryKnownShape, "intermediate"),
            )
            .as(TensorNode);

        builder.addEdge(loopOp, flatValOut, dtype, carryKnownShape);
        builder.addEdge(loopOp, flatIdxOut, DataType.INT64, carryKnownShape);

        // 4. Reshape both outputs to outDims
        const [finalY] = builder.createOp("Reshape", [flatValOut, outShapeNode], {}, [
            { type: dtype, shape: knownOutShape },
        ]);
        const [finalIdx] = builder.createOp("Reshape", [flatIdxOut, outShapeNode], {}, [
            { type: DataType.INT64, shape: knownOutShape },
        ]);

        builder.replaceAllUsesWith(YOut, finalY);
        builder.replaceAllUsesWith(IdxOut, finalIdx);
        builder.removeNode(op);
    }
}
