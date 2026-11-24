import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import { DataType, type ConcreteValueNode, type KnownShape } from "../../../OnnxTypes.js";
import { int64Vec, makeTensorProto, scalarInt64, toStaticShape } from "../../../Utils.js";

export class LowerDetDynamicRecipe implements DecompositionRecipe {
    public readonly name = "LowerDetDynamic";
    public readonly targetOp = "Det";
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 10;
    public readonly shapeSupport = "dynamic-batch" as const;
    public readonly tags = ["linear-algebra", "dynamic-batch"] as const;
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = [
        "Shape",
        "Gather",
        "ConstantOfShape",
        "Cast",
        "Abs",
        "ArgMax",
        "Add",
        "Sub",
        "Mul",
        "Div",
        "Neg",
        "Equal",
        "Not",
        "Greater",
        "Where",
        "ReduceSum",
        "Squeeze",
        "Unsqueeze",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "Det") return false;
        const X = op.getInputs()?.[0];
        if (!X || !Array.isArray(X.shape) || X.shape.length < 3) return false;
        const dims = toStaticShape(X.shape);
        const n = dims[dims.length - 1];
        const m = dims[dims.length - 2];
        return n > 0 && m === n;
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const X = op.getInputs()![0] as ConcreteValueNode;
        const Y = op.getOutputs()[0];
        const dtype = (X.literalType || DataType.FLOAT) as DataType;
        const dims = toStaticShape(X.shape);
        const rank = dims.length;
        const n = dims[rank - 1];
        const batchRank = rank - 2;
        const batchShape = dims.slice(0, batchRank) as KnownShape;

        const elem = (r: number, c: number, tag: string): ConcreteValueNode => {
            const [row] = builder.createOp(
                "Gather",
                [X, builder.createConstant(`detd_r_${tag}_${op.id}`, scalarInt64(r))],
                { axis: rank - 2 },
            );
            return builder.createOp(
                "Gather",
                [row, builder.createConstant(`detd_c_${tag}_${op.id}`, scalarInt64(c))],
                { axis: rank - 2 },
                [{ type: dtype, shape: batchShape }],
            )[0];
        };

        if (n === 1) {
            const sqAxes = builder.createConstant(
                `detd_sq1_${op.id}`,
                int64Vec([rank - 2, rank - 1]),
            );
            const [out] = builder.createOp("Squeeze", [X, sqAxes], {}, [
                { type: dtype, shape: batchShape },
            ]);
            builder.replaceAllUsesWith(Y, out);
            builder.removeNode(op);
            return;
        }

        if (n === 2) {
            const a00 = elem(0, 0, "00");
            const a01 = elem(0, 1, "01");
            const a10 = elem(1, 0, "10");
            const a11 = elem(1, 1, "11");
            const [p1] = builder.createOp("Mul", [a00, a11]);
            const [p2] = builder.createOp("Mul", [a01, a10]);
            const [out] = builder.createOp("Sub", [p1, p2], {}, [
                { type: dtype, shape: batchShape },
            ]);
            builder.replaceAllUsesWith(Y, out);
            builder.removeNode(op);
            return;
        }

        if (n === 3) {
            const a00 = elem(0, 0, "00"),
                a01 = elem(0, 1, "01"),
                a02 = elem(0, 2, "02");
            const a10 = elem(1, 0, "10"),
                a11 = elem(1, 1, "11"),
                a12 = elem(1, 2, "12");
            const a20 = elem(2, 0, "20"),
                a21 = elem(2, 1, "21"),
                a22 = elem(2, 2, "22");

            const [m0] = builder.createOp("Sub", [
                builder.createOp("Mul", [a11, a22])[0],
                builder.createOp("Mul", [a12, a21])[0],
            ]);
            const [m1] = builder.createOp("Sub", [
                builder.createOp("Mul", [a10, a22])[0],
                builder.createOp("Mul", [a12, a20])[0],
            ]);
            const [m2] = builder.createOp("Sub", [
                builder.createOp("Mul", [a10, a21])[0],
                builder.createOp("Mul", [a11, a20])[0],
            ]);
            const [t0] = builder.createOp("Mul", [a00, m0]);
            const [t1] = builder.createOp("Mul", [a01, m1]);
            const [t2] = builder.createOp("Mul", [a02, m2]);
            const [out] = builder.createOp("Add", [builder.createOp("Sub", [t0, t1])[0], t2], {}, [
                { type: dtype, shape: batchShape },
            ]);
            builder.replaceAllUsesWith(Y, out);
            builder.removeNode(op);
            return;
        }

        // General N >= 4: batched Gaussian elimination with partial pivoting using dynamic batchShape
        const [shapeX] = builder.createOp("Shape", [X], {}, [
            { type: DataType.INT64, shape: [rank] },
        ]);
        const batchIdx = builder.createConstant(
            `detd_bidx_${op.id}`,
            int64Vec(Array.from({ length: batchRank }, (_, i) => i)),
        );
        const [batchShapeNode] = builder.createOp("Gather", [shapeX, batchIdx], { axis: 0 }, [
            { type: DataType.INT64, shape: [batchRank] },
        ]);

        const matShape = dims as KnownShape;
        const colShape = [...batchShape, n, 1] as KnownShape;
        const rowShape = [...batchShape, 1, n] as KnownShape;
        const vecShape = [...batchShape, n] as KnownShape;
        const pivExpShape = [...batchShape, 1, 1] as KnownShape;

        const [zeroF] = builder.createOp("ConstantOfShape", [batchShapeNode], {}, [
            { type: DataType.FLOAT, shape: batchShape },
        ]);
        const zeroBatch: ConcreteValueNode =
            dtype === DataType.FLOAT
                ? zeroF
                : builder.createOp("Cast", [zeroF], { to: dtype }, [
                      { type: dtype, shape: batchShape },
                  ])[0];
        const oneC = builder.createConstant(`detd_1_${op.id}`, makeTensorProto(dtype, [], [1]));
        const zeroC = builder.createConstant(`detd_0_${op.id}`, makeTensorProto(dtype, [], [0]));
        const [onesBatch] = builder.createOp("Add", [zeroBatch, oneC], {}, [
            { type: dtype, shape: batchShape },
        ]);

        const rowRangeShape = [...new Array<number>(batchRank).fill(1), n, 1];
        const rowRange = builder.createConstant(
            `detd_rr_${op.id}`,
            makeTensorProto(
                DataType.INT64,
                rowRangeShape,
                Array.from({ length: n }, (_, i) => i),
            ),
        );
        const unsqLast2 = builder.createConstant(
            `detd_u2_${op.id}`,
            int64Vec([rank - 2, rank - 1]),
        );
        const rowAxisVec = builder.createConstant(`detd_rax_${op.id}`, int64Vec([rank - 2]));

        let curM: ConcreteValueNode = X;
        let sign: ConcreteValueNode = onesBatch;
        let det: ConcreteValueNode = onesBatch;

        for (let k = 0; k < n; k++) {
            const kSc = builder.createConstant(`detd_ksc_${k}_${op.id}`, scalarInt64(k));
            const kVec = builder.createConstant(`detd_kv_${k}_${op.id}`, int64Vec([k]));
            const subRowsIdx = builder.createConstant(
                `detd_sr_${k}_${op.id}`,
                int64Vec(Array.from({ length: n - k }, (_, i) => k + i)),
            );

            const subShape = [...batchShape, n - k] as KnownShape;

            // 1. Find pivot row in k..n-1
            const [colK] = builder.createOp("Gather", [curM, kSc], { axis: rank - 1 }, [
                { type: dtype, shape: vecShape },
            ]);
            const [subCol] = builder.createOp("Gather", [colK, subRowsIdx], { axis: rank - 2 }, [
                { type: dtype, shape: subShape },
            ]);
            const [absSub] = builder.createOp("Abs", [subCol], {}, [
                { type: dtype, shape: subShape },
            ]);
            const [relPiv] = builder.createOp(
                "ArgMax",
                [absSub],
                { axis: rank - 2, keepdims: 0, select_last_index: 0 },
                [{ type: DataType.INT64, shape: batchShape }],
            );
            const [pivRow] = builder.createOp("Add", [relPiv, kSc], {}, [
                { type: DataType.INT64, shape: batchShape },
            ]);

            // 2. Swap row k with pivRow
            const [pivRowExp] = builder.createOp("Unsqueeze", [pivRow, unsqLast2], {}, [
                { type: DataType.INT64, shape: pivExpShape },
            ]);
            const [isK] = builder.createOp("Equal", [rowRange, kSc], {}, [
                { type: DataType.BOOL, shape: rowRangeShape as KnownShape },
            ]);
            const [isPiv] = builder.createOp("Equal", [rowRange, pivRowExp], {}, [
                { type: DataType.BOOL, shape: colShape },
            ]);

            const [rowK] = builder.createOp("Gather", [curM, kVec], { axis: rank - 2 }, [
                { type: dtype, shape: rowShape },
            ]);
            const [maskedPiv] = builder.createOp("Where", [isPiv, curM, zeroC], {}, [
                { type: dtype, shape: matShape },
            ]);
            const [rowPiv] = builder.createOp(
                "ReduceSum",
                [maskedPiv, rowAxisVec],
                { keepdims: 1 },
                [{ type: dtype, shape: rowShape }],
            );

            const [swapped1] = builder.createOp("Where", [isK, rowPiv, curM], {}, [
                { type: dtype, shape: matShape },
            ]);
            const [swapped] = builder.createOp("Where", [isPiv, rowK, swapped1], {}, [
                { type: dtype, shape: matShape },
            ]);

            const [sameRow] = builder.createOp("Equal", [pivRow, kSc], {}, [
                { type: DataType.BOOL, shape: batchShape },
            ]);
            const [didSwap] = builder.createOp("Not", [sameRow], {}, [
                { type: DataType.BOOL, shape: batchShape },
            ]);
            const [negSign] = builder.createOp("Neg", [sign], {}, [
                { type: dtype, shape: batchShape },
            ]);
            [sign] = builder.createOp("Where", [didSwap, negSign, sign], {}, [
                { type: dtype, shape: batchShape },
            ]);

            // 3. Accumulate pivot into det
            const [rowPivSq] = builder.createOp("Squeeze", [rowPiv, rowAxisVec], {}, [
                { type: dtype, shape: vecShape },
            ]);
            const [pivVal] = builder.createOp("Gather", [rowPivSq, kSc], { axis: rank - 2 }, [
                { type: dtype, shape: batchShape },
            ]);
            [det] = builder.createOp("Mul", [det, pivVal], {}, [
                { type: dtype, shape: batchShape },
            ]);

            // 4. Eliminate entries below row k
            if (k < n - 1) {
                const [isZeroPiv] = builder.createOp("Equal", [pivVal, zeroC], {}, [
                    { type: DataType.BOOL, shape: batchShape },
                ]);
                const [safePiv] = builder.createOp("Where", [isZeroPiv, oneC, pivVal], {}, [
                    { type: dtype, shape: batchShape },
                ]);
                const [invPiv] = builder.createOp("Reciprocal", [safePiv], {}, [
                    { type: dtype, shape: batchShape },
                ]);
                const [invPivExp] = builder.createOp("Unsqueeze", [invPiv, unsqLast2], {}, [
                    { type: dtype, shape: pivExpShape },
                ]);

                const [colAfter] = builder.createOp("Gather", [swapped, kVec], { axis: rank - 1 }, [
                    { type: dtype, shape: colShape },
                ]);
                const [factors] = builder.createOp("Mul", [colAfter, invPivExp], {}, [
                    { type: dtype, shape: colShape },
                ]);
                const [update] = builder.createOp("Mul", [factors, rowPiv], {}, [
                    { type: dtype, shape: matShape },
                ]);
                const [eliminated] = builder.createOp("Sub", [swapped, update], {}, [
                    { type: dtype, shape: matShape },
                ]);

                const [gtK] = builder.createOp("Greater", [rowRange, kSc], {}, [
                    { type: DataType.BOOL, shape: rowRangeShape as KnownShape },
                ]);
                [curM] = builder.createOp("Where", [gtK, eliminated, swapped], {}, [
                    { type: dtype, shape: matShape },
                ]);
            }
        }

        const [out] = builder.createOp("Mul", [sign, det], {}, [
            { type: dtype, shape: batchShape },
        ]);
        builder.replaceAllUsesWith(Y, out);
        builder.removeNode(op);
    }
}
