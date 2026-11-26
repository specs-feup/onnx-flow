import type OperationNode from "../../../ir/OperationNode.js";
import type { GraphBuilder } from "../../../ir/GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../ir/OnnxTypes.js";
import { DataType } from "../../../ir/OnnxTypes.js";
import {
    asConcreteValueNode,
    int64Vec,
    makeTensorProto,
    scalarInt64,
    toStaticShape,
} from "../../../utils/index.js";

export class LowerDetRecipe implements DecompositionRecipe {
    public readonly name = "LowerDet";
    public readonly targetOp = "Det";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic-batch" as const;
    public readonly tags = ["linear-algebra"];
    public readonly producedOps = [
        "Slice",
        "Squeeze",
        "Unsqueeze",
        "Mul",
        "Sub",
        "Add",
        "Div",
        "Neg",
        "Abs",
        "ArgMax",
        "Range",
        "Reshape",
        "Equal",
        "Not",
        "Greater",
        "Where",
        "ReduceSum",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "Det") return false;
        const X = op.getInputs()?.[0];
        if (!X || !Array.isArray(X.shape)) return false;
        const dims = toStaticShape(X.shape);
        if (dims.length < 2) return false;
        const n = dims[dims.length - 1];
        if (n < 1 || dims[dims.length - 2] !== n) return false;
        // General N >= 4 allocates static batch-sized sign/det constants
        if (n >= 4 && dims.slice(0, -2).includes(-1)) return false;
        return true;
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const X = asConcreteValueNode(op.getInputs()![0]);
        const Y = op.getOutputs()[0];
        const dims = toStaticShape(X.shape);
        const rank = dims.length;
        const n = dims[rank - 1];
        const batchShape = dims.slice(0, rank - 2);
        const dtype = X.literalType as DataType;

        let tag = 0;
        const uid = (p: string) => `det_${p}_${op.id}_${tag++}`;
        const batchMeta = [{ type: dtype, shape: batchShape as KnownShape }];

        const sqLast2Axes = builder.createConstant(uid("sq2"), int64Vec([rank - 2, rank - 1]));

        const elem = (i: number, j: number): ConcreteValueNode => {
            const slShape = [...batchShape, 1, 1];
            const [s] = builder.createOp(
                "Slice",
                [
                    X,
                    builder.createConstant(uid("st"), int64Vec([i, j])),
                    builder.createConstant(uid("en"), int64Vec([i + 1, j + 1])),
                    sqLast2Axes,
                    builder.createConstant(uid("sp"), int64Vec([1, 1])),
                ],
                {},
                [{ type: dtype, shape: slShape as KnownShape }],
            );
            const [sq] = builder.createOp("Squeeze", [s, sqLast2Axes], {}, batchMeta);
            return sq;
        };

        let out: ConcreteValueNode;

        if (n === 1) {
            [out] = builder.createOp("Squeeze", [X, sqLast2Axes], {}, batchMeta);
        } else if (n === 2) {
            const [p0] = builder.createOp("Mul", [elem(0, 0), elem(1, 1)], {}, batchMeta);
            const [p1] = builder.createOp("Mul", [elem(0, 1), elem(1, 0)], {}, batchMeta);
            [out] = builder.createOp("Sub", [p0, p1], {}, batchMeta);
        } else if (n === 3) {
            const minor2 = (r1: number, r2: number, c1: number, c2: number): ConcreteValueNode => {
                const [a] = builder.createOp("Mul", [elem(r1, c1), elem(r2, c2)], {}, batchMeta);
                const [b] = builder.createOp("Mul", [elem(r1, c2), elem(r2, c1)], {}, batchMeta);
                return builder.createOp("Sub", [a, b], {}, batchMeta)[0];
            };
            const [t0] = builder.createOp("Mul", [elem(0, 0), minor2(1, 2, 1, 2)], {}, batchMeta);
            const [t1] = builder.createOp("Mul", [elem(0, 1), minor2(1, 2, 0, 2)], {}, batchMeta);
            const [t2] = builder.createOp("Mul", [elem(0, 2), minor2(1, 2, 0, 1)], {}, batchMeta);
            const [d01] = builder.createOp("Sub", [t0, t1], {}, batchMeta);
            [out] = builder.createOp("Add", [d01, t2], {}, batchMeta);
        } else {
            // General N >= 4: Branch-free Gaussian Elimination with Partial Pivoting
            const rowAxis = rank - 2;
            const colAxis = rank - 1;
            const rowAxisC = builder.createConstant(uid("rax"), int64Vec([rowAxis]));
            const colAxisC = builder.createConstant(uid("cax"), int64Vec([colAxis]));
            const step1C = builder.createConstant(uid("st1"), int64Vec([1]));
            const step11C = builder.createConstant(uid("st11"), int64Vec([1, 1]));
            const zeroC = builder.createConstant(uid("z"), makeTensorProto(dtype, [], [0]));
            const oneC = builder.createConstant(uid("o"), makeTensorProto(dtype, [], [1]));

            const [rVec] = builder.createOp(
                "Range",
                [
                    builder.createConstant(uid("r0"), scalarInt64(0)),
                    builder.createConstant(uid("rN"), scalarInt64(n)),
                    builder.createConstant(uid("r1"), scalarInt64(1)),
                ],
                {},
                [{ type: DataType.INT64, shape: [n] as KnownShape }],
            );
            const rIdxShape = new Array<number>(rank).fill(1);
            rIdxShape[rowAxis] = n;
            const [rIdx] = builder.createOp(
                "Reshape",
                [rVec, builder.createConstant(uid("rsh"), int64Vec(rIdxShape))],
                {},
                [{ type: DataType.INT64, shape: rIdxShape as KnownShape }],
            );

            const batchElems = batchShape.reduce((a, b) => a * b, 1);
            let A: ConcreteValueNode = X;
            let sign: ConcreteValueNode = builder.createConstant(
                uid("sgn"),
                makeTensorProto(dtype, batchShape, new Array<number>(batchElems).fill(1)),
            );
            let det: ConcreteValueNode = builder.createConstant(
                uid("det"),
                makeTensorProto(dtype, batchShape, new Array<number>(batchElems).fill(1)),
            );

            for (let k = 0; k < n; k++) {
                const kC = builder.createConstant(uid("k"), scalarInt64(k));
                const rem = n - k;

                // 1. Slice column k from rows k..n-1 -> [...batch, rem]
                const [colSl] = builder.createOp(
                    "Slice",
                    [
                        A,
                        builder.createConstant(uid("cs"), int64Vec([k, k])),
                        builder.createConstant(uid("ce"), int64Vec([n, k + 1])),
                        sqLast2Axes,
                        step11C,
                    ],
                    {},
                    [{ type: dtype, shape: [...batchShape, rem, 1] as KnownShape }],
                );
                const [colSq] = builder.createOp("Squeeze", [colSl, colAxisC], {}, [
                    { type: dtype, shape: [...batchShape, rem] as KnownShape },
                ]);
                const [absCol] = builder.createOp("Abs", [colSq], {}, [
                    { type: dtype, shape: [...batchShape, rem] as KnownShape },
                ]);

                // 2. Find pivot row index p in k..n-1
                const [pRel] = builder.createOp(
                    "ArgMax",
                    [absCol],
                    { axis: rowAxis, keepdims: 1, select_last_index: 0 },
                    [{ type: DataType.INT64, shape: [...batchShape, 1] as KnownShape }],
                );
                const [p] = builder.createOp("Add", [pRel, kC], {}, [
                    { type: DataType.INT64, shape: [...batchShape, 1] as KnownShape },
                ]);
                const [pBcast] = builder.createOp("Unsqueeze", [p, colAxisC], {}, [
                    { type: DataType.INT64, shape: [...batchShape, 1, 1] as KnownShape },
                ]);

                // 3. Flip sign if p != k
                const [pBatch] = builder.createOp("Squeeze", [p, rowAxisC], {}, [
                    { type: DataType.INT64, shape: batchShape as KnownShape },
                ]);
                const [eqK] = builder.createOp("Equal", [pBatch, kC], {}, [
                    { type: DataType.BOOL, shape: batchShape as KnownShape },
                ]);
                const [swapped] = builder.createOp("Not", [eqK], {}, [
                    { type: DataType.BOOL, shape: batchShape as KnownShape },
                ]);
                const [negSign] = builder.createOp("Neg", [sign], {}, batchMeta);
                [sign] = builder.createOp("Where", [swapped, negSign, sign], {}, batchMeta);

                // 4. Extract row k and row p, then swap them in A
                const rowShape = [...batchShape, 1, n];
                const [rowK] = builder.createOp(
                    "Slice",
                    [
                        A,
                        builder.createConstant(uid("rks"), int64Vec([k])),
                        builder.createConstant(uid("rke"), int64Vec([k + 1])),
                        rowAxisC,
                        step1C,
                    ],
                    {},
                    [{ type: dtype, shape: rowShape as KnownShape }],
                );
                const [isP] = builder.createOp("Equal", [rIdx, pBcast], {}, [
                    { type: DataType.BOOL, shape: [...batchShape, n, 1] as KnownShape },
                ]);
                const [maskedP] = builder.createOp("Where", [isP, A, zeroC], {}, [
                    { type: dtype, shape: dims as KnownShape },
                ]);
                const [rowP] = builder.createOp("ReduceSum", [maskedP, rowAxisC], { keepdims: 1 }, [
                    { type: dtype, shape: rowShape as KnownShape },
                ]);
                const [isK] = builder.createOp("Equal", [rIdx, kC], {}, [
                    { type: DataType.BOOL, shape: rIdxShape as KnownShape },
                ]);
                const [aSwap1] = builder.createOp("Where", [isP, rowK, A], {}, [
                    { type: dtype, shape: dims as KnownShape },
                ]);
                [A] = builder.createOp("Where", [isK, rowP, aSwap1], {}, [
                    { type: dtype, shape: dims as KnownShape },
                ]);

                // 5. Multiply det by pivot element A[..., k, k] = rowP[..., 0, k]
                const [piv] = builder.createOp(
                    "Slice",
                    [
                        rowP,
                        builder.createConstant(uid("ps"), int64Vec([k])),
                        builder.createConstant(uid("pe"), int64Vec([k + 1])),
                        colAxisC,
                        step1C,
                    ],
                    {},
                    [{ type: dtype, shape: [...batchShape, 1, 1] as KnownShape }],
                );
                const [pivBatch] = builder.createOp("Squeeze", [piv, sqLast2Axes], {}, batchMeta);
                [det] = builder.createOp("Mul", [det, pivBatch], {}, batchMeta);

                // 6. Eliminate column k in rows > k
                if (k < n - 1) {
                    const [isZeroPiv] = builder.createOp("Equal", [piv, zeroC], {}, [
                        { type: DataType.BOOL, shape: [...batchShape, 1, 1] as KnownShape },
                    ]);
                    const [safePiv] = builder.createOp("Where", [isZeroPiv, oneC, piv], {}, [
                        { type: dtype, shape: [...batchShape, 1, 1] as KnownShape },
                    ]);
                    const [colK] = builder.createOp(
                        "Slice",
                        [
                            A,
                            builder.createConstant(uid("cks"), int64Vec([k])),
                            builder.createConstant(uid("cke"), int64Vec([k + 1])),
                            colAxisC,
                            step1C,
                        ],
                        {},
                        [{ type: dtype, shape: [...batchShape, n, 1] as KnownShape }],
                    );
                    const [factors] = builder.createOp("Div", [colK, safePiv], {}, [
                        { type: dtype, shape: [...batchShape, n, 1] as KnownShape },
                    ]);
                    const [delta] = builder.createOp("Mul", [factors, rowP], {}, [
                        { type: dtype, shape: dims as KnownShape },
                    ]);
                    const [subA] = builder.createOp("Sub", [A, delta], {}, [
                        { type: dtype, shape: dims as KnownShape },
                    ]);
                    const [belowK] = builder.createOp("Greater", [rIdx, kC], {}, [
                        { type: DataType.BOOL, shape: rIdxShape as KnownShape },
                    ]);
                    [A] = builder.createOp("Where", [belowK, subA, A], {}, [
                        { type: dtype, shape: dims as KnownShape },
                    ]);
                }
            }

            [out] = builder.createOp("Mul", [sign, det], {}, batchMeta);
        }

        builder.replaceAllUsesWith(Y, out);
        builder.removeNode(op);
    }
}
