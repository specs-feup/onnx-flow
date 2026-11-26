import { describe, it, expect } from "vitest";
import { applyAdapters } from "../../../src/onnx/frontend/Adapters.js";
import { createGraph } from "../../../src/io/OnnxGraphLoader.js";
import { DataType, AttributeType, type RawOnnxModel } from "../../../src/onnx/OnnxTypes.js";
import { initializeSchemaRegistry } from "../../../src/onnx/schema/index.js";
import { makeTensorProto } from "../../../src/onnx/utils/index.js";
import RegionArgumentNode from "../../../src/onnx/RegionArgumentNode.js";
import ConstantNode from "../../../src/onnx/ConstantNode.js";

describe("Frontend Adapters & Graph Ingestion (initGraph)", () => {
    initializeSchemaRegistry();

    describe("Adapters.ts", () => {
        it("adapts legacy Clip node attributes (min & max) to input initializers", () => {
            const rawModel: RawOnnxModel = {
                graph: {
                    initializer: [],
                    node: [
                        {
                            opType: "Clip",
                            input: ["X"],
                            output: ["Y"],
                            attribute: [
                                { name: "min", type: "FLOAT", f: -2.0 },
                                { name: "max", type: "FLOAT", f: 2.0 },
                            ],
                        },
                    ],
                },
            };

            applyAdapters(rawModel);
            const clipNode = rawModel.graph?.node?.find((n) => n.opType === "Clip");
            expect(clipNode?.input?.length).toBe(3);
            expect(clipNode?.attribute?.length).toBe(0);
            expect(rawModel.graph?.initializer?.length).toBe(2);
        });

        it("adapts legacy Clip node with only min or only max attribute", () => {
            const rawModelMin: RawOnnxModel = {
                graph: {
                    initializer: [],
                    node: [
                        {
                            opType: "Clip",
                            input: ["X"],
                            output: ["Y"],
                            attribute: [{ name: "min", type: "FLOAT", f: 0.0 }],
                        },
                    ],
                },
            };
            applyAdapters(rawModelMin);
            const clipMin = rawModelMin.graph?.node?.find((n) => n.opType === "Clip");
            expect(clipMin?.input?.length).toBeGreaterThanOrEqual(2);

            const rawModelMax: RawOnnxModel = {
                graph: {
                    initializer: [],
                    node: [
                        {
                            opType: "Clip",
                            input: ["X"],
                            output: ["Y"],
                            attribute: [{ name: "max", type: "FLOAT", f: 5.0 }],
                        },
                    ],
                },
            };
            applyAdapters(rawModelMax);
            const clipMax = rawModelMax.graph?.node?.find((n) => n.opType === "Clip");
            expect(clipMax?.input?.length).toBe(3);
        });

        it("adapts Squeeze, Unsqueeze, Split, and Reshape attribute parameters to input initializers", () => {
            const rawModel: RawOnnxModel = {
                graph: {
                    initializer: [],
                    node: [
                        {
                            opType: "Squeeze",
                            input: ["X"],
                            output: ["Y1"],
                            attribute: [{ name: "axes", type: "INTS", ints: [0, 1] }],
                        },
                        {
                            opType: "Unsqueeze",
                            input: ["X"],
                            output: ["Y2"],
                            attribute: [{ name: "axes", type: "INTS", ints: [1] }],
                        },
                        {
                            opType: "Split",
                            input: ["X"],
                            output: ["Y3", "Y4"],
                            attribute: [{ name: "split", type: "INTS", ints: [2, 2] }],
                        },
                        {
                            opType: "Reshape",
                            input: ["X"],
                            output: ["Y5"],
                            attribute: [{ name: "shape", type: "INTS", ints: [4, 1] }],
                        },
                    ],
                },
            };

            applyAdapters(rawModel);
            for (const node of rawModel.graph!.node!) {
                expect(node.input!.length).toBeGreaterThan(1);
            }
            expect(rawModel.graph!.initializer!.length).toBe(4);
        });

        it("adapts Pad attributes (pads & value) to input initializers", () => {
            const rawModel: RawOnnxModel = {
                graph: {
                    initializer: [],
                    node: [
                        {
                            opType: "Pad",
                            input: ["X"],
                            output: ["Y"],
                            attribute: [
                                { name: "pads", type: "INTS", ints: [0, 1, 0, 1] },
                                { name: "value", type: "FLOAT", f: 0.5 },
                            ],
                        },
                    ],
                },
            };

            applyAdapters(rawModel);
            const padNode = rawModel.graph!.node!.find((n) => n.opType === "Pad");
            expect(padNode?.input!.length).toBe(3);
            expect(rawModel.graph!.initializer!.length).toBe(2);
        });

        it("processes additional legacy operators (Dropout, BatchNormalization, Upsample, Slice)", () => {
            const rawModel: RawOnnxModel = {
                graph: {
                    initializer: [],
                    node: [
                        {
                            opType: "BatchNormalization",
                            input: ["X", "s", "b", "m", "v"],
                            output: ["Y1"],
                            attribute: [{ name: "spatial", type: "INT", i: 1 }],
                        },
                        {
                            opType: "Dropout",
                            input: ["X"],
                            output: ["Y2"],
                            attribute: [{ name: "ratio", type: "FLOAT", f: 0.5 }],
                        },
                        {
                            opType: "Upsample",
                            input: ["X"],
                            output: ["Y3"],
                            attribute: [
                                { name: "scales", type: "FLOATS", floats: [1.0, 1.0, 2.0, 2.0] },
                            ],
                        },
                        {
                            opType: "Slice",
                            input: ["X"],
                            output: ["Y4"],
                            attribute: [
                                { name: "starts", type: "INTS", ints: [0] },
                                { name: "ends", type: "INTS", ints: [2] },
                                { name: "axes", type: "INTS", ints: [0] },
                            ],
                        },
                    ],
                },
            };

            expect(() => applyAdapters(rawModel)).not.toThrow();
            const resizeNode = rawModel.graph?.node?.find((n) => n.opType === "Resize");
            expect(resizeNode).toBeDefined();
            expect(resizeNode?.input?.[1]).toBe("");
            expect(resizeNode?.input?.[2]).toBeTruthy();
            expect(
                resizeNode?.attribute?.find((a) => a.name === "coordinate_transformation_mode")?.s,
            ).toBe("asymmetric");
        });

        it("adapts pre-Opset 7 broadcast=1 and axis attributes to Unsqueeze + Multidirectional Mul", () => {
            const rawModel: RawOnnxModel = {
                irVersion: 3,
                opsetImport: [{ domain: "", version: 5 }],
                graph: {
                    input: [
                        {
                            name: "x4",
                            type: {
                                tensorType: {
                                    elemType: DataType.FLOAT,
                                    shape: {
                                        dim: [
                                            { dimValue: 2 },
                                            { dimValue: 3 },
                                            { dimValue: 4 },
                                            { dimValue: 5 },
                                        ],
                                    },
                                },
                            },
                        },
                        {
                            name: "scale1d",
                            type: {
                                tensorType: {
                                    elemType: DataType.FLOAT,
                                    shape: { dim: [{ dimValue: 3 }] },
                                },
                            },
                        },
                    ],
                    output: [
                        {
                            name: "scaled",
                            type: {
                                tensorType: {
                                    elemType: DataType.FLOAT,
                                    shape: {
                                        dim: [
                                            { dimValue: 2 },
                                            { dimValue: 3 },
                                            { dimValue: 4 },
                                            { dimValue: 5 },
                                        ],
                                    },
                                },
                            },
                        },
                    ],
                    node: [
                        {
                            name: "mul_bcast",
                            opType: "Mul",
                            input: ["x4", "scale1d"],
                            output: ["scaled"],
                            attribute: [
                                { name: "broadcast", type: AttributeType.INT, i: 1 },
                                { name: "axis", type: AttributeType.INT, i: 1 },
                            ],
                        },
                    ],
                },
            };

            applyAdapters(rawModel);

            const nodes = rawModel.graph!.node!;
            const mulNode = nodes.find((n) => n.opType === "Mul");
            expect(mulNode?.attribute?.length ?? 0).toBe(0);
            expect(nodes.some((n) => n.opType === "Unsqueeze")).toBe(true);
        });

        it("recursively adapts nested subgraphs and handles <13 Softmax and <18 ReduceMean", () => {
            const rawModel: RawOnnxModel = {
                irVersion: 8,
                opsetImport: [{ domain: "", version: 11 }],
                graph: {
                    input: [
                        {
                            name: "cond",
                            type: { tensorType: { elemType: DataType.BOOL, shape: { dim: [] } } },
                        },
                        {
                            name: "x4",
                            type: {
                                tensorType: {
                                    elemType: DataType.FLOAT,
                                    shape: {
                                        dim: [
                                            { dimValue: 2 },
                                            { dimValue: 3 },
                                            { dimValue: 4 },
                                            { dimValue: 5 },
                                        ],
                                    },
                                },
                            },
                        },
                    ],
                    output: [
                        {
                            name: "y_out",
                            type: {
                                tensorType: {
                                    elemType: DataType.FLOAT,
                                    shape: {
                                        dim: [
                                            { dimValue: 2 },
                                            { dimValue: 3 },
                                            { dimValue: 4 },
                                            { dimValue: 5 },
                                        ],
                                    },
                                },
                            },
                        },
                    ],
                    node: [
                        {
                            name: "sm_legacy",
                            opType: "Softmax",
                            input: ["x4"],
                            output: ["sm_out"],
                            attribute: [{ name: "axis", type: AttributeType.INT, i: 1 }],
                        },
                        {
                            opType: "If",
                            input: ["cond"],
                            output: ["y_out"],
                            attribute: [
                                {
                                    name: "then_branch",
                                    type: AttributeType.GRAPH,
                                    g: {
                                        node: [
                                            {
                                                name: "inner_clip",
                                                opType: "Clip",
                                                input: ["sm_out"],
                                                output: ["y_out"],
                                                attribute: [
                                                    { name: "min", type: "FLOAT", f: 0.0 },
                                                    { name: "max", type: "FLOAT", f: 1.0 },
                                                ],
                                            },
                                        ],
                                    },
                                },
                                {
                                    name: "else_branch",
                                    type: AttributeType.GRAPH,
                                    g: {
                                        node: [
                                            {
                                                name: "inner_reduce",
                                                opType: "ReduceMean",
                                                input: ["sm_out"],
                                                output: ["reduced"],
                                                attribute: [
                                                    { name: "axes", type: "INTS", ints: [2, 3] },
                                                    { name: "keepdims", type: "INT", i: 1 },
                                                ],
                                            },
                                        ],
                                    },
                                },
                            ],
                        },
                    ],
                },
            };

            applyAdapters(rawModel);

            const nodes = rawModel.graph!.node!;
            // Legacy Softmax along axis 1 on 4D input lowers to Shape -> Flatten(1) -> Softmax(1) -> Reshape
            expect(nodes.some((n) => n.opType === "Flatten")).toBe(true);
            const updatedSoftmax = nodes.find((n) => n.opType === "Softmax");
            expect(updatedSoftmax?.attribute?.find((a) => a.name === "axis")?.i).toBe(1);

            const ifNode = nodes.find((n) => n.opType === "If")!;
            const thenClip = ifNode.attribute![0].g!.node![0];
            expect(thenClip.input!.length).toBe(3);
            expect(thenClip.attribute!.length).toBe(0);

            const elseReduce = ifNode.attribute![1].g!.node![0];
            expect(elseReduce.input!.length).toBe(2);
            expect(elseReduce.attribute!.some((a) => a.name === "axes")).toBe(false);
        });

        it("produces deterministic initializer names across repeated adapter runs", () => {
            const makeModel = (): RawOnnxModel => ({
                graph: {
                    node: [
                        {
                            name: "clip0",
                            opType: "Clip",
                            input: ["X"],
                            output: ["Y"],
                            attribute: [{ name: "min", type: "FLOAT", f: -1.0 }],
                        },
                    ],
                },
            });

            const m1 = makeModel();
            const m2 = makeModel();
            applyAdapters(m1);
            applyAdapters(m2);
            expect(m1.graph!.initializer![0].name).toBe(m2.graph!.initializer![0].name);
        });
    });

    describe("initGraph.ts", () => {
        it("handles diverse shape definitions, valueInfo, initializers, and attributes", () => {
            const rawModel: RawOnnxModel = {
                graph: {
                    input: [
                        {
                            name: "in_A",
                            type: {
                                tensorType: {
                                    elemType: DataType.FLOAT,
                                    shape: {
                                        dim: [
                                            { dimParam: "batch" },
                                            { dim_param: "seq" },
                                            { dimValue: 4 },
                                            { dim_value: 8 },
                                            { dimParam: "" },
                                        ],
                                    },
                                },
                            },
                        },
                    ],
                    initializer: [
                        {
                            ...makeTensorProto(DataType.FLOAT, [1], [3.14]),
                            name: "init_const",
                        },
                    ],
                    valueInfo: [
                        {
                            name: "inter_var",
                            type: {
                                tensor_type: {
                                    elemType: DataType.FLOAT,
                                    shape: { dim: [{ dimValue: 4 }] },
                                },
                            },
                        },
                        {
                            name: "in_A",
                            type: { tensorType: { elemType: DataType.FLOAT } },
                        },
                    ],
                    output: [
                        {
                            name: "out_Y",
                            type: {
                                tensorType: {
                                    elemType: DataType.FLOAT,
                                    shape: { dim: [{ dimValue: 4 }] },
                                },
                            },
                        },
                    ],
                    node: [
                        {
                            opType: "Constant",
                            input: [],
                            output: ["c_node"],
                            attribute: [
                                {
                                    name: "value",
                                    t: makeTensorProto(DataType.INT64, [1], [99]),
                                },
                            ],
                        },
                        {
                            opType: "CustomOp",
                            input: ["in_A", ""],
                            output: ["inter_var"],
                            attribute: [
                                { name: "attr_f", type: AttributeType.FLOAT, f: 1.5 },
                                { name: "attr_f_str", type: "FLOAT", f: 2.5 },
                                { name: "attr_i", type: AttributeType.INT, i: 10 },
                                { name: "attr_i_str", type: "INT", i: 20 },
                                { name: "attr_s", type: AttributeType.STRING, s: "val" },
                                {
                                    name: "attr_floats",
                                    type: AttributeType.FLOATS,
                                    floats: [1.1, 2.2],
                                },
                                { name: "attr_ints", type: AttributeType.INTS, ints: [100, 200] },
                                {
                                    name: "attr_t",
                                    type: AttributeType.TENSOR,
                                    t: makeTensorProto(DataType.FLOAT, [1], [0.5]),
                                },
                            ],
                        },
                        {
                            opType: "Identity",
                            input: ["inter_var"],
                            output: ["out_Y"],
                        },
                    ],
                },
            };

            const graph = createGraph(rawModel);
            expect(graph.hasNode("in_A")).toBe(true);
            expect(graph.hasNode("init_const")).toBe(true);
            expect(graph.hasNode("c_node")).toBe(true);
            expect(graph.hasNode("inter_var")).toBe(true);
            expect(graph.hasNode("out_Y")).toBe(true);

            const inputTensor = graph.getNodeById("in_A");
            expect(inputTensor?.is(ConstantNode)).toBe(false);
        });

        it("parses nested control flow (If, Loop) and captures variables across scopes", () => {
            const rawModel: RawOnnxModel = {
                graph: {
                    input: [
                        {
                            name: "cond",
                            type: { tensorType: { elemType: DataType.BOOL, shape: { dim: [] } } },
                        },
                        {
                            name: "outer_var",
                            type: {
                                tensorType: {
                                    elemType: DataType.FLOAT,
                                    shape: { dim: [{ dimValue: 4 }] },
                                },
                            },
                        },
                    ],
                    output: [
                        {
                            name: "out",
                            type: {
                                tensorType: {
                                    elemType: DataType.FLOAT,
                                    shape: { dim: [{ dimValue: 4 }] },
                                },
                            },
                        },
                    ],
                    node: [
                        {
                            opType: "If",
                            input: ["cond"],
                            output: ["out"],
                            attribute: [
                                {
                                    name: "then_branch",
                                    type: AttributeType.GRAPH,
                                    g: {
                                        node: [
                                            {
                                                opType: "Relu",
                                                input: ["outer_var"],
                                                output: ["out"],
                                            },
                                        ],
                                        output: [
                                            {
                                                name: "out",
                                                type: {
                                                    tensorType: {
                                                        elemType: DataType.FLOAT,
                                                        shape: { dim: [{ dimValue: 4 }] },
                                                    },
                                                },
                                            },
                                        ],
                                    },
                                },
                                {
                                    name: "else_branch",
                                    type: AttributeType.GRAPH,
                                    g: {
                                        node: [
                                            {
                                                opType: "Identity",
                                                input: ["outer_var"],
                                                output: ["out"],
                                            },
                                        ],
                                        output: [
                                            {
                                                name: "out",
                                                type: {
                                                    tensorType: {
                                                        elemType: DataType.FLOAT,
                                                        shape: { dim: [{ dimValue: 4 }] },
                                                    },
                                                },
                                            },
                                        ],
                                    },
                                },
                            ],
                        },
                    ],
                },
            };

            const graph = createGraph(rawModel);
            const ifOp = graph
                .getOperationNodes()
                .toArray()
                .find((o) => o.type === "If");
            expect(ifOp).toBeDefined();
            expect(ifOp?.regions.length).toBe(2);

            const thenRegion = ifOp!.regions[0];
            const capturedNodes = thenRegion.getNodes().filterIs(RegionArgumentNode).toArray();
            expect(capturedNodes.length).toBeGreaterThan(0);
            expect(capturedNodes[0].originalName).toBe("outer_var");
        });

        it("propagates sourceOpset, irVersion, and explicit targetOpset across graph and child regions", () => {
            const rawModel: RawOnnxModel = {
                irVersion: 11,
                opsetImport: [{ domain: "", version: 23 }],
                graph: {
                    input: [
                        {
                            name: "cond",
                            type: { tensorType: { elemType: DataType.BOOL, shape: { dim: [] } } },
                        },
                        {
                            name: "x",
                            type: {
                                tensorType: {
                                    elemType: DataType.FLOAT,
                                    shape: { dim: [{ dimValue: 2 }, { dimValue: 4 }] },
                                },
                            },
                        },
                    ],
                    output: [
                        {
                            name: "out",
                            type: {
                                tensorType: {
                                    elemType: DataType.FLOAT,
                                    shape: { dim: [{ dimValue: 2 }, { dimValue: 4 }] },
                                },
                            },
                        },
                    ],
                    node: [
                        {
                            opType: "If",
                            input: ["cond"],
                            output: ["out"],
                            attribute: [
                                {
                                    name: "then_branch",
                                    type: AttributeType.GRAPH,
                                    g: {
                                        node: [
                                            {
                                                opType: "Relu",
                                                input: ["x"],
                                                output: ["out"],
                                            },
                                        ],
                                    },
                                },
                                {
                                    name: "else_branch",
                                    type: AttributeType.GRAPH,
                                    g: {
                                        node: [
                                            {
                                                opType: "Identity",
                                                input: ["x"],
                                                output: ["out"],
                                            },
                                        ],
                                    },
                                },
                            ],
                        },
                    ],
                },
            };

            const graph = createGraph(rawModel, 26);
            expect(graph.getSourceOpset()).toBe(23);
            expect(graph.getTargetOpset()).toBe(26);
            expect(graph.getIrVersion()).toBe(11);
            expect(graph.hasExplicitIrVersion()).toBe(false);

            const ifOp = graph
                .getOperationNodes()
                .toArray()
                .find((o) => o.type === "If")!;
            expect(ifOp.regions[0].getSourceOpset()).toBe(23);
            expect(ifOp.regions[0].getTargetOpset()).toBe(26);
        });

        it("terminates gracefully without throwing on unresolved or circular inputs", () => {
            const rawModel: RawOnnxModel = {
                graph: {
                    node: [
                        {
                            opType: "Add",
                            input: ["missing_A", "missing_B"],
                            output: ["missing_C"],
                        },
                    ],
                },
            };

            const graph = createGraph(rawModel);
            expect(graph).toBeDefined();
            expect(graph.getOperationNodes().length).toBe(1);
        });
    });
});
