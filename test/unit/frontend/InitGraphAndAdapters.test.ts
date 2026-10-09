import { describe, it, expect } from "vitest";
import { applyAdapters } from "../../../src/Onnx/Frontend/Adapters.js";
import { createGraph } from "../../../src/initGraph.js";
import { DataType, AttributeType, type RawOnnxModel } from "../../../src/Onnx/OnnxTypes.js";
import { initializeSchemaRegistry } from "../../../src/Onnx/Schema/index.js";
import { makeTensorProto } from "../../../src/Onnx/Utils.js";
import RegionArgumentNode from "../../../src/Onnx/RegionArgumentNode.js";
import ConstantNode from "../../../src/Onnx/ConstantNode.js";

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
