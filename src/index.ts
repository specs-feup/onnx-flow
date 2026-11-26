#!/usr/bin/env node

import fs from "fs";
import { resolve } from "path";
import { fileURLToPath } from "url";
import { hideBin } from "yargs/helpers";
import { createGraph } from "./io/OnnxGraphLoader.js";
import { onnx2json } from "./io/OnnxToJson.js";
import { json2onnx } from "./io/JsonToOnnx.js";
import { convertFlowGraphToOnnxJson } from "./io/OnnxGraphExporter.js";
import OnnxDotFormatter from "./onnx/dot/OnnxDotFormatter.js";
import CgraDotFormatter from "./onnx/dot/CgraDotFormatter.js";
import { BASE_TEN, safeWriteJson } from "./onnx/utils/index.js";
import type { DecompositionOptions, IndexingMode } from "./DecompositionOptions.js";
import { defaultDecompositionOptions } from "./DecompositionOptions.js";
import { splitByAncestor } from "./onnx/partitioning/Strategies.js";
import { partitionGraph } from "./onnx/partitioning/Partition.js";
import type OnnxGraph from "./onnx/ir/OnnxGraph.js";
import validateGraph from "./onnx/validation/ValidateGraph.js";
import type { RawOnnxModel } from "./onnx/ir/OnnxTypes.js";
import OnnxGraphTransformer from "./onnx/transformation/index.js";
import { startExplorerServer } from "./explorer/ExplorerServer.js";
import { ExplorerSession } from "./explorer/ExplorerSession.js";
import inferShapes from "./onnx/schema/InferShapes.js";
import { generateUnifiedExplorerJson } from "./explorer/ExplorerSerializer.js";
import { CliParser } from "./cli/CliOptions.js";
import { GraphvizVisualizer } from "./cli/GraphvizServer.js";

/** Decodes a binary `.onnx` file into a raw JSON-compatible ONNX model object. */
export async function parseOnnxFile(inputFilePath: string): Promise<RawOnnxModel> {
    return await onnx2json(inputFilePath);
}

/** Encodes a JSON ONNX model file back into binary `.onnx` format. */
export async function jsonToOnnx(
    jsonFilePath: string,
    outputFilePath: string,
    options?: { targetOpset?: number; irVersion?: number },
): Promise<void> {
    return await json2onnx(jsonFilePath, outputFilePath, options);
}

/** Strips the file extension from a path without losing directory prefixes. */
function getBasePath(filePath: string): string {
    const extIndex = filePath.lastIndexOf(".");
    return extIndex === -1 ? filePath : filePath.slice(0, extIndex);
}

function getReconvertedPaths(inputPath: string): { json: string; onnx: string } {
    const reconvertedBase = `${getBasePath(inputPath)}_reconverted`;
    return {
        json: `${reconvertedBase}.json`,
        onnx: `${reconvertedBase}.onnx`,
    };
}

function logGraphRepresentation(
    label: string,
    graph: OnnxGraph.Class,
    outputFormat: "json" | "dot",
    dotFormatter: OnnxDotFormatter | CgraDotFormatter,
): void {
    if (outputFormat === "json") {
        console.log(`${label} in JSON Format:`, JSON.stringify(graph.toCy().json(), null, 2));
    } else {
        console.log(`${label} in DOT Format:`, graph.toString(dotFormatter));
    }
}

function writeGraphOutput(
    filePath: string,
    graph: OnnxGraph.Class,
    outputFormat: "json" | "dot",
    dotFormatter: OnnxDotFormatter | CgraDotFormatter,
): void {
    if (outputFormat === "json") {
        const unifiedJson = generateUnifiedExplorerJson(graph);
        fs.writeFileSync(filePath, JSON.stringify(unifiedJson, null, 2));
    } else {
        fs.writeFileSync(filePath, graph.toString(dotFormatter));
    }
}

/**
 * Loads a raw ONNX model into an `OnnxGraph`, optionally runs transformations and
 * structural validation, and returns either the graph or its DOT representation.
 */
export function loadGraph(
    onnxObject: RawOnnxModel,
    enableLowLevel: boolean = true,
    enableOptimize: boolean = true,
    dotOutput: boolean = true,
    canonicalize: boolean = defaultDecompositionOptions.canonicalize,
    fuse: boolean = defaultDecompositionOptions.fuse,
    recurse: boolean = defaultDecompositionOptions.recurse,
    coalesce: boolean = defaultDecompositionOptions.coalesce,
    loopLowering: boolean = defaultDecompositionOptions.loopLowering,
    optimize: boolean = defaultDecompositionOptions.optimize,
    decomposeForCgra: boolean = defaultDecompositionOptions.decomposeForCgra,
    indexingMode: IndexingMode = defaultDecompositionOptions.indexingMode,
    validate: boolean = true,
    tileSize: number | undefined = defaultDecompositionOptions.tileSize,
    targetOpset?: number,
    irVersion?: number,
): string | OnnxGraph.Class {
    let graph = createGraph(onnxObject, targetOpset);
    if (targetOpset !== undefined) {
        graph.setTargetOpset(targetOpset, true);
    }
    if (irVersion !== undefined) {
        graph.setIrVersion(irVersion, true);
    }

    if (validate) {
        validateGraph(graph);
    }

    if (enableLowLevel) {
        const decompOptions: DecompositionOptions = {
            canonicalize,
            fuse,
            recurse,
            coalesce,
            loopLowering,
            optimize,
            decomposeForCgra,
            indexingMode,
            ...(tileSize !== undefined ? { tileSize } : {}),
            ...(targetOpset !== undefined ? { targetOpset } : {}),
            ...(irVersion !== undefined ? { irVersion } : {}),
        };
        graph = graph.apply(new OnnxGraphTransformer(decompOptions));

        if (validate) {
            validateGraph(graph);
        }
    }

    if (enableLowLevel && enableOptimize && validate) {
        validateGraph(graph);
    }

    if (dotOutput) {
        return graph.toString(decomposeForCgra ? new CgraDotFormatter() : new OnnxDotFormatter());
    }
    return graph;
}

/** Renders a DOT graph description into SVG markup. */
export async function renderDotToSVG(dotGraph: string): Promise<string> {
    return await GraphvizVisualizer.renderDotToSVG(dotGraph);
}

/** Generates a Graphviz Online URL for the given DOT graph string. */
export function generateGraphvizOnlineLink(dotGraph: string): string {
    return GraphvizVisualizer.generateOnlineLink(dotGraph);
}

/** Placeholder hook for downstream code generation. */
export function generateGraphCode(_graph: OnnxGraph.Class): string {
    return "TODO";
}

/**
 * Main CLI execution routine for `onnx-flow`.
 */
export async function runCli(rawArgs: string[] = hideBin(process.argv)): Promise<void> {
    const argv = await CliParser.parse(rawArgs);
    const {
        inputFilePath,
        verbosity,
        targetOpset,
        irVersion,
        output: outputFilePath,
        format: outputFormat,
        visualization: visualizationOption,
        validate,
    } = argv;
    const dotFormatter =
        argv.formatter === "cgra" ? new CgraDotFormatter() : new OnnxDotFormatter();

    // Partitioning overrides transformation passes to keep split points recognizable.
    const isPartitioning = argv.partition !== undefined && argv.partition.length > 0;
    const noLowLevel = isPartitioning ? true : argv.noLowLevel;

    // 1. Load input model (ONNX binary or pre-parsed JSON)
    const onnxObject: RawOnnxModel = inputFilePath.endsWith(".json")
        ? (JSON.parse(fs.readFileSync(inputFilePath, "utf8")) as RawOnnxModel)
        : await parseOnnxFile(inputFilePath);

    if (verbosity > 1) {
        console.log("Input ONNX/JSON Graph:", JSON.stringify(onnxObject, null, 2));
    }

    // 2. Build internal OnnxGraph representation
    const graph = createGraph(onnxObject, targetOpset);
    if (targetOpset !== undefined) {
        graph.setTargetOpset(targetOpset, true);
    }
    if (irVersion !== undefined) {
        graph.setIrVersion(irVersion, true);
    }

    if (verbosity > 1) {
        logGraphRepresentation("Initial Graph", graph, outputFormat, dotFormatter);
    }

    const decompOptions = CliParser.toDecompositionOptions(argv);

    // 3. Interactive explorer mode (bypasses automatic batch passes)
    if (argv.interactive) {
        console.log("\nStarting Interactive Mode. Automatic passes bypassed.");
        const session = new ExplorerSession(graph, decompOptions);
        startExplorerServer(session, 3000);
        return;
    }

    // 4. Transformation pipeline
    if (!noLowLevel) {
        graph.apply(new OnnxGraphTransformer(decompOptions));

        if (validate) {
            try {
                if (verbosity > 0) console.log("Running Graph Validation...");
                validateGraph(graph);
                if (verbosity > 0) console.log("Graph Validation Passed.");
            } catch (error) {
                console.error("Graph Validation Failed:");
                throw error;
            }
        }
    }

    if (verbosity > 1) {
        logGraphRepresentation("Low-level Graph", graph, outputFormat, dotFormatter);
    }

    // 5. Optional partitioning into head / tail graphs
    let partitions: { head: OnnxGraph.Class; tail: OnnxGraph.Class } | undefined;
    if (isPartitioning && argv.partition !== undefined) {
        const targetNodeId = resolvePartitionTargetNodeId(graph, argv.partition);
        if (verbosity > 0) console.log(`Partitioning graph at node: ${targetNodeId}`);

        const sets = splitByAncestor(graph, targetNodeId);
        partitions = partitionGraph(graph, sets);

        const exportPartition = async (g: OnnxGraph.Class, suffix: string): Promise<void> => {
            const base = getBasePath(outputFilePath ?? inputFilePath);
            const fileName = `${base}_${suffix}`;
            inferShapes(g);

            if (!argv.noReconversion) {
                const onnxJson = convertFlowGraphToOnnxJson(g, `${suffix}_graph`);
                safeWriteJson(`${fileName}.onnx.json`, onnxJson);
                await jsonToOnnx(`${fileName}.onnx.json`, `${fileName}.onnx`);
                if (verbosity > 0) console.log(`Created ONNX partition: ${fileName}.onnx`);
            }

            writeGraphOutput(`${fileName}.${outputFormat}`, g, outputFormat, dotFormatter);
        };

        await exportPartition(partitions.head, "head");
        await exportPartition(partitions.tail, "tail");
    }

    // 6. Write unpartitioned output file
    if (outputFilePath !== undefined && !isPartitioning) {
        writeGraphOutput(outputFilePath, graph, outputFormat, dotFormatter);
        if (verbosity > 0) {
            console.log(`Output Graph written to ${outputFilePath} in ${outputFormat} format`);
        }
    }

    // 7. Reconvert graph back to ONNX JSON & binary
    if (!argv.noReconversion && !isPartitioning) {
        const { json: reconvertedJsonPath, onnx: reconvertedOnnxPath } =
            getReconvertedPaths(inputFilePath);
        const exportOptions = {
            ...(targetOpset !== undefined ? { targetOpset } : {}),
            ...(irVersion !== undefined ? { irVersion } : {}),
        };
        const onnxCompatibleJson = convertFlowGraphToOnnxJson(
            graph,
            undefined,
            undefined,
            exportOptions,
        );

        safeWriteJson(reconvertedJsonPath, onnxCompatibleJson);
        console.log(`Reconverted JSON written to ${reconvertedJsonPath}`);

        await jsonToOnnx(reconvertedJsonPath, reconvertedOnnxPath, exportOptions);
        console.log(`Reconverted ONNX written to ${reconvertedOnnxPath}`);
    } else if (verbosity > 0 && !isPartitioning) {
        console.log("Skipping ONNX reconversion.");
    }

    // 8. Print output representation to stdout
    if (verbosity > 0) {
        if (isPartitioning && partitions !== undefined) {
            logGraphRepresentation(
                "Output Head Graph",
                partitions.head,
                outputFormat,
                dotFormatter,
            );
            logGraphRepresentation(
                "Output Tail Graph",
                partitions.tail,
                outputFormat,
                dotFormatter,
            );
        } else {
            logGraphRepresentation("Output Graph", graph, outputFormat, dotFormatter);
        }
    }

    // 9. Optional ONNXRuntime equivalence verification
    if (argv.checkEquivalence) {
        await runOptionalEquivalenceCheck(inputFilePath, argv.noReconversion);
    }

    // 10. Optional Graphviz visualization
    if (visualizationOption > 0) {
        const getDot = (): string =>
            partitions !== undefined
                ? GraphvizVisualizer.buildPartitionedDot(
                      partitions.head,
                      partitions.tail,
                      dotFormatter,
                  )
                : graph.toString(dotFormatter);

        if (visualizationOption === 1) {
            console.log("Graphviz Online Link:", generateGraphvizOnlineLink(getDot()));
        } else {
            GraphvizVisualizer.startServer(getDot, 8080);
        }
    }
}

function resolvePartitionTargetNodeId(graph: OnnxGraph.Class, partitionArgs: string[]): string {
    if (partitionArgs.length === 1) {
        const nodeId = partitionArgs[0];
        if (!graph.hasNode(nodeId)) {
            throw new Error(`Could not find node with id '${nodeId}' in graph.`);
        }
        return nodeId;
    }

    const opType = partitionArgs[0];
    const instance = parseInt(partitionArgs[1], BASE_TEN);
    if (isNaN(instance)) {
        throw new Error(`Invalid instance number: ${partitionArgs[1]}`);
    }
    const resolvedNode = graph.getNodeByTypeAndInstance(opType, instance);
    if (resolvedNode === undefined) {
        throw new Error(`Could not find instance ${instance} of operation '${opType}'`);
    }
    return resolvedNode.id;
}

async function runOptionalEquivalenceCheck(
    inputFilePath: string,
    noReconversion: boolean,
): Promise<void> {
    const { onnx: reconvertedOnnxPath } = getReconvertedPaths(inputFilePath);
    if (noReconversion && !fs.existsSync(reconvertedOnnxPath)) {
        console.log(
            `--checkEquivalence requested but reconversion was disabled and ` +
                `no reconverted model was found at ${reconvertedOnnxPath}. Skipping equivalence check.`,
        );
        return;
    }

    try {
        const { runEquivalenceForOriginalPath } = await import("../test/helpers/modelRegistry.js");
        const result = await runEquivalenceForOriginalPath(inputFilePath, { skipCli: true });

        if (result === "no-config") {
            console.log(
                `No test input specification found for '${inputFilePath}'. ` +
                    `Equivalence check skipped without affecting the rest of the run.`,
            );
        } else {
            console.log(`Equivalence check result for '${inputFilePath}': ${result}`);
        }
    } catch (error) {
        console.error("❌ Failed to run equivalence check:", error);
    }
}

function isDirectExecution(): boolean {
    if (!process.argv[1]) return false;
    try {
        const entry = fs.realpathSync(resolve(process.argv[1])).toLowerCase();
        const current = fs.realpathSync(fileURLToPath(import.meta.url)).toLowerCase();
        return entry === current;
    } catch {
        return false;
    }
}

if (isDirectExecution()) {
    runCli().catch((error) => {
        console.error(error instanceof Error ? error.message : error);
        process.exit(1);
    });
}
