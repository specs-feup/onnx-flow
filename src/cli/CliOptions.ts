import fs from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";
import yargs from "yargs";
import type { DecompositionOptions, IndexingMode } from "../DecompositionOptions.js";
import { defaultDecompositionOptions } from "../DecompositionOptions.js";

/**
 * Typed representation of parsed CLI arguments for `onnx-flow`.
 */
export interface CliArguments {
    inputFilePath: string;
    output?: string;
    format: "json" | "dot";
    verbosity: number;
    noLowLevel: boolean;
    noOptimize: boolean;
    noCodegen: boolean;
    noReconversion: boolean;
    visualization: number;
    interactive: boolean;
    fuse: boolean;
    coalesce: boolean;
    recurse: boolean;
    canonicalize: boolean;
    loopLowering: boolean;
    optimize: boolean;
    indexingMode: IndexingMode;
    tileSize?: number;
    targetOpset?: number;
    irVersion?: number;
    useScanOutputs?: boolean;
    formatter: "default" | "cgra";
    decomposeForCgra: boolean;
    checkEquivalence: boolean;
    partition?: string[];
    validate: boolean;
}

/**
 * Parses and validates CLI flags for `onnx-flow`, mapping them into strongly
 * typed `CliArguments` and `DecompositionOptions`.
 */
export class CliParser {
    /**
     * Resolves the package version by walking up from this module directory to
     * locate `package.json` (works both from `src/` and compiled `out/src/`).
     */
    public static resolvePackageVersion(): string {
        let cur = dirname(fileURLToPath(import.meta.url));
        for (let depth = 0; depth < 6; depth++) {
            const candidate = join(cur, "package.json");
            if (fs.existsSync(candidate)) {
                try {
                    const pkg = JSON.parse(fs.readFileSync(candidate, "utf8")) as {
                        version?: string;
                    };
                    if (typeof pkg.version === "string" && pkg.version.length > 0) {
                        return pkg.version;
                    }
                } catch {
                    break;
                }
            }
            const parent = dirname(cur);
            if (parent === cur) break;
            cur = parent;
        }
        return "unknown";
    }

    /**
     * Parses raw CLI arguments using yargs with strict option validation.
     */
    public static async parse(rawArgs: string[]): Promise<CliArguments> {
        const version = CliParser.resolvePackageVersion();

        const argv = await yargs(rawArgs)
            .exitProcess(false)
            .fail((msg: string, err: Error | undefined) => {
                if (err !== undefined) {
                    throw err;
                }
                throw new Error(msg);
            })
            .usage("Usage: onnx-flow <input_file> [options]")
            .version(version)
            .parserConfiguration({
                "short-option-groups": false,
                "camel-case-expansion": true,
                "boolean-negation": true,
                "duplicate-arguments-array": false,
            })
            .strictOptions()
            .demandCommand(1, "You need to provide an input file (ONNX or JSON)")
            .group(["partition"], "Partitioning Options (overrides Transformation Options):")
            .group(
                [
                    "canonicalize",
                    "fuse",
                    "coalesce",
                    "recurse",
                    "loopLowering",
                    "optimize",
                    "decomposeForCgra",
                    "indexingMode",
                    "tileSize",
                    "targetOpset",
                    "irVersion",
                    "noLowLevel",
                    "noOptimize",
                ],
                "Transformation Options:",
            )
            .group(
                ["output", "format", "visualization", "formatter", "interactive"],
                "Output & Visualization Options:",
            )
            .group(
                ["checkEquivalence", "verbosity", "noCodegen", "noReconversion"],
                "Development & Testing Options:",
            )
            .option("output", {
                alias: "o",
                describe: "Output resulting graph to a file",
                type: "string",
            })
            .option("format", {
                alias: "fm",
                describe: "Output format (json or dot)",
                type: "string",
                choices: ["json", "dot"] as const,
                default: "json" as const,
            })
            .option("verbosity", {
                alias: "v",
                describe: "Control verbosity (0 = silent, 1 = normal/outputs, 2 = verbose)",
                type: "number",
                default: 1,
            })
            .option("noLowLevel", {
                alias: "nl",
                describe: "Disable the low-level conversion",
                type: "boolean",
                default: false,
            })
            .option("noOptimize", {
                alias: "no",
                describe: "Disable optimization steps",
                type: "boolean",
                default: false,
            })
            .option("noCodegen", {
                alias: "nc",
                describe: "Disable code generation step",
                type: "boolean",
                default: false,
            })
            .option("noReconversion", {
                alias: "nr",
                describe: "Skip ONNX reconversion",
                type: "boolean",
                default: false,
            })
            .option("visualization", {
                alias: "vz",
                describe:
                    "Choose visualization option (0 = none, 1 = Graphviz online link, 2 = Graphviz server)",
                type: "number",
                default: 2,
            })
            .option("interactive", {
                alias: "i",
                describe: "Start the ONNX-Flow Explorer interactive REST API server",
                type: "boolean",
                default: false,
            })
            .option("fuse", {
                alias: "f",
                describe: "Fuse supported ops into a single Loop when possible",
                type: "boolean",
                default: defaultDecompositionOptions.fuse,
            })
            .option("coalesce", {
                alias: "c",
                describe: "Use coalesced scalar MAC for MatMul inside Loop bodies",
                type: "boolean",
                default: defaultDecompositionOptions.coalesce,
            })
            .option("recurse", {
                alias: "r",
                describe: "Recursively decompose inside generated Loop bodies",
                type: "boolean",
                default: defaultDecompositionOptions.recurse,
            })
            .option("canonicalize", {
                alias: "can",
                describe:
                    "Canonicalize complex operations into simpler primitives (use --no-canonicalize to disable)",
                type: "boolean",
                default: defaultDecompositionOptions.canonicalize,
            })
            .option("loopLowering", {
                alias: "ll",
                describe:
                    "Enable loop lowering (explicit Loop nodes); use --no-loop-lowering to disable",
                type: "boolean",
                default: defaultDecompositionOptions.loopLowering,
            })
            .option("optimize", {
                alias: "opt",
                describe: "Run optimization passes (DeadCodeElimination, ShapeOptimization, etc.)",
                type: "boolean",
                default: defaultDecompositionOptions.optimize,
            })
            .option("indexingMode", {
                alias: "im",
                describe:
                    "Loop indexing mode: 'pointer' (AGU-style math) or 'radix' (div/mod decoding)",
                type: "string",
                choices: ["pointer", "radix"] as const,
                default: defaultDecompositionOptions.indexingMode,
            })
            .option("tileSize", {
                alias: "ts",
                describe: "Tile size for loop strip-mining (disabled if <= 1)",
                type: "number",
                default: defaultDecompositionOptions.tileSize,
            })
            .option("targetOpset", {
                describe: "Target ONNX opset version for schema resolution & export (up to 28)",
                type: "number",
            })
            .option("irVersion", {
                describe: "Explicit ONNX IR version for export (up to 14)",
                type: "number",
            })
            .option("useScanOutputs", {
                describe:
                    "Optional Loop codegen: emit ONNX Loop scan_outputs instead of ScatterElements for non-self-carry loops",
                type: "boolean",
                default: defaultDecompositionOptions.useScanOutputs,
            })
            .option("formatter", {
                alias: "fmtr",
                describe: "Specify the DOT formatter to use (default or cgra)",
                type: "string",
                choices: ["default", "cgra"] as const,
                default: "default" as const,
            })
            .option("decomposeForCgra", {
                alias: "dgc",
                describe: "Decompose the graph for CGRA mapping",
                type: "boolean",
                default: false,
            })
            .option("checkEquivalence", {
                alias: "qe",
                describe: "Run ONNXRuntime equivalence check using test inputs (when available)",
                type: "boolean",
                default: false,
            })
            .option("partition", {
                alias: "pt",
                describe: "Partition graph into head/tail. Provide <nodeId> or <OpType> <Instance>",
                type: "string",
                array: true,
            })
            .option("validate", {
                alias: "val",
                describe: "Run structural validation after graph loading and transformations",
                type: "boolean",
                default: true,
            })
            .check((parsed) => {
                if (parsed.partition !== undefined) {
                    if (parsed.partition.length < 1 || parsed.partition.length > 2) {
                        throw new Error(
                            'Argument check failed: --partition requires either 1 argument "<nodeId>" or 2 arguments "<OpType> <Instance>")',
                        );
                    }
                }
                return true;
            })
            .help("help")
            .alias("help", "h")
            .describe("help", "Show help and usage instructions")
            .parseAsync(rawArgs);

        const inputFilePath = argv._[0];
        if (typeof inputFilePath !== "string") {
            throw new Error("The input file path must be a string.");
        }

        return {
            inputFilePath,
            format: argv.format,
            verbosity: argv.verbosity,
            noLowLevel: argv.noLowLevel,
            noOptimize: argv.noOptimize,
            noCodegen: argv.noCodegen,
            noReconversion: argv.noReconversion,
            visualization: argv.visualization,
            interactive: argv.interactive,
            fuse: argv.fuse,
            coalesce: argv.coalesce,
            recurse: argv.recurse,
            canonicalize: argv.canonicalize,
            loopLowering: argv.loopLowering,
            optimize: argv.optimize,
            indexingMode: argv.indexingMode as IndexingMode,
            formatter: argv.formatter,
            decomposeForCgra: argv.decomposeForCgra,
            checkEquivalence: argv.checkEquivalence,
            validate: argv.validate,
            ...(argv.output !== undefined ? { output: argv.output } : {}),
            ...(argv.tileSize !== undefined ? { tileSize: argv.tileSize } : {}),
            ...(argv.targetOpset !== undefined ? { targetOpset: argv.targetOpset } : {}),
            ...(argv.irVersion !== undefined ? { irVersion: argv.irVersion } : {}),
            ...(argv.useScanOutputs !== undefined ? { useScanOutputs: argv.useScanOutputs } : {}),
            ...(argv.partition !== undefined ? { partition: argv.partition } : {}),
        };
    }

    /**
     * Converts parsed CLI arguments into a `DecompositionOptions` object.
     */
    public static toDecompositionOptions(argv: CliArguments): DecompositionOptions {
        return {
            canonicalize: argv.canonicalize,
            fuse: argv.fuse,
            recurse: argv.recurse,
            coalesce: argv.coalesce,
            optimize: argv.optimize,
            decomposeForCgra: argv.decomposeForCgra,
            loopLowering: argv.loopLowering,
            indexingMode: argv.indexingMode,
            ...(argv.useScanOutputs !== undefined ? { useScanOutputs: argv.useScanOutputs } : {}),
            ...(argv.tileSize !== undefined ? { tileSize: argv.tileSize } : {}),
            ...(argv.targetOpset !== undefined ? { targetOpset: argv.targetOpset } : {}),
            ...(argv.irVersion !== undefined ? { irVersion: argv.irVersion } : {}),
        };
    }
}
