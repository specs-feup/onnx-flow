#!/usr/bin/env node

import fs from 'fs';
import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import express, { Request, Response } from "express";
import { graphviz } from "node-graphviz";
import { createGraph } from './initGraph.js';
import OnnxGraphTransformer from './Onnx/transformation/loop-lowering/index.js';
import OnnxGraphOptimizer from './Onnx/transformation/shape-optimization/index.js';
import OnnxDotFormatter from "./Onnx/dot/OnnxDotFormatter.js";
import CgraDotFormatter from "./Onnx/dot/CgraDotFormatter.js";
import { generateCode } from './codeGeneration.js';
import { onnx2json } from './onnx2json.js';
import { json2onnx } from "./json2onnx.js";
import { convertFlowGraphToOnnxJson } from "./flow2json.js";
import { safeWriteJson } from './Onnx/Utils.js';
import { DecompositionOptions, defaultDecompositionOptions } from './DecompositionOptions.js';


export async function parseOnnxFile(inputFilePath: string){
  return await onnx2json(inputFilePath);
}

export async function jsonToOnnx(jsonFilePath: string, outputFilePath: string){
  return await json2onnx(jsonFilePath, outputFilePath);
}

export function loadGraph(
  onnxObject: any,
  enableLowLevel: boolean = true,
  enableOptimize: boolean = true,
  dotOutput: boolean = true,
  fuse: boolean = defaultDecompositionOptions.fuse,
  recurse: boolean = defaultDecompositionOptions.recurse,
  coalesce: boolean = defaultDecompositionOptions.coalesce,
  loopLowering: boolean = defaultDecompositionOptions.loopLowering,
  decomposeForCgra: boolean = defaultDecompositionOptions.decomposeForCgra
) {
  let graph = createGraph(onnxObject);

  if (enableLowLevel) {
    const decompOptions: DecompositionOptions = { fuse, recurse, coalesce, loopLowering, decomposeForCgra };
    graph = graph.apply(new OnnxGraphTransformer(decompOptions));
  }

  if (enableLowLevel && enableOptimize) {
    graph = graph.apply(new OnnxGraphOptimizer());
  }

  if (dotOutput) {
    return graph.toString(dotFormatter);
  }
  return graph;
}

export async function renderDotToSVG(dotGraph: string): Promise<string> {
  return await graphviz.layout(dotGraph);
}

export function generateGraphvizOnlineLink(dotGraph: string): string {
  const baseUrl = "https://dreampuf.github.io/GraphvizOnline/#";
  return baseUrl + encodeURIComponent(dotGraph);
}

export function generateGraphCode(graph: any): string {

  return generateCode(graph);
}


const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

let version = 'unknown';
try {
  const packageJson = JSON.parse(
    fs.readFileSync(join(__dirname, '..', '..', 'package.json'), 'utf8')
  );
  version = packageJson.version;
} catch (error) {
  console.warn('Warning: Unable to read package.json for version info.', error);
}

const argv = await yargs(hideBin(process.argv))
  .usage('Usage: onnx-flow <input_file> [options]')
  .version(version)
  .parserConfiguration({
    'short-option-groups': false,
    'camel-case-expansion': true,
    'boolean-negation': true,
    'duplicate-arguments-array': false,
  })
  .strictOptions()
  .demandCommand(1, 'You need to provide an input file (ONNX or JSON)')
  .option('output', {
    alias: 'o',
    describe: 'Output resulting graph to a file',
    type: 'string',
  })
  .option('format', {
    alias: 'fm',
    describe: 'Output format (json or dot)',
    type: 'string',
    choices: ['json', 'dot'],
    default: 'json',
  })
  .option('verbosity', {
    alias: 'v',
    describe: 'Control verbosity (0 = silent, 1 = normal/outputs, 2 = verbose)',
    type: 'number',
    default: 1,
  })
  .option('noLowLevel', {
    alias: 'nl',
    describe: 'Disable the low-level conversion',
    type: 'boolean',
    default: false,
  })
  .option('noOptimize', {
    alias: 'no',
    describe: 'Disable optimization steps',
    type: 'boolean',
    default: false,
  })
  .option('noCodegen', {
    alias: 'nc',
    describe: 'Disable code generation step',
    type: 'boolean',
    default: false,
  })
  .option('noReconversion', {
    alias: 'nr',
    describe: 'Skip ONNX reconversion',
    type: 'boolean',
    default: false,
  })
  .option('visualization', {
    alias: 'vz',
    describe: 'Choose visualization option (0 = none, 1 = Graphviz online link, 2 = Graphviz server)',
    type: 'number',
    default: 2,
  })
  .option('fuse', {
    alias: 'f',
    describe: 'Fuse supported ops into a single Loop when possible',
    type: 'boolean',
    default: defaultDecompositionOptions.fuse,
  })
  .option('coalesce', {
    alias: 'c',
    describe: 'Use coalesced scalar MAC for MatMul inside Loop bodies',
    type: 'boolean',
    default: defaultDecompositionOptions.coalesce,
  })
  .option('recurse', {
    alias: 'r',
    describe: 'Recursively decompose inside generated Loop bodies',
    type: 'boolean',
    default: defaultDecompositionOptions.recurse,
  })
    .option('loopLowering', {
    alias: 'll',
    describe: 'Enable loop lowering (explicit Loop nodes); use --no-loop-lowering to disable',
    type: 'boolean',
    default: defaultDecompositionOptions.loopLowering,
  })
    .option("formatter", {
    alias: "fmtr",
    describe: "Specify the DOT formatter to use (0 = default, 1 = cgra)",
    type: "string",
    choices: ["default", "cgra"],
    default: "default",
  })
  .option("decomposeForCgra", {
    alias: "dgc",
    describe: "Decompose the graph for CGRA mapping",
    type: "boolean",
    default: false,
  })
    .option('checkEquivalence', {
    alias: 'qe',
    describe: 'Run ONNXRuntime equivalence check using test inputs (when available)',
    type: 'boolean',
    default: false,
  })
  .help()
  .argv;

const inputFilePath = argv._[0]; // Positional argument for input file

if (typeof inputFilePath !== 'string') {
  throw new Error('The input file path must be a string.');
}

const verbosity = argv.verbosity;
const outputFilePath = argv.output;
const outputFormat = argv.format;
const visualizationOption = argv.visualization;
const dotFormatter =
  argv.formatter === "cgra" ? new CgraDotFormatter() : new OnnxDotFormatter();


(async function main() {
  try {
    let onnxObject;

    // Step 1: Load the input (either ONNX or pre-existing JSON graph)
    if (inputFilePath.endsWith('.json')) {
      onnxObject = JSON.parse(fs.readFileSync(inputFilePath, 'utf8'));
    } else {
      onnxObject = await parseOnnxFile(inputFilePath);
      //fs.writeFileSync("./examples/onnxmodel.json", JSON.stringify(onnxObject, null, 2));
    }

    if (verbosity > 1) console.log('Input ONNX/JSON Graph:', JSON.stringify(onnxObject, null, 2));

    // Step 2: Process the graph
    let graph = createGraph(onnxObject);

    if (verbosity > 1){
      if (outputFormat === 'json') {
        console.log('Initial Graph in JSON Format:', JSON.stringify(graph.toCy().json(), null, 2));
      } else if (outputFormat === 'dot') {
        console.log('Initial Graph in DOT Format:', graph.toString(dotFormatter));
      }
    }

    if (!argv.noLowLevel) {
      const decompOptions: DecompositionOptions = {
        fuse: argv.fuse,
        recurse: argv.recurse,
        coalesce: argv.coalesce,
        decomposeForCgra: argv.decomposeForCgra,
        loopLowering: argv.loopLowering,
      };
      graph.apply(new OnnxGraphTransformer(decompOptions));
    }

    if (verbosity > 1){
      if (outputFormat === 'json') {
        console.log('Low-level Graph in JSON Format:', JSON.stringify(graph.toCy().json(), null, 2));
      } else if (outputFormat === 'dot') {
        console.log('Low-level Graph in DOT Format:', graph.toString(dotFormatter));
      }
    }

    if (!argv.noLowLevel && !argv.noOptimize) {
      graph.apply(new OnnxGraphOptimizer());
    }

    // Step 3: Output the graph if requested
    if (outputFilePath) {
      if (outputFormat === 'json') {
        fs.writeFileSync(outputFilePath, JSON.stringify(graph.toCy().json(), null, 2));
      } else if (outputFormat === 'dot') {
        fs.writeFileSync(outputFilePath, graph.toString(dotFormatter));
      }
      if (verbosity > 0) console.log(`Output Graph written to ${outputFilePath} in ${outputFormat} format`);
    }

    function getReconvertedPaths(inputPath: string): { json: string; onnx: string } {
      const extIndex = inputPath.lastIndexOf('.');
      const base = extIndex === -1 ? inputPath : inputPath.slice(0, extIndex);
      const reconvertedBase = base + '_reconverted';

      return {
        json: reconvertedBase + '.json',
        onnx: reconvertedBase + '.onnx',
      };
    }


    // Convert the ONNX JSON format to ONNX binary format if not disabled
    if (!argv.noReconversion) {
      const { json: reconvertedJsonPath, onnx: reconvertedOnnxPath } = getReconvertedPaths(inputFilePath);
      const onnxCompatibleJson = convertFlowGraphToOnnxJson(graph);

      // Use streaming writer to avoid RangeError for huge SC* models
      safeWriteJson(reconvertedJsonPath, onnxCompatibleJson);
      console.log(`Reconverted JSON written to ${reconvertedJsonPath}`);

      await jsonToOnnx(reconvertedJsonPath, reconvertedOnnxPath);
      console.log(`Reconverted ONNX written to ${reconvertedOnnxPath}`);
    }
    else if (verbosity > 0) {
      console.log('Skipping ONNX reconversion.');
    }

    // Print the output graph to stdout
    if (verbosity > 0) {
      if (outputFormat === 'json') {
        console.log('Output Graph in JSON Format:', JSON.stringify(graph.toCy().json(), null, 2));
      } else if (outputFormat === 'dot') {
        console.log('Output Graph in DOT Format:', graph.toString(dotFormatter));
      }
    }

    // Step 4: Code generation
    if (!argv.noLowLevel && !argv.noCodegen) {
      const generatedCode = generateGraphCode(graph);
      if (verbosity > 0) console.log('Generated Code:', generatedCode);
    }

    // Optional: run ORT equivalence check using the test metadata, if requested.
    if (argv.checkEquivalence) {
      const { onnx: reconvertedOnnxPath } = getReconvertedPaths(inputFilePath);

      // If reconversion was disabled and we don’t already have a reconverted file, just skip.
      if (argv.noReconversion && !fs.existsSync(reconvertedOnnxPath)) {
        console.log(
          `--checkEquivalence requested but reconversion was disabled and ` +
          `no reconverted model was found at ${reconvertedOnnxPath}. Skipping equivalence check.`
        );
      } else {
        try {
          // Dynamic import so that compatibility_test.ts is only loaded if needed
          const { runEquivalenceForOriginalPath } = await import('../test/compatibility_test.js');

          const result = await runEquivalenceForOriginalPath(inputFilePath, {
            skipCli: true,   // reconversion was already done by this CLI run
          });

          if (result === 'no-config') {
            // Exactly the behaviour you described for “we don’t know the input information”
            console.log(
              `No test input specification found for '${inputFilePath}'. ` +
              `Equivalence check skipped without affecting the rest of the run.`
            );
          } else {
            console.log(`Equivalence check result for '${inputFilePath}': ${result}`);
          }
        } catch (e) {
          console.error('❌ Failed to run equivalence check:', e);
        }
      }
    }

    // Step 5: Graphviz Online link generation
    if (visualizationOption > 0) {
      if (visualizationOption == 1){
        console.log('Graphviz Online Link:', generateGraphvizOnlineLink(graph.toString(dotFormatter)));
      }
      else{
        const app = express();
        const port = 8080;

        // Serve HTML with embedded SVG at the root ("/")
        app.get("/", async (req: Request, res: Response) => {
          try {
            // Render the DOT graph to SVG
            const svgContent = await renderDotToSVG(graph.toString(dotFormatter));

            // Create an HTML page with the embedded SVG
            const htmlContent = `
              <!DOCTYPE html>
              <html lang="en">
              <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>onnx-flow Graphviz Visualization</title>
                <style>
                  /* General Reset */
                  body, html {
                    margin: 0;
                    padding: 0;
                    height: 100vh; /* Set height to 100% of the viewport */
                    overflow: hidden; /* Remove page scroll bar */
                    font-family: Arial, sans-serif;
                    background-color: #f9f9f9; /* Light background */
                    text-align: center;
                  }
                  /* Title Styling */
                  h1 {
                    margin: 20px 0; /* Top margin */
                    font-size: 2.5em;
                    color: #333;
                  }
                  /* Container for the graph */
                  .graph-container {
                    max-width: 90%; /* Prevent horizontal overflow */
                    height: calc(100vh - 160px); /* Subtract title height from viewport height */
                    margin: 0 auto; /* Center horizontally */
                    overflow: auto; /* Enable scrolling inside the container */
                    border: 1px solid #ddd;
                    padding: 10px;
                    background-color: #fff;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                    border-radius: 8px;
                  }
                  /* SVG Responsiveness */
                  svg {
                    max-width: 100%; /* Scale the graph to fit the container */
                    height: auto;
                  }
                </style>
              </head>
              <body>
                <h1>onnx-flow Graphviz Visualization</h1>
                <div class="graph-container">
                  ${svgContent}
                </div>
              </body>
              </html>
            `;


            // Send the HTML response
            res.setHeader("Content-Type", "text/html");
            res.send(htmlContent);

          } catch (error) {
            console.error("Error rendering graph:", error);
            res.status(500).send(error);
          }
        });

        // Start the server
        app.listen(port, () => {
          console.log(`Graphviz Visualization server running at http://localhost:${port}`);
        });
      }
    }

  } catch (error) {
    console.error('Error:', error);
  }
})();