#!/usr/bin/env node

import express, { Request, Response } from "express";
import { graphviz } from "node-graphviz";
import { createGraph } from './initGraph.js';
import OnnxGraphTransformer from './Onnx/transformation/LowLevelTransformation/LowLevelConversion.js';
import OnnxGraphOptimizer from './Onnx/transformation/OptimizeForDimensions/OptimizeForDimensions.js';
import OnnxDotFormatter from "./Onnx/dot/OnnxDotFormatter.js";
import { generateCode } from './codeGeneration.js';
import { onnx2json } from './onnx2json.js';
import fs from 'fs';
import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';

export async function onnxFileParser(inputFilePath: string){
  return await onnx2json(inputFilePath);
}

export function loadGraph(onnxObject: any, enableLowLevel: boolean = true, enableOptimize: boolean = true) {
  let graph = createGraph(onnxObject);

  if (enableLowLevel) {
    graph = graph.apply(new OnnxGraphTransformer());
  }

  if (enableOptimize) {
    graph = graph.apply(new OnnxGraphOptimizer());
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

const argv = await yargs(hideBin(process.argv))
  .usage('Usage: onnx2cytoscape <input_file> [options]')
  .demandCommand(1, 'You need to provide an input file (ONNX or JSON)')
  .option('output', {
    alias: 'o',
    describe: 'Output resulting graph to a file',
    type: 'string',
  })
  .option('format', {
    alias: 'f',
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
  .option('visualization', {
    alias: 'vz',
    describe: 'Choose visualization option (0 = none, 1 = Graphviz online link, 2 = Graphviz server)',
    type: 'number',
    default: 2,
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
const dotFormatter = new OnnxDotFormatter();

(async function main() {
  try {
    let onnxObject;

    // Step 1: Load the input (either ONNX or pre-existing JSON graph)
    if (inputFilePath.endsWith('.json')) {
      onnxObject = JSON.parse(fs.readFileSync(inputFilePath, 'utf8'));
    } else {
      onnxObject = await onnxFileParser(inputFilePath);
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

    if(!argv.noLowLevel){
      graph.apply(new OnnxGraphTransformer());
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

    // Step 5: Graphviz Online link generation
    if (visualizationOption) {
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
                <title>Graphviz Visualization</title>
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
                <h1>Graphviz Visualization</h1>
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