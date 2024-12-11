#!/usr/bin/env node

import { createGraph } from './initGraph.js';
import OnnxGraphTransformer from './Onnx/transformation/LowLevelTransformation/LowLevelConversion.js';
import OnnxGraphOptimizer from './Onnx/transformation/OptimizeForDimensions/OptimizeForDimensions.js';
import OnnxDotFormatter from "./Onnx/dot/OnnxDotFormatter.js";
import { generateCode } from './codeGeneration.js';
import { onnx2json } from './onnx2json.js';
import fs from 'fs';
import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';

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
  .help()
  .argv;

const inputFilePath = argv._[0]; // Positional argument for input file

if (typeof inputFilePath !== 'string') {
  throw new Error('The input file path must be a string.');
}

const verbosity = argv.verbosity;
const outputFilePath = argv.output;
const outputFormat = argv.format;
const dotFormatter = new OnnxDotFormatter();

(async function main() {
  try {
    let onnxObject;

    // Step 1: Load the input (either ONNX or pre-existing JSON graph)
    if (inputFilePath.endsWith('.json')) {
      onnxObject = JSON.parse(fs.readFileSync(inputFilePath, 'utf8'));
    } else {
      onnxObject = await onnx2json(inputFilePath);
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
      const generatedCode = generateCode(graph);
      if (verbosity > 0) console.log('Generated Code:', generatedCode);
    }

  } catch (error) {
    console.error('Error:', error);
  }
})();