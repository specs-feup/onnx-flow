#!/usr/bin/env node

import express from 'express';
import http from 'http';
import { WebSocketServer } from 'ws';
import { createGraph } from './initGraph.js';
import { transformOpps, optimizeForDimensions } from './transformOpps.js';
import { generateCode } from './codeGeneration.js';
import { onnx2json } from './onnx2json.js';
import fs from 'fs';
import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';

const argv = yargs(hideBin(process.argv))
  .usage('Usage: $0 <input_file> [options]')
  .demandCommand(1, 'You need to provide an input file (ONNX or JSON)')
  .option('output', {
    alias: 'o',
    describe: 'Output resulting graph to a JSON file',
    type: 'string',
  })
  .option('verbosity', {
    alias: 'v',
    describe: 'Control verbosity (0 = silent, 1 = normal/outputs, 2 = verbose)',
    type: 'number',
    default: 1,
  })
  .option('no-optimize', {
    describe: 'Disable optimization steps',
    type: 'boolean',
    default: false,
  })
  .option('no-codegen', {
    describe: 'Disable code generation step',
    type: 'boolean',
    default: false,
  })
  .option('no-visualization', {
    describe: 'Disable web visualization',
    type: 'boolean',
    default: false,
  })
  .help()
  .argv;

const inputFilePath = argv._[0]; // Positional argument for input file
const verbosity = argv.verbosity;
const outputFilePath = argv.output;

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
    let cy = createGraph(onnxObject);
    if (verbosity > 1) console.log('Generated Cytoscape Graph:', JSON.stringify(cy.json(), null, 2));

    if (!argv.noOptimize) {
      transformOpps(cy);
      optimizeForDimensions(cy);
    }

    if (!argv.noCodegen) {
      const generatedCode = generateCode(cy, onnxObject);
      if (verbosity > 0) console.log('Generated Code:', generatedCode);
    }

    // Step 3: Output the graph if requested
    if (outputFilePath) {
      fs.writeFileSync(outputFilePath, JSON.stringify(cy.json(), null, 2));
      if (verbosity > 0) console.log(`Output Graph JSON written to ${outputFilePath}`);
    }
    if (verbosity > 0) console.log('Output Cytoscape Graph:', JSON.stringify(cy.json(), null, 2));

    // Step 4: Serve visualization if enabled
    if (!argv.noVisualization) {
      const app = express();
      const server = http.createServer(app);
      const wss = new WebSocketServer({ server });

      // Serve static HTML and frontend code for Cytoscape.js
      app.use(express.static('public'));

      // Handle WebSocket connections to send the graph
      wss.on('connection', function connection(ws) {
        ws.send(JSON.stringify({ graph: cy.json() }));
      });

      server.listen(8080, () => {
        console.log('Visualization server listening on http://localhost:8080');
      });
    }

  } catch (error) {
    console.error('Error:', error);
  }
})();

