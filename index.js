import express from 'express';
import http from 'http';
import { WebSocketServer } from 'ws';
import {createGraph} from './initGraph.js'
import {transformOpps, optimizeForDimensions} from './transformOpps.js'
import {generateCode} from './codeGeneration.js'
import {onnx2json} from './onnx2json.js'

const onnxFilePath = process.argv[2];

if (!onnxFilePath) {
    console.error('Please provide a path to the ONNX file.');
    process.exit(1);
}

(async function main() {
  try {
      // Step 1: Load and process the ONNX graph
      const onnxObject = await onnx2json(onnxFilePath);
      console.log("Input ONNX Graph: " + JSON.stringify(onnxObject, null, 2));

      let cy = createGraph(onnxObject);
      console.log("Generated Cytoscape Graph: " + JSON.stringify(cy.json(), null, 2));

      transformOpps(cy);
      optimizeForDimensions(cy);

      console.log("Result Code: " + generateCode(cy, onnxObject));
      console.log("Result Cytoscape Graph: " + JSON.stringify(cy.json(), null, 2));

      // Step 2: Setup the server after the processing is complete
      const app = express();
      const server = http.createServer(app);
      const wss = new WebSocketServer({ server });

      // Serve the HTML and frontend code for Cytoscape.js
      app.use(express.static('public'));

      // Step 3: Handle WebSocket connections to send the graph
      wss.on('connection', function connection(ws) {
          ws.send(JSON.stringify({ graph: cy.json() }));
      });

      server.listen(8080, () => {
          console.log('Server listening on http://localhost:8080');
      });
  } catch (error) {
      console.error('Error:', error);
  }
})();

//select the operation nodes with no parent and check if there are optimizations to be made

//FAZER SLIDES
//colocar otimizações de lado por enquanto
//adicionar suporte para n dimensões e utilizar a ferramenta onnx2json
//gerar os testes -> testes de integração são
//tornar a coisa funcional, ou seja, dividir em passos

// MAIS TARDE LER LITERATURA SOBRE OTIMIZAÇÃO DE OPERAÇÕES
