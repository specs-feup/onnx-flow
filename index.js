import {createGraph} from './initGraph.js'
import {transformOpps, optimizeForDimensions} from './transformOpps.js'
import {generateCode} from './codeGeneration.js'
import {onnx2json} from './onnx2json.js'

const onnxFilePath = process.argv[2];

if (!onnxFilePath) {
    console.error('Please provide a path to the ONNX file.');
    process.exit(1);
}

const onnxObject = await onnx2json(onnxFilePath);
console.log("Input ONNX Graph: " + JSON.stringify(onnxObject, null, 2));

Promise.all([
  onnxObject
])
.then(function(dataArray) {
    let cy = createGraph(dataArray[0]);
    console.log("Generated Cytoscape Graph: " + JSON.stringify(cy.json(), null, 2));
    transformOpps(cy);
    optimizeForDimensions(cy);
    console.log("Result Code: " + generateCode(cy, dataArray[0]));
    console.log("Result Cytoscape Graph: " + JSON.stringify(cy.json(), null, 2));
});

//select the operation nodes with no parent and check if there are optimizations to be made

//FAZER SLIDES
//colocar otimizações de lado por enquanto
//adicionar suporte para n dimensões e utilizar a ferramenta onnx2json
//gerar os testes -> testes de integração são
//tornar a coisa funcional, ou seja, dividir em passos

// MAIS TARDE LER LITERATURA SOBRE OTIMIZAÇÃO DE OPERAÇÕES
