import {createGraph } from './initGraph.js'
import {transformOpps, optimizeForDimensions} from './transformOpps.js'
import {generateCode} from './codeGeneration.js'
import {layoutAndStyling } from './layoutAndStyling.js'

Promise.all([
  fetch('MatMul.json')
    .then(function(res) {
      return res.json();
    }),
  fetch('cyStyling.json')
    .then(function(res) {
      return res.json();
    })
])
.then(function(dataArray) {
    let cy = createGraph(dataArray[0])
    transformOpps(cy)
    optimizeForDimensions(cy)
    //generateCode(cy, dataArray[0])
    layoutAndStyling(cy,dataArray[1])
});

//select the operation nodes with no parent and check if there are optimizations to be made

//FAZER SLIDES
//colocar otimizações de lado por enquanto
//adicionar suporte para n dimensões e utilizar a ferramenta onnx2json
//gerar os testes -> testes de integração são
//tornar a coisa funcional, ou seja, dividir em passos

// MAIS TARDE LER LITERATURA SOBRE OTIMIZAÇÃO DE OPERAÇÕES