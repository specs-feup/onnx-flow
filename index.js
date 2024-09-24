import {createGraph } from './initGraph.js'
import {transformOpps} from './transformOpps.js'
import {generateCode} from './codeGeneration.js'
import {layoutAndStyling } from './layoutAndStyling.js'

Promise.all([
  fetch('AddAddAdd.json')
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
    //transformOpps(cy)
    //generateCode(cy, dataArray[0])
    layoutAndStyling(cy,dataArray[1])
});


//FAZER SLIDES
//colocar otimizações de lado por enquanto
//adicionar suporte para n dimensões e utilizar a ferramenta onnx2json
//gerar os testes -> testes de integração são
//tornar a coisa funcional

// MAIS TARDE LER LITERATURA SOBRE OTIMIZAÇÃO DE OPERAÇÕES