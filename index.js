import {createGraph } from './initGraph.js'

Promise.all([
  fetch('TestFail.json')
    .then(function(res) {
      return res.json();
    }),
  fetch('cyStyling.json')
    .then(function(res) {
      return res.json();
    })
])
.then(function(dataArray) {
  let inputs = []
  let outputs = []
  let intermediateVariables = []
  let nodes = []
  let inputEdgesWithKnownDims = []
  let outputEdgesWithKnownDims = []
  let inputEdgesWithUnknownDims = []
  let outputEdgesWithUnknownDims = []
  const cy = createGraph(dataArray, nodes, inputs, outputs, intermediateVariables, 
    inputEdgesWithKnownDims, outputEdgesWithKnownDims, inputEdgesWithUnknownDims, outputEdgesWithUnknownDims)
});


