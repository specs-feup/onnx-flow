import {createGraph } from './initGraph.js'
import {transformOpps } from './transformOpps.js'
import {styleCytoscape } from './initGraph.js'

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
    let cy = createGraph(dataArray)
    cy.edges().forEach(edge => {console.log(edge.data())})
    transformOpps(cy)
    styleCytoscape(cy,dataArray[1])
});
