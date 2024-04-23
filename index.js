import {createGraph } from './initGraph.js'
//import {transformOpps } from './transformOpps.js'

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
    
    //transformOpps(cy)
});
