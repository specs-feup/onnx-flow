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
    const cy = createGraph(dataArray)
});
