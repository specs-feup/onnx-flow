import {createGraph } from './initGraph.js'
import {transformOpps} from './transformOpps.js'
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
    layoutAndStyling(cy,dataArray[1])
});
