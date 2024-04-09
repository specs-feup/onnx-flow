document.addEventListener('DOMContentLoaded', function(){
  // Load the ONNX graph JSON
  fetch('MultiplyAndAdd.json')
    .then(response => response.json())
    .then(data => {
      const inputs = [];
      const outputs = [];
      intermediateValues = [];
      const nodes = [];
      const edges = [];


      // Add inputs from data
      data.graph.input.forEach(input => {
        inputs.push({
          data: {
              id: input.name,
              label: input.name,
              elemType: input.type.tensorType.elemType,
              dimensions: input.type.tensorType.shape.dim
          },
          classes: 'input'
        });
      });

      // Add outputs from data
      data.graph.output.forEach(output => {
        outputs.push({
          data: {
              id: output.name,
              label: output.name,
              elemType: output.type.tensorType.elemType,
              dimensions: output.type.tensorType.shape.dim
          },
          classes: 'output'
        });
      });


      data.graph.node.forEach((node, index) => {


        node.input.forEach(input => {
          const inputFound = inputs.find(inp => inp.data.id === input);
          if (inputFound) {
            edges.push({
              data: {
                source: input,
                target: index.toString()
              },
              elemType: inputFound.elemType,
              dimensions: inputFound.data.dimensions
            });
          }
          else {
            intermediateValues.push({
              data: {
                id: input,
                label: input,
                elemType: 'None',
                dimensions: 'None'
              }
            });
            edges.push({
              data: {
                source: input,
                target: index.toString()
              },
              elemType: 'None',
              dimensions: 'None'
            });
          }
        });
        
        node.output.forEach( output => {
          const outputFound = outputs.find(out => out.data.id === output);
          if (outputFound) {
            edges.push({
              data: {
                source: index.toString(),
                target: output
              },
              elemType: outputFound.elemType,
              dimensions: outputFound.data.dimensions
            })
          }
          else {
            edges.push({
              data: {
                source: index.toString(),
                target: output
              },
              elemType: 'None',
              dimensions: 'None'
            })
          }
        });

        nodes.push({
          data: {
            id: index.toString(),
            label: node.opType
          },
          classes: 'operation',
        });
      });



     
      
      // Initialize Cytoscape
      const cy = cytoscape({
        container: document.getElementById('cy'),
        elements: {
          nodes: nodes.concat(inputs, outputs, intermediateValues),
          edges: edges
        },
        style: [
          {
            selector: 'node',
            style: {
              'background-color': '#666',
              'label': 'data(label)'
            }
          },
          {
            selector: '.input',
            style: {
              'background-color': '#8AFFAB' // Green
            }
          },
          {
            selector: '.output',
            style: {
              'background-color': '#FF8A8A' // Red
            }
          },
          {
            selector: '.operation',
            style: {
              'background-color': '#8ACBFF' // Blue
            }
          }, 
          {
            selector: 'edge',
            style: {
              'curve-style': 'bezier',
              'target-arrow-shape': 'triangle',
              'target-arrow-color': '#ccc',
              'line-color': '#ccc',
              'width': 2
            }
          }
        ],
        layout: {
          name: 'grid',
          rows: 1,
          position: function(node) {
            // Adjust the position of nodes based on their type
            if (node.hasClass('input')) {
              return { row: 0, col: node.data('index') };
            } else if (node.hasClass('output')) {
              return { row: 2, col: node.data('index') };
            } else {
              return { row: 1, col: node.data('index') };
            }
          },
          avoidOverlap: true,
        }
        
      });
    });
});
