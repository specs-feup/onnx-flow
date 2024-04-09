import {getAllDimensions } from './handleDims.js'

function addInputNodes(inputs, data) {
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
  };

function addOutputNodes(outputs, data) {
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
};

function addEdgesToInputs (index, opType, nodeInputs, inputEdgesWithKnownDims, inputEdgesWithUnknownDims, inputs, intermediateVariables) {
//if the input is one of the graph inputs array, create an edge and store input info in the edge
    nodeInputs.forEach(input => {
        const inputFound = inputs.find(inp => inp.data.id === input);
        if (inputFound) {
        inputEdgesWithKnownDims.push({
            data: {
            source: input,
            target: index.toString(),
            //label: inputFound.data.dimensions[0].dimValue
            },
            elemType: inputFound.data.elemType,
            dimensions: inputFound.data.dimensions,
            opType: opType
        });
        }

        //else add a node to the intermediateVariables array and create and edge without info about the input
        else {
        intermediateVariables.push({
            data: {
            id: input,
            label: input,
            elemType: 'None',
            dimensions: [
                {
                  dimValue: -1
                },
                {
                  dimValue: -1
                }
            ],
            }
        });
        inputEdgesWithUnknownDims.push({
            data: {
            source: input,
            target: index.toString(),
            //label: 'undefined'
            },
            elemType: 'None',
            dimensions: [
                {
                  dimValue: -1
                },
                {
                  dimValue: -1
                }
            ],
            opType: opType
        });
        }
    });
}



function addEdgesToOutputs (index, opType, nodeOutputs, outputEdgesWithKnownDims, outputEdgesWithUnknownDims, outputs) {
    nodeOutputs.forEach( output => {
        const outputFound = outputs.find(out => out.data.id === output);
    //if the output is one of the graph outputs array, create an edge and store output info in the edge
        if (outputFound) {
            outputEdgesWithKnownDims.push({
                data: {
                source: index.toString(),
                target: output,
                //label: outputFound.data.dimensions[0].dimValue
                },
                elemType: outputFound.data.elemType,
                dimensions: outputFound.data.dimensions,
                opType: opType
            })
        }

        //else create and edge without info about the output
        else {
            outputEdgesWithUnknownDims.push({
                data: {
                source: index.toString(),
                target: output,
                //label: 'undefined'
                },
                elemType: 'None',
                dimensions: [
                    {
                      dimValue: -1
                    },
                    {
                      dimValue: -1
                    }
                ],
                opType: opType
            })
        }

    });
}

function addOppNodes(node, index, nodes) {
    nodes.push({
        data: {
        id: index.toString(),
        label: node.opType
        },
        classes: 'operation',
    });
}

function addOpperationNodesAndEdges (data, nodes, inputs, outputs, inputEdgesWithKnownDims, outputEdgesWithKnownDims, inputEdgesWithUnknownDims, outputEdgesWithUnknownDims, intermediateVariables) {
    data.graph.node.forEach((node, index) => {
        addEdgesToInputs (index, node.opType, node.input, inputEdgesWithKnownDims, inputEdgesWithUnknownDims, inputs, intermediateVariables);
        addEdgesToOutputs (index, node.opType, node.output, outputEdgesWithKnownDims, outputEdgesWithUnknownDims, outputs);
        addOppNodes(node, index, nodes);
    });
}


function initializeCytoscapeGraph(nodes, inputs, outputs, intermediateVariables, edges, sty) {
    const cy = cytoscape({
        container: document.getElementById('cy'),
        elements: {
        nodes: nodes.concat(inputs, outputs, intermediateVariables),
        edges: edges
        },
        style: sty,
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
    return cy;
}


export function createGraph(data, nodes, inputs, outputs, intermediateVariables, inputEdgesWithKnownDims, outputEdgesWithKnownDims, inputEdgesWithUnknownDims, outputEdgesWithUnknownDims) {
    addInputNodes(inputs, data[0])
    addOutputNodes(outputs, data[0])
    addOpperationNodesAndEdges(data[0], nodes, inputs, outputs, inputEdgesWithKnownDims, outputEdgesWithKnownDims, inputEdgesWithUnknownDims, outputEdgesWithUnknownDims, intermediateVariables)
    let edges = getAllDimensions(inputEdgesWithKnownDims, outputEdgesWithKnownDims, inputEdgesWithUnknownDims, outputEdgesWithUnknownDims)
    
    //IMPORTANTE!!!
    //1.Pensar se vale a pena, cada vez que vou à procura de intermediate variables como inputs, 
    //poderia também determinar o edge que as tem como outputs -> DONE
    //Acho que ainda haveria a possibilidade de matmul fazer index++

    //2.adicionar dims aos intermediateNodes ou nada feito -> TO BE DONE

    //3.Dividir o ficheiro em getDimensions e o rest -> DONE

    //4.Começar com add para development -> TO BE DONE

    //mudar as default not working dimensions para outra coisa -> TO BE DONE 
    //Perguntar se deveria adicionar security, isto é, verificar se a informação dada é válida, 
    //se as dimensoes num MatMul estão certas?
    //Dimensões para vetores são [5][1] ou só [5]
    //Devo adicionar quais opps?

    //dar update às dimensões das intermediate variables

    return initializeCytoscapeGraph(nodes, inputs, outputs, intermediateVariables, edges, data[1]);
}
    