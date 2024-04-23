function addInput(data, cy) {
    data.graph.input.forEach(input => {
        cy.add({
            data: {
                id: input.name,
                label: input.name,
                elemType: input.type.tensorType.elemType,
                dimensions: input.type.tensorType.shape.dim
            },
            classes: 'input'
        })
    })
}

function addInputNodes(data, nodes) {
    data.graph.input.forEach(input => {
      nodes.push({
        data: {
            id: input.name,
            label: input.name,
            elemType: input.type.tensorType.elemType,
            dimensions: input.type.tensorType.shape.dim
        },
        classes: 'input'
      })
    })
}

function addOutputNodes(data, nodes) {
    data.graph.output.forEach(output => {
      nodes.push({
        data: {
            id: output.name,
            label: output.name,
            elemType: output.type.tensorType.elemType,
            dimensions: output.type.tensorType.shape.dim
        },
        classes: 'output'
      })
    })
}

function addNodes(data, nodes, mapNodeAndOutput) {
    data.graph.node.forEach((node, index) => {
        nodes.push({
            data: {
                id: index.toString(),
                label: node.opType,
                opType: node.opType,
                inputs: node.input,
                outputs: node.output
            },
            classes: 'operation'
        })
        node.output.forEach(output => {
            mapNodeAndOutput.push({nodeId: index.toString(), output: output})
        })
    })
}

function addEdges(nodes, mapNodeAndOutput, edges) {
    nodes.forEach(node => {
        if (node.classes === 'operation') {
            node.data.inputs.forEach(input => {
                const inputFound = nodes.find(inp => inp.data.id === input)
                if (inputFound) {
                    edges.push({
                        data: {
                            source: input,
                            target: node.data.id,
                            dims: inputFound.data.dimensions, 
                            elemType: inputFound.data.elemType
                        }
                    })
                }
                else {
                    const nodeWithCorrespondingOutput = mapNodeAndOutput.find(elem => elem.output === input)
                    if (nodeWithCorrespondingOutput) {
                        edges.push({
                            data: {
                                source: nodeWithCorrespondingOutput.nodeId,
                                target: node.data.id,
                                dims: 'None', 
                                elemType: 'None'
                            }
                        })
                    }
                }
            })
            node.data.outputs.forEach(output => {
                const outputFound = nodes.find(out => out.data.id === output)
                if (outputFound) {
                    edges.push({
                        data: {
                            source: node.data.id,
                            target: output,
                            dims: 'None', 
                            elemType: 'None'
                        }
                    })
                }
            })
        }

    })
}



function initializeCytoscapeGraph (nodes, edges, sty) {
    const cy = cytoscape({
        container: document.getElementById('cy'),
        elements: {
            nodes: nodes,
            edges:edges
        },
        style: sty,
        layout: {
            name: 'breadthfirst',
            directed: true,
            roots: '.input',
            padding: 10
        }
        
    })
    return cy;
}

function betterRecursion(outputNodes, cy) {
    outputNodes.forEach(node => {
        let incomingEdges = node.incomers('edge')
        incomingEdges.forEach(edge => {
            if (edge.data('dims') === 'None') {
                betterRecursion(cy.$('#' + edge.data('source')), cy)

            }
        })

        let outgoingEdges = node.outgoers('edge')
        let firstEdgeDims = incomingEdges[0].data('dims')
        let firstEdgeElemType = incomingEdges[0].data('elemType')

        if (node.data('opType') === 'MatMul') {
            let secondEdgeDims = incomingEdges[1].data('dims')
            if (firstEdgeDims[1].dimValue === secondEdgeDims[0].dimValue) {
                outgoingEdges.forEach(edge => {
                    edge.data('dims', [{dimValue: firstEdgeDims[0].dimValue}, {dimValue: secondEdgeDims[1].dimValue}])
                    edge.data('elemType', firstEdgeElemType)
                })
            }
            else {
                outgoingEdges.forEach(edge => {
                    edge.data('dims', [{dimValue: firstEdgeDims[1].dimValue}, {dimValue: secondEdgeDims[0].dimValue}])
                    edge.data('elemType', firstEdgeElemType)
                })
            }
        }
        else {
            outgoingEdges.forEach(edge => {
                edge.data('dims', firstEdgeDims)
                edge.data('elemType', firstEdgeElemType)
            })
        }
    })
}
/*
function recursion(node, cy) {
    let incomingEdges = node.incomers('edge')
    incomingEdges.forEach(edge => {
        if (edge.data('dims') === 'None') {
            recursion(cy.$('#' + edge.data('source')), cy)
            
        }
    })

    let outgoingEdges = node.outgoers('edge')
    let firstEdgeDims = incomingEdges[0].data('dims')
    let firstEdgeElemType = incomingEdges[0].data('elemType')

    if (node.data('opType') === 'MatMul') {
        let secondEdgeDims = incomingEdges[1].data('dims')
        if (firstEdgeDims[1].dimValue === secondEdgeDims[0].dimValue) {
            outgoingEdges.forEach(edge => {
                edge.data('dims', [{dimValue: firstEdgeDims[0].dimValue}, {dimValue: secondEdgeDims[1].dimValue}])
                edge.data('elemType', firstEdgeElemType)
            })
        }
        else {
            outgoingEdges.forEach(edge => {
                edge.data('dims', [{dimValue: firstEdgeDims[1].dimValue}, {dimValue: secondEdgeDims[0].dimValue}])
                edge.data('elemType', firstEdgeElemType)
            })
        }
    }
    else {
        outgoingEdges.forEach(edge => {
            edge.data('dims', firstEdgeDims)
            edge.data('elemType', firstEdgeElemType)
        })
    }
}*/


export function createGraph(data) {
    let nodes = [];
    addInputNodes(data[0], nodes)
    addOutputNodes(data[0], nodes)
    let edges = []
    let mapNodeAndOutput = []
    addNodes(data[0], nodes, mapNodeAndOutput)
    addEdges(nodes, mapNodeAndOutput, edges)
    let cy =  initializeCytoscapeGraph(nodes, edges, data[1])
    betterRecursion(cy.nodes('.output'),cy)
    cy.edges().forEach(edge => {console.log(edge.data())})

    return cy
} 


//separar addNodes (não misturar coisas diferentes) e talvez usar selectors nos métodos anteriores
//começar dos inputs na bfs 
//ou fazer recursivamente