function addInputNodes(data, cy) {
    data.graph.input.forEach(input => {
        cy.add({
            group: 'nodes',
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

function addOutputNodes(data, cy) {
    data.graph.output.forEach(output => {
        cy.add({
            group: 'nodes',
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

function addNodes(data, cy, mapNodeAndOutput) {
    data.graph.node.forEach((node, index) => {
        cy.add({
            group: 'nodes',
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

function addEdges(cy, mapNodeAndOutput) {
    cy.nodes('.operation').forEach(node => {
        node.data('inputs').forEach(input => {
            const inputFound = cy.$('#' + input)
            if (inputFound.data()) {
                cy.add({
                    group: 'edges',
                    data: {
                        source: input,
                        target: node.data('id'),
                        dims: inputFound.data('dimensions'),
                        elemType: inputFound.data('elemType')
                    }
                })
            }
            else {
                const nodeWithCorrespondingOutput = mapNodeAndOutput.find(elem => elem.output === input)
                if (nodeWithCorrespondingOutput) {
                    cy.add({
                        group: 'edges',
                        data: {
                            source: nodeWithCorrespondingOutput.nodeId,
                            target: node.data('id'),
                            dims: 'None',
                            elemType: 'None'
                        }
                    })
                }
            }

        })
        node.data('outputs').forEach(output => {
            const outputFound = cy.$('#' + output)
            if (outputFound.data()) {
                cy.add({
                    group: 'edges',
                    data: {
                        source: node.data('id'),
                        target: output,
                        dims: 'None',
                        elemType: 'None'
                    }
                })
            }
        })
    })
}



function initializeCytoscapeGraph (nodes, edges, sty) {
    return cytoscape({
        container: document.getElementById('cy'),
        elements: [],
        style:[],
        layout: []
    })
}

function findDims(outputNodes, cy) {
    outputNodes.forEach(node => {
        let incomingEdges = node.incomers('edge')
        incomingEdges.forEach(edge => {
            if (edge.data('dims') === 'None') {
                findDims(cy.$('#' + edge.data('source')), cy)
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

function styleCytoscape(cy, sty){
    cy.style(sty)
    cy.layout({
        name: 'breadthfirst',
        directed: true,
        roots: '.input',
        padding: 10
    }).run()
    return cy
}


export function createGraph(data) {
    let cy =  initializeCytoscapeGraph(data[1])
    addInputNodes(data[0], cy)
    addOutputNodes(data[0], cy)
    let mapNodeAndOutput = []
    addNodes(data[0], cy, mapNodeAndOutput)
    addEdges(cy, mapNodeAndOutput)
    findDims(cy.nodes('.output'),cy)
    styleCytoscape(cy, data[1])
    cy.edges().forEach(edge => {console.log(edge.data())})
    return cy
}
