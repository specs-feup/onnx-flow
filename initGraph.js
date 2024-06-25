
/*
Method that takes the onnx graph in json and adds its inputs to the cytoscape graph as input nodes
 */
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

/*
Method that takes the onnx graph in json and adds its outputs to the cytoscape graph as output nodes
 */
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

/*
Method that takes the onnx graph in json and adds its nodes (operations) to the cytoscape graph,
while mapping their id in cytoscape graph to their outputs
 */
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


/*
Method that creates the edges in cytoscape graph that connect operation nodes to outputs and
inputs of the graph and between each other
 */
function addEdges(cy, mapNodeAndOutput) {
    cy.nodes('.operation').forEach(node => {
        node.data('inputs').forEach(input => {
            const inputFound = cy.$('#' + input)

            if (inputFound.data()) {
                const dimensions = inputFound.data('dimensions').map(dim => dim.dimValue).join(',')
                cy.add({
                    group: 'edges',
                    data: {
                        source: input,
                        target: node.data('id'),
                        label: dimensions,
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



/*
Method that initializes the cytoscape graph. Its structure will include
input nodes and output nodes (which are variables) and the operation nodes.
The operation nodes connect to each other if a node's output is the input to another node
 */
function initializeCytoscapeGraph () {
    return cytoscape({
        container: document.getElementById('cy'),
        elements: [],
        style:[],
        layout: []
    })
}

/*
Recursive method that finds the dimensions of each variable in the graph. It starts from the output nodes, checks if
its inputs dimensions are known. If they aren't, the method is called on the inputs of the output nodes, and so on,
until a node in which all inputs have known dimensions is reached. Then it determines the dimensions of the resulting outputs of this node,
considering the operation type and the inputs' dimensions. It repeats this process, until all operation's nodes dimensions are known.
This assumes that all the output and input nodes of the graph have known dimensions beforehand.
(these dimensions are stored in the edges)
 */
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
                    edge.data('label', firstEdgeDims[0].dimValue + ',' + secondEdgeDims[1].dimValue)
                })
            }
            else {
                outgoingEdges.forEach(edge => {
                    edge.data('dims', [{dimValue: firstEdgeDims[1].dimValue}, {dimValue: secondEdgeDims[0].dimValue}])
                    edge.data('elemType', firstEdgeElemType)
                    edge.data('label', firstEdgeDims[0].dimValue + ',' + secondEdgeDims[1].dimValue)
                })
            }
        }
        else {
            const dimensions = firstEdgeDims.map(dim => dim.dimValue).join(',')
            outgoingEdges.forEach(edge => {
                edge.data('dims', firstEdgeDims)
                edge.data('label', dimensions)
                edge.data('elemType', firstEdgeElemType)
            })
        }
    })
}



export function createGraph(data) {
    let cy =  initializeCytoscapeGraph()
    addInputNodes(data, cy)
    addOutputNodes(data, cy)
    let mapNodeAndOutput = []
    addNodes(data, cy, mapNodeAndOutput)
    addEdges(cy, mapNodeAndOutput)
    findDims(cy.nodes('.output'),cy)
    return cy
}


//there is a dependency between the order in which the inputs are introduced in MatMul and its resulting dimensions
//this can affect the final dimensions in the case of dimensions like (3,2) and (2,3)

// this is solved, because the edges are ran in the order that they were added. And they are
// added by the same order that is in the onxx graph json file