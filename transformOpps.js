
const typeSizeMap = {
    "0": "0",                              // onnx.TensorProto.UNDEFINED
    "1": "4",                              // onnx.TensorProto.FLOAT
    "2": "1",                              // onnx.TensorProto.UINT8
    "3": "1",                              // onnx.TensorProto.INT8
    "4": "2",                              // onnx.TensorProto.UINT16
    "5": "2",                              // onnx.TensorProto.INT16
    "6": "4",                              // onnx.TensorProto.INT32
    "7": "8",                              // onnx.TensorProto.INT64
    "8": "-1",                             // onnx.TensorProto.STRING (Variable size)
    "9": "1",                              // onnx.TensorProto.BOOL
    "10": "2",                             // onnx.TensorProto.FLOAT16
    "11": "8",                             // onnx.TensorProto.DOUBLE
    "12": "4",                             // onnx.TensorProto.UINT32
    "13": "8",                             // onnx.TensorProto.UINT64
    "14": "8",                             // onnx.TensorProto.COMPLEX64
    "15": "16",                            // onnx.TensorProto.COMPLEX128
    "16": "2",                             // onnx.TensorProto.BFLOAT16
    "17": "1",                             // onnx.TensorProto.FLOAT8E4M3FN
    "18": "1",                             // onnx.TensorProto.FLOAT8E4M3FNUZ
    "19": "2",                             // onnx.TensorProto.FLOAT8E5M2
    "20": "2",                             // onnx.TensorProto.FLOAT8E5M2FNUZ
    "21": "1",                             // onnx.TensorProto.UINT4
    "22": "1"                              // onnx.TensorProto.INT4
}


//Method that expands the Add operation node into simple arithmetic nodes

function transformAdd(node, cy) {
    const incomingEdges = node.incomers('edge')
    const outgoingEdges = node.outgoers('edge')
    const dimensions = incomingEdges[0].data('dims')
    const type = incomingEdges[0].data('elemType')

    // Optimization the case that the dimensions of the Add's inputs have 1 dimension, with value 1.
    // (Just like adding 2 integers, for example)
    if (dimensions[0].dimValue === '1' && dimensions.length === 1) {
        cy.add([
            {group: 'nodes', data: {id: node.data('id') + 'Addition', label: '+', opType: 'Addition'}, classes: 'operation'},
            {group: 'edges', data: {source: incomingEdges[0].data('source'), target: node.data('id') + 'Addition', dims: incomingEdges[0].data('dims'), elemType: incomingEdges[0].data('elemType')}},
            {group: 'edges', data: {source: incomingEdges[1].data('source'), target: node.data('id') + 'Addition', dims: incomingEdges[1].data('dims'), elemType: incomingEdges[1].data('elemType')}}
        ])
        outgoingEdges.forEach(edge => {
            cy.add({group: 'edges', data: {source: node.data('id') + 'Addition', target: edge.data('target'), dims: edge.data('dims'), elemType: edge.data('elemType')}})
        })
    }
    // The other case is a loop where the inputs of the onnx graph are iterated over and their values summed.
    // (Just like adding the values of a vector or matrix )
    else {
        let numberOfIterations = dimensions.reduce((total, dim) => total + parseInt(dim.dimValue), 0)
        let order = 0
        const nodeId = node.data('id')
        let displacementInMemory = typeSizeMap[type]
        cy.add([
            {group: 'nodes', data: {id: nodeId + 'LoopIterations', label: '# of loop iterations', value: numberOfIterations}, classes: 'constant'},
            {group: 'nodes', data: {id: nodeId + 'Add', label: 'Add', opType: 'Add'}},
            {group: 'nodes', data: {id: nodeId + 'index', parent: nodeId + 'Add', label: 'index'}, classes: 'input'},
            {group: 'nodes', data: {id: nodeId + 'displacementInMemory', parent: nodeId + 'Add', label: 'displacement In Memory', value: displacementInMemory}, classes: 'input'},
            {group: 'nodes', data: {id: nodeId + incomingEdges[0].data('source'), label: '&' + incomingEdges[0].data('source'), parent: nodeId + 'Add'}, classes: 'input'},
            {group: 'nodes', data: {id: nodeId + incomingEdges[1].data('source'), label: '&' + incomingEdges[1].data('source'), parent: nodeId + 'Add'}, classes: 'input'},
            {group: 'nodes', data: {id: nodeId + 'res', label: '&Result', parent: nodeId + 'Add'}, classes: 'output'},
            {group: 'nodes', data: {id: nodeId + 'Multiplication', parent: nodeId + 'Add', label: '*', opType: 'Multiplication'}, classes: 'operation'},
            {group: 'nodes', data: {id: nodeId + 'Load0', parent: nodeId + 'Add', label: 'Load', opType: 'Load'}, classes: 'operation'},
            {group: 'nodes', data: {id: nodeId + 'Load1', parent: nodeId + 'Add', label: 'Load', opType: 'Load'}, classes: 'operation'},
            {group: 'nodes', data: {id: nodeId + 'Addition', parent: nodeId + 'Add', label: '+', opType: 'Addition'}, classes: 'operation'},
            {group: 'nodes', data: {id: nodeId + 'Addition1', parent: nodeId + 'Add', label: '+', opType: 'Addition'}, classes: 'operation'},
            {group: 'nodes', data: {id: nodeId + 'Store', parent: nodeId + 'Add', label: 'Store', opType: 'Store'}, classes: 'operation'},
            {group: 'nodes', data: {id: nodeId + '1', parent: nodeId + 'Add', label: '1'}, classes: 'constant'},

            //edges for the loop inputs
            {group: 'edges', data: {source: incomingEdges[0].data('source'), label: incomingEdges.data('label'), target: nodeId + 'Add', dims: incomingEdges[0].data('dims'), elemType: incomingEdges[0].data('elemType')}},
            {group: 'edges', data: {source: incomingEdges[1].data('source'), label: incomingEdges.data('label'), target: nodeId + 'Add', dims: incomingEdges[1].data('dims'), elemType: incomingEdges[1].data('elemType')}},
            {group: 'edges', data: {source: nodeId + 'LoopIterations', target: nodeId + 'Add'}},

            //edges for the loop environment
            {group: 'edges', data: {source: nodeId + 'index', target: nodeId + 'Multiplication', parent: nodeId + 'Add', order: order++}, classes: 'index'},
            {group: 'edges', data: {source: nodeId + 'displacementInMemory', target: nodeId + 'Multiplication', parent: nodeId + 'Add', order: order++, value: displacementInMemory }, classes: 'constant'},
            {group: 'edges', data: {source: nodeId + incomingEdges[0].data('source'), target: nodeId + 'Load0', parent: nodeId + 'Add', order: order++}, classes: 'input'},
            {group: 'edges', data: {source: nodeId + 'Multiplication', target: nodeId + 'Load0', parent: nodeId + 'Add', order: order++, opType: 'Multiplication'}, classes: 'operation'},
            {group: 'edges', data: {source: nodeId + incomingEdges[1].data('source'), target: nodeId + 'Load1', parent: nodeId + 'Add', order: order++}, classes: 'input'},
            {group: 'edges', data: {source: nodeId + 'Multiplication', target: nodeId + 'Load1', parent: nodeId + 'Add', order: order++}, classes: 'operation'},
            {group: 'edges', data: {source: nodeId + 'Load0', target: nodeId + 'Addition', parent: nodeId + 'Add', order: order++, opType: 'Load'}, classes: 'operation'},
            {group: 'edges', data: {source: nodeId + 'Load1', target: nodeId + 'Addition', parent: nodeId + 'Add', order: order++, opType: 'Load'}, classes: 'operation'},
            {group: 'edges', data: {source: nodeId + 'Addition', target: nodeId + 'Store', parent: nodeId + 'Add', order: order++, opType: 'Addition'}, classes: 'operation'},
            {group: 'edges', data: {source: nodeId + 'Multiplication', target: nodeId + 'Store', parent: nodeId + 'Add', order: order++, opType: 'Multiplication'}, classes: 'operation'},
            {group: 'edges', data: {source: nodeId + 'Store', target: nodeId + 'res', parent: nodeId + 'Add', order: order++, opType: 'Store'}, classes: 'operation'},

            {group: 'edges', data: {source: nodeId + 'index', target: nodeId + 'Addition1', parent: nodeId + 'Add', order: order++}, classes: 'index'},
            {group: 'edges', data: {source: nodeId + '1', target: nodeId + 'Addition1', parent: nodeId + 'Add', order: order++, value: 1}, classes: 'constant'},
            {group: 'edges', data: {source: nodeId + 'Addition1', target: nodeId + 'index', parent: nodeId + 'Add', order: order++, opType: 'Addition'}, classes: 'operation variable'},

        ])
        //edges for the loop outputs
        outgoingEdges.forEach(edge => {
            cy.add({group: 'edges', data: {source: nodeId + 'Add', target: edge.data('target'), dims: edge.data('dims'), elemType: edge.data('elemType')}})
        })
    }
    cy.remove(node)
}

function transformMatMul(node, cy) {
    const incomingEdges = node.incomers('edge')
    const outgoingEdges = node.outgoers('edge')
    const dimensions0 = incomingEdges[0].data('dims')
    const dimensions1 = incomingEdges[1].data('dims')
    const type = incomingEdges[0].data('elemType')
    let numberOfIterations = dimensions0[0].dimValue * dimensions1[0].dimValue * dimensions1[1].dimValue

    const nodeId = node.data('id')

    const instanceVal = dimensions0[0].dimValue + dimensions1[0].dimValue + dimensions1[1].dimValue
    let pattern = instanceVal[0] === '1' ? '1' : '0'
    pattern += instanceVal[1] === '1' ? '1' : '0'
    pattern += instanceVal[2] === '1' ? '1' : '0';

    if (pattern === '111') {
        cy.add([
            {group: 'nodes', data: {id: node.data('id') + ' Multiplication', label: '*', opType: 'Multiplication'}, classes: 'operation'},
            {group: 'edges', data: {source: incomingEdges[0].data('source'), target: node.data('id') + ' Multiplication', dims: incomingEdges[0].data('dims'), elemType: incomingEdges[0].data('elemType')}},
            {group: 'edges', data: {source: incomingEdges[1].data('source'), target: node.data('id') + ' Multiplication', dims: incomingEdges[1].data('dims'), elemType: incomingEdges[1].data('elemType')}}
        ])
        outgoingEdges.forEach(edge => {
            cy.add({group: 'edges', data: {source: node.data('id') + ' Multiplication', target: edge.data('target'), dims: edge.data('dims'), elemType: edge.data('elemType')}})
        })
        cy.remove(node)
        return
    }

    cy.add([
        {group: 'nodes', data: {id: nodeId + 'LoopIterations', label: '# of loop iterations', value: numberOfIterations.toString()}, classes: 'constant'},
        {group: 'nodes', data: {id: nodeId + 'MatMul', label: 'MatMul', opType: 'MatMul'}},

        {group: 'edges', data: {source: incomingEdges[0].data('source'), target: nodeId + 'MatMul', dims: incomingEdges[0].data('dims'), elemType: incomingEdges[0].data('elemType')}},
        {group: 'edges', data: {source: incomingEdges[1].data('source'), target: nodeId + 'MatMul', dims: incomingEdges[0].data('dims'), elemType: incomingEdges[0].data('elemType')}},
        {group: 'edges', data: {source: nodeId + 'LoopIterations', target: nodeId + 'MatMul'}},
    ])

    switch (pattern) {
        case '000':
            cy.add([
                {group: 'nodes', data: {id: nodeId + 'i', parent: nodeId +'MatMul', label: 'i'}, classes: 'input'},
                {group: 'nodes', data: {id: nodeId + 'j', parent: nodeId +'MatMul', label: 'j'}, classes: 'input'},
                {group: 'nodes', data: {id: nodeId + 'k', parent: nodeId +'MatMul', label: 'k'}, classes: 'input'},
                {group: 'nodes', data: {id: nodeId + '#columns1', parent: nodeId +'MatMul', label: '# of columns of ' + incomingEdges[1].data('source'), value: dimensions1[1].dimValue}, classes: 'input'},
                {group: 'nodes', data: {id: nodeId + '#rows1', parent: nodeId +'MatMul', label: '# of rows of ' + incomingEdges[1].data('source'), value: dimensions1[0].dimValue}, classes: 'input'},
                {group: 'nodes', data: {id: nodeId + 'displacementInMemory', parent: nodeId + 'MatMul', label: 'displacement In Memory', value: typeSizeMap[type]}, classes: 'input'},
                {group: 'nodes', data: {id: nodeId + 'Multiplication0', parent: nodeId + 'MatMul', label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: nodeId + 'Multiplication1', parent: nodeId + 'MatMul', label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: nodeId + 'Index0', parent: nodeId + 'MatMul', label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: nodeId + 'Index1', parent: nodeId + 'MatMul', label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: nodeId + 'Addition0', parent: nodeId + 'MatMul', label: '+', opType: 'Addition'}, classes: 'operation'},
                {group: 'nodes', data: {id: nodeId + 'Addition1', parent: nodeId + 'MatMul', label: '+', opType: 'Addition'}, classes: 'operation'},

                //Multiplication0 = i * number of rows of matrix B
                {group: 'edges', data: {source: nodeId + 'i', target: nodeId + 'Multiplication0'}},
                {group: 'edges', data: {source: nodeId + '#rows1', target: nodeId + 'Multiplication0'}},

                //Multiplication1 = k * number of columns of matrix B
                {group: 'edges', data: {source: nodeId + 'k', target: nodeId + 'Multiplication1'}},
                {group: 'edges', data: {source: nodeId + '#columns1', target: nodeId + 'Multiplication1'}},

                //Addition0 = Multiplication0 + k
                {group: 'edges', data: {source: nodeId + 'Multiplication0', target: nodeId + 'Addition0'}},
                {group: 'edges', data: {source: nodeId + 'k', target: nodeId + 'Addition0'}},

                //Addition1 = Multiplication1 + j
                {group: 'edges', data: {source: nodeId + 'Multiplication1', target: nodeId + 'Addition1'}},
                {group: 'edges', data: {source: nodeId + 'j', target: nodeId + 'Addition1'}},

                //Index0 (the memory position for matrix A) = Addition0 * the displacement in memory
                {group: 'edges', data: {source: nodeId + 'Addition0', target: nodeId + 'Index0'}},
                {group: 'edges', data: {source: nodeId + 'displacementInMemory', target: nodeId + 'Index0'}},

                //Index1 (the memory position for matrix B) = Addition1 * the displacement in memory
                {group: 'edges', data: {source: nodeId + 'Addition1', target: nodeId + 'Index1'}},
                {group: 'edges', data: {source: nodeId + 'displacementInMemory', target: nodeId + 'Index1'}},

                //store's index
                {group: 'nodes', data: {id: nodeId + 'Multiplication5', parent: nodeId + 'MatMul', label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: nodeId + 'IndexRes', parent: nodeId + 'MatMul', label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: nodeId + 'Addition3', parent: nodeId + 'MatMul', label: '+', opType: 'Addition'}, classes: 'operation'},

                //Multiplication5 = i * number of columns of matrix B
                {group: 'edges', data: {source: nodeId + 'i', target: nodeId + 'Multiplication5'}},
                {group: 'edges', data: {source: nodeId + '#columns1', target: nodeId + 'Multiplication5'}},

                //Addition3 = j * Multiplication 5
                {group: 'edges', data: {source: nodeId + 'j', target: nodeId + 'Addition3'}},
                {group: 'edges', data: {source: nodeId + 'Multiplication5', target: nodeId + 'Addition3'}},

                //IndexRes (the memory position for the result matrix) = Addition3 * the displacement in memory
                {group: 'edges', data: {source: nodeId + 'Addition3', target: nodeId + 'IndexRes'}},
                {group: 'edges', data: {source: nodeId + 'displacementInMemory', target: nodeId + 'IndexRes'}}

            ])
            break
        case '100':
            cy.add([
                {group: 'nodes', data: {id: nodeId + 'j', parent: nodeId +'MatMul', label: 'j'}, classes: 'input'},
                {group: 'nodes', data: {id: nodeId + 'k', parent: nodeId +'MatMul', label: 'k'}, classes: 'input'},
                {group: 'nodes', data: {id: nodeId + '#rows1', parent: nodeId +'MatMul', label: '# of rows of ' + incomingEdges[1].data('source'), value: dimensions1[0].dimValue}, classes: 'input'},
                {group: 'nodes', data: {id: nodeId + '#columns1', parent: nodeId +'MatMul', label: '# of columns of ' + incomingEdges[1].data('source'), value: dimensions1[1].dimValue}, classes: 'input'},
                {group: 'nodes', data: {id: nodeId + 'displacementInMemory', parent: nodeId + 'MatMul', label: 'displacement In Memory', value: typeSizeMap[type]}, classes: 'input'},
                {group: 'nodes', data: {id: nodeId + 'Multiplication1', parent: nodeId + 'MatMul', label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: nodeId + 'Index0', parent: nodeId + 'MatMul', label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: nodeId + 'Index1', parent: nodeId + 'MatMul', label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: nodeId + 'Addition1', parent: nodeId + 'MatMul', label: '+', opType: 'Addition'}, classes: 'operation'},


                {group: 'edges', data: {source: nodeId + 'k', target: nodeId + 'Multiplication1'}},
                {group: 'edges', data: {source: nodeId + '#columns1', target: nodeId + 'Multiplication1'}},

                {group: 'edges', data: {source: nodeId + 'k', target: nodeId + 'Index0'}},
                {group: 'edges', data: {source: nodeId + 'displacementInMemory', target: nodeId + 'Index0'}},

                {group: 'edges', data: {source: nodeId + 'j', target: nodeId + 'Addition1'}},
                {group: 'edges', data: {source: nodeId + 'Multiplication1', target: nodeId + 'Addition1'}},

                {group: 'edges', data: {source: nodeId + 'Addition1', target: nodeId + 'Index1'}},
                {group: 'edges', data: {source: nodeId + 'displacementInMemory', target: nodeId + 'Index1'}},

                //store's index

                {group: 'nodes', data: {id: nodeId + 'IndexRes', parent: nodeId + 'MatMul', label: '*', opType: 'Multiplication'}, classes: 'operation'},

                {group: 'edges', data: {source: nodeId + 'j', target: nodeId + 'IndexRes'}},
                {group: 'edges', data: {source: nodeId + 'displacementInMemory', target: nodeId + 'IndexRes'}}

            ])
            break
        case '001':
            cy.add([
                {group: 'nodes', data: {id: nodeId + 'j', parent: nodeId +'MatMul', label: 'j'}, classes: 'input'},
                {group: 'nodes', data: {id: nodeId + 'k', parent: nodeId +'MatMul', label: 'k'}, classes: 'input'},
                {group: 'nodes', data: {id: nodeId + '#columns1', parent: nodeId +'MatMul', label: '# of columns of ' + incomingEdges[1].data('source'), value: dimensions1[1].dimValue}, classes: 'input'},
                {group: 'nodes', data: {id: nodeId + '#rows1', parent: nodeId +'MatMul', label: '# of rows of ' + incomingEdges[1].data('source'), value: dimensions1[0].dimValue}, classes: 'input'},
                {group: 'nodes', data: {id: nodeId + 'displacementInMemory', parent: nodeId + 'MatMul', label: 'displacement In Memory', value: typeSizeMap[type]}, classes: 'input'},
                {group: 'nodes', data: {id: nodeId + 'Multiplication0', parent: nodeId + 'MatMul', label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: nodeId + 'Index0', parent: nodeId + 'MatMul', label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: nodeId + 'Index1', parent: nodeId + 'MatMul', label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: nodeId + 'Addition0', parent: nodeId + 'MatMul', label: '+', opType: 'Addition'}, classes: 'operation'},

                {group: 'edges', data: {source: nodeId + 'j', target: nodeId + 'Multiplication0'}},
                {group: 'edges', data: {source: nodeId + '#rows1', target: nodeId + 'Multiplication0'}},

                {group: 'edges', data: {source: nodeId + 'Multiplication0', target: nodeId + 'Addition0'}},
                {group: 'edges', data: {source: nodeId + 'k', target: nodeId + 'Addition0'}},

                {group: 'edges', data: {source: nodeId + 'k', target: nodeId + 'Index1'}},

                {group: 'edges', data: {source: nodeId + 'Addition0', target: nodeId + 'Index0'}},
                {group: 'edges', data: {source: nodeId + 'displacementInMemory', target: nodeId + 'Index0'}},

                {group: 'edges', data: {source: nodeId + 'displacementInMemory', target: nodeId + 'Index1'}},

                //store's index

                {group: 'nodes', data: {id: nodeId + 'IndexRes', parent: nodeId + 'MatMul', label: '*', opType: 'Multiplication'}, classes: 'operation'},

                {group: 'edges', data: {source: nodeId + 'j', target: nodeId + 'IndexRes'}},
                {group: 'edges', data: {source: nodeId + 'displacementInMemory', target: nodeId + 'IndexRes'}}

            ])
            break
        case '101':
            cy.add([
                {group: 'nodes', data: {id: nodeId + 'k', parent: nodeId +'MatMul', label: 'k'}, classes: 'input'},
                {group: 'nodes', data: {id: nodeId + 'displacementInMemory', parent: nodeId + 'MatMul', label: 'displacement In Memory', value: typeSizeMap[type]}, classes: 'input'},
                {group: 'nodes', data: {id: nodeId + 'Index0', parent: nodeId + 'MatMul', label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: nodeId + 'Index1', parent: nodeId + 'MatMul', label: '*', opType: 'Multiplication'}, classes: 'operation'},

                {group: 'edges', data: {source: nodeId + 'k', target: nodeId + 'Index0'}},
                {group: 'edges', data: {source: nodeId + 'k', target: nodeId + 'Index1'}},
                {group: 'edges', data: {source: nodeId + 'displacementInMemory', target: nodeId + 'Index0'}},
                {group: 'edges', data: {source: nodeId + 'displacementInMemory', target: nodeId + 'Index1'}},
                //store's index

                {group: 'nodes', data: {id: nodeId + 'IndexRes', parent: nodeId + 'MatMul', label: '*', opType: 'Multiplication'}, classes: 'operation'},
                //aqui é necessário fazer uma ligação direta ao store
                {group: 'edges', data: {source: nodeId + 'displacementInMemory', target: nodeId + 'IndexRes'}}

            ])
            break
        case '010':
            cy.add([
                {group: 'nodes', data: {id: nodeId + 'j', parent: nodeId +'MatMul', label: 'j'}, classes: 'input'},
                {group: 'nodes', data: {id: nodeId + 'k', parent: nodeId +'MatMul', label: 'k'}, classes: 'input'},
                {group: 'nodes', data: {id: nodeId + '#columns1', parent: nodeId +'MatMul', label: '# of columns of ' + incomingEdges[1].data('source'), value: dimensions1[1].dimValue}, classes: 'input'},
                {group: 'nodes', data: {id: nodeId + '#rows1', parent: nodeId +'MatMul', label: '# of rows of ' + incomingEdges[1].data('source'), value: dimensions1[0].dimValue}, classes: 'input'},
                {group: 'nodes', data: {id: nodeId + 'displacementInMemory', parent: nodeId + 'MatMul', label: 'displacement In Memory', value: typeSizeMap[type]}, classes: 'input'},
                {group: 'nodes', data: {id: nodeId + 'Index0', parent: nodeId + 'MatMul', label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: nodeId + 'Index1', parent: nodeId + 'MatMul', label: '*', opType: 'Multiplication'}, classes: 'operation'},

                {group: 'edges', data: {source: nodeId + 'j', target: nodeId + 'Index0'}},
                {group: 'edges', data: {source: nodeId + 'displacementInMemory', target: nodeId + 'Index0'}},

                {group: 'edges', data: {source: nodeId + 'k', target: nodeId + 'Index1'}},
                {group: 'edges', data: {source: nodeId + 'displacementInMemory', target: nodeId + 'Index1'}},

                //store's index

                {group: 'nodes', data: {id: nodeId + 'Multiplication5', parent: nodeId + 'MatMul', label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: nodeId + 'IndexRes', parent: nodeId + 'MatMul', label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: nodeId + 'Addition3', parent: nodeId + 'MatMul', label: '+', opType: 'Addition'}, classes: 'operation'},

                {group: 'edges', data: {source: nodeId + 'j', target: nodeId + 'Multiplication5'}},
                {group: 'edges', data: {source: nodeId + '#columns1', target: nodeId + 'Multiplication5'}},

                {group: 'edges', data: {source: nodeId + 'k', target: nodeId + 'Addition3'}},
                {group: 'edges', data: {source: nodeId + 'Multiplication5', target: nodeId + 'Addition3'}},

                {group: 'edges', data: {source: nodeId + 'Addition3', target: nodeId + 'IndexRes'}},
                {group: 'edges', data: {source: nodeId + 'displacementInMemory', target: nodeId + 'IndexRes'}}
            ])
            break
        case '110':
            cy.add([
                {group: 'nodes', data: {id: nodeId + 'j', parent: nodeId +'MatMul', label: '1'}, classes: 'input'},
                {group: 'nodes', data: {id: nodeId + 'k', parent: nodeId +'MatMul', label: 'k'}, classes: 'input'},
                {group: 'nodes', data: {id: nodeId + '#columns1', parent: nodeId +'MatMul', label: '# of columns of ' + incomingEdges[1].data('source'), value: dimensions1[1].dimValue}, classes: 'input'},
                {group: 'nodes', data: {id: nodeId + '#rows1', parent: nodeId +'MatMul', label: '# of rows of ' + incomingEdges[1].data('source'), value: dimensions1[0].dimValue}, classes: 'input'},
                {group: 'nodes', data: {id: nodeId + 'displacementInMemory', parent: nodeId + 'MatMul', label: 'displacement In Memory', value: typeSizeMap[type]}, classes: 'input'},
                {group: 'nodes', data: {id: nodeId + 'Index0', parent: nodeId + 'MatMul', label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: nodeId + 'Index1', parent: nodeId + 'MatMul', label: '*', opType: 'Multiplication'}, classes: 'operation'},

                {group: 'edges', data: {source: nodeId + 'j', target: nodeId + 'Index0'}},
                {group: 'edges', data: {source: nodeId + 'displacementInMemory', target: nodeId + 'Index0'}},

                {group: 'edges', data: {source: nodeId + 'k', target: nodeId + 'Index1'}},
                {group: 'edges', data: {source: nodeId + 'displacementInMemory', target: nodeId + 'Index1'}},

                //store's index

                {group: 'nodes', data: {id: nodeId + 'IndexRes', parent: nodeId + 'MatMul', label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'edges', data: {source: nodeId + 'k', target: nodeId + 'IndexRes'}},
            ])
            break
        case '011':
            cy.add([
                {group: 'nodes', data: {id: nodeId + 'j', parent: nodeId +'MatMul', label: '1'}, classes: 'input'},
                {group: 'nodes', data: {id: nodeId + 'k', parent: nodeId +'MatMul', label: 'k'}, classes: 'input'},
                {group: 'nodes', data: {id: nodeId + '#columns1', parent: nodeId +'MatMul', label: '# of columns of ' + incomingEdges[1].data('source'), value: dimensions1[1].dimValue}, classes: 'input'},
                {group: 'nodes', data: {id: nodeId + '#rows1', parent: nodeId +'MatMul', label: '# of rows of ' + incomingEdges[1].data('source'), value: dimensions1[0].dimValue}, classes: 'input'},
                {group: 'nodes', data: {id: nodeId + 'displacementInMemory', parent: nodeId + 'MatMul', label: 'displacement In Memory', value: typeSizeMap[type]}, classes: 'input'},
                {group: 'nodes', data: {id: nodeId + 'Index0', parent: nodeId + 'MatMul', label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: nodeId + 'Index1', parent: nodeId + 'MatMul', label: '*', opType: 'Multiplication'}, classes: 'operation'},

                {group: 'edges', data: {source: nodeId + 'k', target: nodeId + 'Index0'}},
                {group: 'edges', data: {source: nodeId + 'displacementInMemory', target: nodeId + 'Index0'}},

                {group: 'edges', data: {source: nodeId + 'j', target: nodeId + 'Index1'}},
                {group: 'edges', data: {source: nodeId + 'displacementInMemory', target: nodeId + 'Index1'}},

                //store's index

                {group: 'nodes', data: {id: nodeId + 'IndexRes', parent: nodeId + 'MatMul', label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'edges', data: {source: nodeId + 'k', target: nodeId + 'IndexRes'}},
            ])
            break
    }

    //multiplication of the loaded values and storing them in the result matrix (equal for any optimization)
    cy.add([

        {group: 'nodes', data: {id: nodeId + incomingEdges[0].data('source'), label: '&' + incomingEdges[0].data('source'), parent: nodeId + 'MatMul'}, classes: 'input'},
        {group: 'nodes', data: {id: nodeId + incomingEdges[1].data('source'), label: '&' + incomingEdges[1].data('source'), parent: nodeId + 'MatMul'}, classes: 'input'},
        {group: 'nodes', data: {id: nodeId + 'res', label: '&Result', parent: nodeId + 'MatMul'}, classes: 'output'},


        {group: 'nodes', data: {id: nodeId + 'Multiplication4', parent: nodeId + 'MatMul', label: '*', opType: 'Multiplication'}, classes: 'operation'},
        {group: 'nodes', data: {id: nodeId + 'Addition2', parent: nodeId + 'MatMul', label: '+', opType: 'Addition'}, classes: 'operation'},
        {group: 'nodes', data: {id: nodeId + 'Load0', parent: nodeId + 'MatMul', label: 'Load', opType: 'Load'}, classes: 'operation'},
        {group: 'nodes', data: {id: nodeId + 'Load1', parent: nodeId + 'MatMul', label: 'Load', opType: 'Load'}, classes: 'operation'},
        {group: 'nodes', data: {id: nodeId + 'Store', parent: nodeId + 'MatMul', label: 'Store', opType: 'Store'}, classes: 'operation'},



        {group: 'edges', data: {source: nodeId + 'Index0', target: nodeId + 'Load0'}},
        {group: 'edges', data: {source: nodeId + incomingEdges[0].data('source'), target: nodeId + 'Load0'}},

        {group: 'edges', data: {source: nodeId + 'Index1', target: nodeId + 'Load1'}},
        {group: 'edges', data: {source: nodeId + incomingEdges[1].data('source'), target: nodeId + 'Load1'}},

        {group: 'edges', data: {source: nodeId + 'Load0', target: nodeId + 'Multiplication4'}},
        {group: 'edges', data: {source: nodeId + 'Load1', target: nodeId + 'Multiplication4'}},

        {group: 'edges', data: {source: nodeId + 'Multiplication4', target: nodeId + 'Addition2'}},
        {group: 'edges', data: {source: nodeId + 'res', target: nodeId + 'Addition2'}},

        {group: 'edges', data: {source: nodeId + 'Addition2', target: nodeId + 'Store'}},

        {group: 'edges', data: {source: nodeId + 'IndexRes', target: nodeId + 'Store'}},

        {group: 'edges', data: {source: nodeId + 'Store', target: nodeId + 'res'}},

    ])

    switch (pattern) {
        case '000':
            cy.add([
                {group: 'nodes', data: {id: nodeId + '1', parent: nodeId + 'MatMul', label: '1'}, classes: 'constant'},

                {group: 'nodes', data: {id: nodeId + 'Addition4', parent: nodeId + 'MatMul', label: '+', opType: 'Addition'}, classes: 'operation'},
                {group: 'nodes', data: {id: nodeId + 'Addition5', parent: nodeId + 'MatMul', label: '+', opType: 'Addition'}, classes: 'operation'},
                {group: 'nodes', data: {id: nodeId + 'Addition6', parent: nodeId + 'MatMul', label: '+', opType: 'Addition'}, classes: 'operation'},
                {group: 'nodes', data: {id: nodeId + 'Addition7', parent: nodeId + 'MatMul', label: '+', opType: 'Addition'}, classes: 'operation'},
                {group: 'nodes', data: {id: nodeId + 'Multiplication7', parent: nodeId + 'MatMul', label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: nodeId + 'Multiplication8', parent: nodeId + 'MatMul', label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: nodeId + 'Multiplication9', parent: nodeId + 'MatMul', label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: nodeId + 'Multiplication10', parent: nodeId + 'MatMul', label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: nodeId + 'Multiplication11', parent: nodeId + 'MatMul', label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: nodeId + 'Multiplication12', parent: nodeId + 'MatMul', label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: nodeId + 'Equality0', parent: nodeId + 'MatMul', label: '==', opType: 'Equality'}, classes: 'operation'},
                {group: 'nodes', data: {id: nodeId + 'Equality1', parent: nodeId + 'MatMul', label: '==', opType: 'Equality'}, classes: 'operation'},
                {group: 'nodes', data: {id: nodeId + 'Not0', parent: nodeId + 'MatMul', label: '!', opType: 'Not'}, classes: 'operation'},
                {group: 'nodes', data: {id: nodeId + 'Not1', parent: nodeId + 'MatMul', label: '!', opType: 'Not'}, classes: 'operation'},


                {group: 'edges', data: {source: nodeId + 'k', target: nodeId + 'Equality0'}},
                {group: 'edges', data: {source: nodeId + '#rows1', target: nodeId + 'Equality0'}},

                {group: 'edges', data: {source: nodeId + '1', target: nodeId + 'Addition4'}},
                {group: 'edges', data: {source: nodeId + 'j', target: nodeId + 'Addition4'}},

                {group: 'edges', data: {source: nodeId + 'Equality0', target: nodeId + 'Multiplication7'}},
                {group: 'edges', data: {source: nodeId + 'Addition4', target: nodeId + 'Multiplication7'}},

                {group: 'edges', data: {source: nodeId + 'Equality0', target: nodeId + 'Not0'}},

                {group: 'edges', data: {source: nodeId + 'Not0', target: nodeId + 'Multiplication8'}},
                {group: 'edges', data: {source: nodeId + 'j', target: nodeId + 'Multiplication8'}},

                {group: 'edges', data: {source: nodeId + 'Multiplication7', target: nodeId + 'Addition5'}},
                {group: 'edges', data: {source: nodeId + 'Multiplication8', target: nodeId + 'Addition5'}},

                {group: 'edges', data: {source: nodeId + 'Addition5', target: nodeId + 'j'}},
                {group: 'edges', data: {source: nodeId + 'Not0', target: nodeId + 'Multiplication9'}},
                {group: 'edges', data: {source: nodeId + 'k', target: nodeId + 'Multiplication9'}},

                {group: 'edges', data: {source: nodeId + 'Multiplication9', target: nodeId + 'k'}},

                {group: 'edges', data: {source: nodeId + 'j', target: nodeId + 'Equality1'}},
                {group: 'edges', data: {source: nodeId + '#columns1', target: nodeId + 'Equality1'}},

                {group: 'edges', data: {source: nodeId + '1', target: nodeId + 'Addition6'}},
                {group: 'edges', data: {source: nodeId + 'i', target: nodeId + 'Addition6'}},

                {group: 'edges', data: {source: nodeId + 'Equality1', target: nodeId + 'Multiplication10'}},
                {group: 'edges', data: {source: nodeId + 'Addition6', target: nodeId + 'Multiplication10'}},

                {group: 'edges', data: {source: nodeId + 'Equality1', target: nodeId + 'Not1'}},

                {group: 'edges', data: {source: nodeId + 'Not1', target: nodeId + 'Multiplication11'}},
                {group: 'edges', data: {source: nodeId + 'i', target: nodeId + 'Multiplication11'}},

                {group: 'edges', data: {source: nodeId + 'Multiplication10', target: nodeId + 'Addition7'}},
                {group: 'edges', data: {source: nodeId + 'Multiplication11', target: nodeId + 'Addition7'}},

                {group: 'edges', data: {source: nodeId + 'Addition7', target: nodeId + 'i'}},

                {group: 'edges', data: {source: nodeId + 'Not1', target: nodeId + 'Multiplication12'}},
                {group: 'edges', data: {source: nodeId + 'j', target: nodeId + 'Multiplication12'}},

                {group: 'edges', data: {source: nodeId + 'Multiplication12', target: nodeId + 'j'}}
            ])
            break
        case '100':
        case '001':
        case '010':
            cy.add([
                {group: 'nodes', data: {id: nodeId + '1', parent: nodeId + 'MatMul', label: '1'}, classes: 'constant'},
                {group: 'nodes', data: {id: nodeId + 'Addition4', parent: nodeId + 'MatMul', label: '+', opType: 'Addition'}, classes: 'operation'},
                {group: 'nodes', data: {id: nodeId + 'Addition5', parent: nodeId + 'MatMul', label: '+', opType: 'Addition'}, classes: 'operation'},
                {group: 'nodes', data: {id: nodeId + 'Multiplication7', parent: nodeId + 'MatMul', label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: nodeId + 'Multiplication8', parent: nodeId + 'MatMul', label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: nodeId + 'Multiplication9', parent: nodeId + 'MatMul', label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: nodeId + 'Equality0', parent: nodeId + 'MatMul', label: '==', opType: 'Equality'}, classes: 'operation'},
                {group: 'nodes', data: {id: nodeId + 'Not0', parent: nodeId + 'MatMul', label: '!', opType: 'Not'}, classes: 'operation'},


                {group: 'edges', data: {source: nodeId + 'k', target: nodeId + 'Equality0'}},
                {group: 'edges', data: {source: nodeId + '#rows1', target: nodeId + 'Equality0'}},

                {group: 'edges', data: {source: nodeId + '1', target: nodeId + 'Addition4'}},
                {group: 'edges', data: {source: nodeId + 'j', target: nodeId + 'Addition4'}},

                {group: 'edges', data: {source: nodeId + 'Equality0', target: nodeId + 'Multiplication7'}},
                {group: 'edges', data: {source: nodeId + 'Addition4', target: nodeId + 'Multiplication7'}},

                {group: 'edges', data: {source: nodeId + 'Equality0', target: nodeId + 'Not0'}},

                {group: 'edges', data: {source: nodeId + 'Not0', target: nodeId + 'Multiplication8'}},
                {group: 'edges', data: {source: nodeId + 'j', target: nodeId + 'Multiplication8'}},

                {group: 'edges', data: {source: nodeId + 'Multiplication7', target: nodeId + 'Addition5'}},
                {group: 'edges', data: {source: nodeId + 'Multiplication8', target: nodeId + 'Addition5'}},

                {group: 'edges', data: {source: nodeId + 'Addition5', target: nodeId + 'j'}},
                {group: 'edges', data: {source: nodeId + 'Not0', target: nodeId + 'Multiplication9'}},
                {group: 'edges', data: {source: nodeId + 'k', target: nodeId + 'Multiplication9'}},

                {group: 'edges', data: {source: nodeId + 'Multiplication9', target: nodeId + 'k'}}
            ])
            break
    }
    //checking of indexes


    outgoingEdges.forEach(edge => {
        cy.add({group: 'edges', data: {source: nodeId + 'MatMul', target: edge.data('target'), dims: edge.data('dims'), elemType: edge.data('elemType')}})
    })


    cy.remove(node)
}

export function transformOpps(cy) {
    cy.nodes('.operation').forEach(node => {
        if (node.data('opType') === 'Add'){
            transformAdd(node, cy)
        }
        else if (node.data('opType') === 'MatMul') {
            transformMatMul(node, cy)
        }
    })
}
