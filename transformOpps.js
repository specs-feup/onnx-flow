
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

function formatId(name, nodeId) {
    return `${name}_${nodeId}`;
}



function transformSimpleLoopOperations(node, cy, edgeOrder) {

    let opLabel
    let opType

    switch (node.data('opType')) {
        case 'Add':
            opLabel = '+'
            opType = 'Addition'
            break
        case 'Sub':
            opLabel = '-'
            opType = 'Subtraction'
            break
        case 'Mul':
            opLabel = '*'
            opType = 'Multiplication'
            break
        case 'Div':
            opLabel = '/'
            opType = 'Division'
            break
        default:
            return
    }

    const incomingEdges = node.incomers('edge');
    const outgoingEdges = node.outgoers('edge');
    const dimensions = incomingEdges[0].data('dims');
    const type = incomingEdges[0].data('elemType');

    const nodeId = node.data('id');TestFail

    if (dimensions[0].dimValue === '1' && dimensions[1].dimValue === '1') {
        cy.add([
            {group: 'nodes', data: {id: formatId(opType, nodeId), label: opLabel, opType: opType}, classes: 'operation'},
            {group: 'edges', data: {source: incomingEdges[0].data('source'), target: formatId(opType, nodeId), label: incomingEdges[0].data('label'), dims: incomingEdges[0].data('dims'), elemType: incomingEdges[0].data('elemType'), opType: incomingEdges[0].data('opType'), order: edgeOrder.value++}, classes: incomingEdges[0].classes()},
            {group: 'edges', data: {source: incomingEdges[1].data('source'), target: formatId(opType, nodeId), label: incomingEdges[1].data('label'), dims: incomingEdges[1].data('dims'), elemType: incomingEdges[1].data('elemType'), opType: incomingEdges[1].data('opType'), order: edgeOrder.value++}, classes: incomingEdges[1].classes()}
        ]);
        outgoingEdges.forEach(edge => {
            cy.add({group: 'edges', data: {source: formatId(opType, nodeId), target: edge.data('target'), label: edge.data('label'), dims: edge.data('dims'), elemType: edge.data('elemType'), opType: opType, order: edgeOrder.value++}, classes: 'operation variable outer'});
        });
    } else {
        let numberOfIterations = dimensions.reduce((total, dim) => total + parseInt(dim.dimValue), 0);
        let order = 0;
        let displacementInMemory = typeSizeMap[type];
        cy.add([
            {group: 'nodes', data: {id: formatId('LoopIterations', nodeId), label: '# of loop iterations', value: numberOfIterations}, classes: 'constant'},
            {group: 'nodes', data: {id: formatId(node.data('opType'), nodeId), label: node.data('opType'), opType: node.data('opType')}},
            {group: 'nodes', data: {id: formatId('index', nodeId), parent: formatId(node.data('opType'), nodeId), label: 'index'}, classes: 'input'},
            {group: 'nodes', data: {id: formatId('displacementInMemory', nodeId), parent: formatId(node.data('opType'), nodeId), label: 'displacement In Memory', value: displacementInMemory}, classes: 'input'},
            {group: 'nodes', data: {id: formatId(incomingEdges[0].data('source'), nodeId), label: '&' + incomingEdges[0].data('source'), parent: formatId(node.data('opType'), nodeId)}, classes: 'input'},
            {group: 'nodes', data: {id: formatId(incomingEdges[1].data('source'), nodeId), label: '&' + incomingEdges[1].data('source'), parent: formatId(node.data('opType'), nodeId)}, classes: 'input'},
            {group: 'nodes', data: {id: formatId(`${node.data('opType')}_${nodeId}`, nodeId), label: '&Result', parent: formatId(node.data('opType'), nodeId)}, classes: 'output'},
            {group: 'nodes', data: {id: formatId('Multiplication', nodeId), parent: formatId(node.data('opType'), nodeId), label: '*', opType: 'Multiplication'}, classes: 'operation'},
            {group: 'nodes', data: {id: formatId('Load0', nodeId), parent: formatId(node.data('opType'), nodeId), label: 'Load', opType: 'Load'}, classes: 'operation'},
            {group: 'nodes', data: {id: formatId('Load1', nodeId), parent: formatId(node.data('opType'), nodeId), label: 'Load', opType: 'Load'}, classes: 'operation'},
            {group: 'nodes', data: {id: formatId(opType, nodeId), parent: formatId(node.data('opType'), nodeId), label: '+', opType: opType}, classes: 'operation'},
            {group: 'nodes', data: {id: formatId('Addition1', nodeId), parent: formatId(node.data('opType'), nodeId), label: '+', opType: 'Addition'}, classes: 'operation'},
            {group: 'nodes', data: {id: formatId('Store', nodeId), parent: formatId(node.data('opType'), nodeId), label: 'Store', opType: 'Store'}, classes: 'operation'},
            {group: 'nodes', data: {id: formatId('1', nodeId), parent: formatId(node.data('opType'), nodeId), label: '1'}, classes: 'constant'},

            {group: 'edges', data: {source: formatId('LoopIterations', nodeId), target: formatId(node.data('opType'), nodeId), value: numberOfIterations, order: edgeOrder.value++}, classes: 'constant'},
            {group: 'edges', data: {source: incomingEdges[0].data('source'), label: incomingEdges[0].data('label'), target: formatId(node.data('opType'), nodeId), dims: incomingEdges[0].data('dims'), elemType: incomingEdges[0].data('elemType'), opType: incomingEdges[0].data('opType'), order: edgeOrder.value++}, classes: incomingEdges[0].classes()},
            {group: 'edges', data: {source: incomingEdges[1].data('source'), label: incomingEdges[1].data('label'), target: formatId(node.data('opType'), nodeId), dims: incomingEdges[1].data('dims'), elemType: incomingEdges[1].data('elemType'), opType: incomingEdges[1].data('opType'), order: edgeOrder.value++}, classes: incomingEdges[1].classes()},

            {group: 'edges', data: {source: formatId('index', nodeId), target: formatId('Multiplication', nodeId), parent: formatId(node.data('opType'), nodeId), order: order++}, classes: 'declareBefore index'},
            {group: 'edges', data: {source: formatId('displacementInMemory', nodeId), target: formatId('Multiplication', nodeId), parent: formatId(node.data('opType'), nodeId), order: order++, value: displacementInMemory}, classes: 'constant'},
            {group: 'edges', data: {source: formatId(incomingEdges[0].data('source'), nodeId), target: formatId('Load0', nodeId), parent: formatId(node.data('opType'), nodeId), order: order++}, classes: 'input'},
            {group: 'edges', data: {source: formatId('Multiplication', nodeId), target: formatId('Load0', nodeId), parent: formatId(node.data('opType'), nodeId), order: order++, opType: 'Multiplication'}, classes: 'operation'},
            {group: 'edges', data: {source: formatId(incomingEdges[1].data('source'), nodeId), target: formatId('Load1', nodeId), parent: formatId(node.data('opType'), nodeId), order: order++}, classes: 'input'},
            {group: 'edges', data: {source: formatId('Multiplication', nodeId), target: formatId('Load1', nodeId), parent: formatId(node.data('opType'), nodeId), order: order++}, classes: 'operation'},
            {group: 'edges', data: {source: formatId('Load0', nodeId), target: formatId(opType, nodeId), parent: formatId(node.data('opType'), nodeId), order: order++, opType: 'Load'}, classes: 'operation'},
            {group: 'edges', data: {source: formatId('Load1', nodeId), target: formatId(opType, nodeId), parent: formatId(node.data('opType'), nodeId), order: order++, opType: 'Load'}, classes: 'operation'},
            {group: 'edges', data: {source: formatId(opType, nodeId), target: formatId('Store', nodeId), parent: formatId(node.data('opType'), nodeId), order: order++, opType: opType}, classes: 'operation'},
            {group: 'edges', data: {source: formatId('Multiplication', nodeId), target: formatId('Store', nodeId), parent: formatId(node.data('opType'), nodeId), order: order++, opType: 'Multiplication'}, classes: 'operation'},
            {group: 'edges', data: {source: formatId('Store', nodeId), target: formatId(`${node.data('opType')}_${nodeId}`, nodeId), parent: formatId(node.data('opType'), nodeId), order: order++, opType: 'Store'}, classes: 'operation result'},

            {group: 'edges', data: {source: formatId('index', nodeId), target: formatId('Addition1', nodeId), parent: formatId(node.data('opType'), nodeId), order: order++}, classes: 'declareBefore'},
            {group: 'edges', data: {source: formatId('1', nodeId), target: formatId('Addition1', nodeId), parent: formatId(node.data('opType'), nodeId), order: order++, value: 1}, classes: 'constant'},
            {group: 'edges', data: {source: formatId('Addition1', nodeId), target: formatId('index', nodeId), parent: formatId(node.data('opType'), nodeId), order: order++, opType: 'Addition'}, classes: 'operation variable'}
        ]);

        outgoingEdges.forEach(edge => {
            cy.add({group: 'edges', data: {source: formatId(node.data('opType'), nodeId), target: edge.data('target'), label: edge.data('label'), dims: edge.data('dims'), elemType: edge.data('elemType'), order: edgeOrder.value++}, classes: 'compound variable'});
        });
    }
    cy.remove(node);
}


function transformMatMul(node, cy, edgeOrder) {
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
            {group: 'nodes', data: {id: formatId('Multiplication', nodeId), label: '*', opType: 'Multiplication'}, classes: 'operation'},
            {group: 'edges', data: {source: incomingEdges[0].data('source'), target: formatId('Multiplication', nodeId), dims: incomingEdges[0].data('dims'), elemType: incomingEdges[0].data('elemType'), opType: incomingEdges[0].data('opType'), order: edgeOrder.value++}, classes: incomingEdges[0].classes()},
            {group: 'edges', data: {source: incomingEdges[1].data('source'), target: formatId('Multiplication', nodeId), dims: incomingEdges[1].data('dims'), elemType: incomingEdges[1].data('elemType'), opType: incomingEdges[1].data('opType'), order: edgeOrder.value++},  classes: incomingEdges[1].classes()}
        ])
        outgoingEdges.forEach(edge => {
            cy.add({group: 'edges', data: {source: formatId('Multiplication', nodeId), target: edge.data('target'), dims: edge.data('dims'), elemType: edge.data('elemType'), order: edgeOrder.value++, opType: 'Multiplication'}, classes: 'operation variable outer'})
        })
        cy.remove(node)
        return
    }

    cy.add([
        {group: 'nodes', data: {id: formatId('LoopIterations', nodeId), label: '# of loop iterations', value: numberOfIterations}, classes: 'constant'},
        {group: 'nodes', data: {id: formatId('MatMul', nodeId), label: 'MatMul', opType: 'MatMul'}},

        {group: 'edges', data: {source: formatId('LoopIterations', nodeId), target: formatId('MatMul', nodeId), order: edgeOrder.value++, value: numberOfIterations}, classes: 'constant'},
        {group: 'edges', data: {source: incomingEdges[0].data('source'), target: formatId('MatMul', nodeId), dims: incomingEdges[0].data('dims'), elemType: incomingEdges[0].data('elemType'), opType: incomingEdges[0].data('opType'), order: edgeOrder.value++}, classes: incomingEdges[0].classes()},
        {group: 'edges', data: {source: incomingEdges[1].data('source'), target: formatId('MatMul', nodeId), dims: incomingEdges[1].data('dims'), elemType: incomingEdges[1].data('elemType'), opType: incomingEdges[1].data('opType'), order: edgeOrder.value++}, classes: incomingEdges[1].classes()},

    ])

    let order = 0;

    switch (pattern) {
        case '000':
            cy.add([
                {group: 'nodes', data: {id: formatId('i', nodeId), parent: formatId('MatMul', nodeId), label: 'i'}, classes: 'input'},
                {group: 'nodes', data: {id: formatId('j', nodeId), parent: formatId('MatMul', nodeId), label: 'j'}, classes: 'input'},
                {group: 'nodes', data: {id: formatId('k', nodeId), parent: formatId('MatMul', nodeId), label: 'k'}, classes: 'input'},
                {group: 'nodes', data: {id: formatId('#columns1', nodeId), parent: formatId('MatMul', nodeId), label: '# of columns of ' + incomingEdges[1].data('source'), value: dimensions1[1].dimValue}, classes: 'input'},
                {group: 'nodes', data: {id: formatId('#rows1', nodeId), parent: formatId('MatMul', nodeId), label: '# of rows of ' + incomingEdges[1].data('source'), value: dimensions1[0].dimValue}, classes: 'input'},
                {group: 'nodes', data: {id: formatId('displacementInMemory', nodeId), parent: formatId('MatMul', nodeId), label: 'displacement In Memory', value: typeSizeMap[type]}, classes: 'input'},
                {group: 'nodes', data: {id: formatId('Multiplication0', nodeId), parent: formatId('MatMul', nodeId), label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: formatId('Multiplication1', nodeId), parent: formatId('MatMul', nodeId), label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: formatId('Index0', nodeId), parent: formatId('MatMul', nodeId), label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: formatId('Index1', nodeId), parent: formatId('MatMul', nodeId), label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: formatId('Addition0', nodeId), parent: formatId('MatMul', nodeId), label: '+', opType: 'Addition'}, classes: 'operation'},
                {group: 'nodes', data: {id: formatId('Addition1', nodeId), parent: formatId('MatMul', nodeId), label: '+', opType: 'Addition'}, classes: 'operation'},

                //Multiplication0 = i * number of rows of matrix B
                {group: 'edges', data: {source: formatId('i', nodeId), target: formatId('Multiplication0', nodeId), parent: formatId('MatMul', nodeId), order: order++}, classes: 'declareBefore'},
                {group: 'edges', data: {source: formatId('#rows1', nodeId), target: formatId('Multiplication0', nodeId), parent: formatId('MatMul', nodeId), order: order++, value: dimensions1[0].dimValue}, classes: 'constant'},

                //Multiplication1 = k * number of columns of matrix B
                {group: 'edges', data: {source: formatId('k', nodeId), target: formatId('Multiplication1', nodeId), parent: formatId('MatMul', nodeId), order: order++}, classes: 'declareBefore'},
                {group: 'edges', data: {source: formatId('#columns1', nodeId), target: formatId('Multiplication1', nodeId), parent: formatId('MatMul', nodeId), order: order++, value: dimensions1[1].dimValue}, classes: 'constant'},

                //Addition0 = Multiplication0 + k
                {group: 'edges', data: {source: formatId('Multiplication0', nodeId), target: formatId('Addition0', nodeId), parent: formatId('MatMul', nodeId), order: order++, opType: 'Multiplication'}, classes: 'operation'},
                {group: 'edges', data: {source: formatId('k', nodeId), target: formatId('Addition0', nodeId), parent: formatId('MatMul', nodeId), order: order++}, classes: 'declareBefore'},

                //Addition1 = Multiplication1 + j
                {group: 'edges', data: {source: formatId('Multiplication1', nodeId), target: formatId('Addition1', nodeId), parent: formatId('MatMul', nodeId), order: order++, opType: 'Multiplication'}, classes: 'operation'},
                {group: 'edges', data: {source: formatId('j', nodeId), target: formatId('Addition1', nodeId), parent: formatId('MatMul', nodeId), order: order++}, classes: 'declareBefore'},

                //Index0 (the memory position for matrix A) = Addition0 * the displacement in memory
                {group: 'edges', data: {source: formatId('Addition0', nodeId), target: formatId('Index0', nodeId), parent: formatId('MatMul', nodeId), order: order++, opType: 'Addition'}, classes: 'operation'},
                {group: 'edges', data: {source: formatId('displacementInMemory', nodeId), target: formatId('Index0', nodeId), parent: formatId('MatMul', nodeId), order: order++, value: typeSizeMap[type]}, classes: 'constant'},

                //Index1 (the memory position for matrix B) = Addition1 * the displacement in memory
                {group: 'edges', data: {source: formatId('Addition1', nodeId), target: formatId('Index1', nodeId), parent: formatId('MatMul', nodeId), order: order++, opType: 'Addition'}, classes: 'operation'},
                {group: 'edges', data: {source: formatId('displacementInMemory', nodeId), target: formatId('Index1', nodeId), parent: formatId('MatMul', nodeId), order: order++, value: typeSizeMap[type]}, classes: 'constant'},

                //store's index
                {group: 'nodes', data: {id: formatId('Multiplication5', nodeId), parent: formatId('MatMul', nodeId), label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: formatId('IndexRes', nodeId), parent: formatId('MatMul', nodeId), label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: formatId('Addition3', nodeId), parent: formatId('MatMul', nodeId), label: '+', opType: 'Addition'}, classes: 'operation'},

                //Multiplication5 = i * number of columns of matrix B
                {group: 'edges', data: {source: formatId('i', nodeId), target: formatId('Multiplication5', nodeId), parent: formatId('MatMul', nodeId), order: order++}, classes: 'declareBefore'},
                {group: 'edges', data: {source: formatId('#columns1', nodeId), target: formatId('Multiplication5', nodeId), parent: formatId('MatMul', nodeId), order: order++, value: dimensions1[1].dimValue}, classes: 'constant'},

                //Addition3 = j * Multiplication 5
                {group: 'edges', data: {source: formatId('j', nodeId), target: formatId('Addition3', nodeId), parent: formatId('MatMul', nodeId), order: order++}, classes: 'declareBefore'},
                {group: 'edges', data: {source: formatId('Multiplication5', nodeId), target: formatId('Addition3', nodeId), parent: formatId('MatMul', nodeId), order: order++, opType: 'Multiplication'}, classes: 'operation'},

                //IndexRes (the memory position for the result matrix) = Addition3 * the displacement in memory
                {group: 'edges', data: {source: formatId('Addition3', nodeId), target: formatId('IndexRes', nodeId), parent: formatId('MatMul', nodeId), order: order++, opType: 'Addition'}, classes: 'operation'},
                {group: 'edges', data: {source: formatId('displacementInMemory', nodeId), target: formatId('IndexRes', nodeId), parent: formatId('MatMul', nodeId), order: order++, value: typeSizeMap[type]}, classes: 'constant'}


            ])
            break
        case '100':

            cy.add([
                {group: 'nodes', data: {id: formatId('j', nodeId), parent: formatId('MatMul', nodeId), label: 'j'}, classes: 'input'},
                {group: 'nodes', data: {id: formatId('k', nodeId), parent: formatId('MatMul', nodeId), label: 'k'}, classes: 'input'},
                {group: 'nodes', data: {id: formatId('#rows1', nodeId), parent: formatId('MatMul', nodeId), label: '# of rows of ' + incomingEdges[1].data('source'), value: dimensions1[0].dimValue}, classes: 'input'},
                {group: 'nodes', data: {id: formatId('#columns1', nodeId), parent: formatId('MatMul', nodeId), label: '# of columns of ' + incomingEdges[1].data('source'), value: dimensions1[1].dimValue}, classes: 'input'},
                {group: 'nodes', data: {id: formatId('displacementInMemory', nodeId), parent: formatId('MatMul', nodeId), label: 'displacement In Memory', value: typeSizeMap[type]}, classes: 'input'},
                {group: 'nodes', data: {id: formatId('Multiplication1', nodeId), parent: formatId('MatMul', nodeId), label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: formatId('Index0', nodeId), parent: formatId('MatMul', nodeId), label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: formatId('Index1', nodeId), parent: formatId('MatMul', nodeId), label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: formatId('Addition1', nodeId), parent: formatId('MatMul', nodeId), label: '+', opType: 'Addition'}, classes: 'operation'},


                {group: 'edges', data: {source: formatId('k', nodeId), target: formatId('Multiplication1', nodeId), parent: formatId('MatMul', nodeId), order: order++}, classes: 'declareBefore'},
                {group: 'edges', data: {source: formatId('#columns1', nodeId), target: formatId('Multiplication1', nodeId), parent: formatId('MatMul', nodeId), order: order++, value: dimensions1[1].dimValue}, classes: 'constant'},

                {group: 'edges', data: {source: formatId('k', nodeId), target: formatId('Index0', nodeId), parent: formatId('MatMul', nodeId), order: order++}, classes: 'declareBefore'},
                {group: 'edges', data: {source: formatId('displacementInMemory', nodeId), target: formatId('Index0', nodeId), parent: formatId('MatMul', nodeId), order: order++, value: typeSizeMap[type]}, classes: 'constant'},

                {group: 'edges', data: {source: formatId('j', nodeId), target: formatId('Addition1', nodeId), parent: formatId('MatMul', nodeId), order: order++}, classes: 'declareBefore'},
                {group: 'edges', data: {source: formatId('Multiplication1', nodeId), target: formatId('Addition1', nodeId), parent: formatId('MatMul', nodeId), order: order++, opType: 'Multiplication'}, classes: 'operation'},

                {group: 'edges', data: {source: formatId('Addition1', nodeId), target: formatId('Index1', nodeId), parent: formatId('MatMul', nodeId), order: order++, opType: 'Addition'}, classes: 'operation'},
                {group: 'edges', data: {source: formatId('displacementInMemory', nodeId), target: formatId('Index1', nodeId), parent: formatId('MatMul', nodeId), order: order++, value: typeSizeMap[type]}, classes: 'constant'},

                //store's index

                {group: 'nodes', data: {id: formatId('IndexRes', nodeId), parent: formatId('MatMul', nodeId), label: '*', opType: 'Multiplication'}, classes: 'operation'},

                {group: 'edges', data: {source: formatId('j', nodeId), target: formatId('IndexRes', nodeId), parent: formatId('MatMul', nodeId), order: order++}, classes: 'declareBefore'},
                {group: 'edges', data: {source: formatId('displacementInMemory', nodeId), target: formatId('IndexRes', nodeId), parent: formatId('MatMul', nodeId), order: order++, value: typeSizeMap[type]}, classes: 'constant'}


            ])
            break
        case '001':
            cy.add([
                {group: 'nodes', data: {id: formatId('j', nodeId), parent: formatId('MatMul', nodeId), label: 'j'}, classes: 'input'},
                {group: 'nodes', data: {id: formatId('k', nodeId), parent: formatId('MatMul', nodeId), label: 'k'}, classes: 'input'},
                {group: 'nodes', data: {id: formatId('#columns1', nodeId), parent: formatId('MatMul', nodeId), label: '# of columns of ' + incomingEdges[1].data('source'), value: dimensions1[1].dimValue}, classes: 'input'},
                {group: 'nodes', data: {id: formatId('#rows1', nodeId), parent: formatId('MatMul', nodeId), label: '# of rows of ' + incomingEdges[1].data('source'), value: dimensions1[0].dimValue}, classes: 'input'},
                {group: 'nodes', data: {id: formatId('displacementInMemory', nodeId), parent: formatId('MatMul', nodeId), label: 'displacement In Memory', value: typeSizeMap[type]}, classes: 'input'},
                {group: 'nodes', data: {id: formatId('Multiplication0', nodeId), parent: formatId('MatMul', nodeId), label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: formatId('Index0', nodeId), parent: formatId('MatMul', nodeId), label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: formatId('Index1', nodeId), parent: formatId('MatMul', nodeId), label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: formatId('Addition0', nodeId), parent: formatId('MatMul', nodeId), label: '+', opType: 'Addition'}, classes: 'operation'},

                {group: 'edges', data: {source: formatId('j', nodeId), target: formatId('Multiplication0', nodeId), parent: formatId('MatMul', nodeId), order: order++}, classes: 'declareBefore'},
                {group: 'edges', data: {source: formatId('#rows1', nodeId), target: formatId('Multiplication0', nodeId), parent: formatId('MatMul', nodeId), order: order++, value: dimensions1[0].dimValue}, classes: 'constant'},

                {group: 'edges', data: {source: formatId('Multiplication0', nodeId), target: formatId('Addition0', nodeId), parent: formatId('MatMul', nodeId), order: order++, opType: 'Multiplication'}, classes: 'operation'},
                {group: 'edges', data: {source: formatId('k', nodeId), target: formatId('Addition0', nodeId), parent: formatId('MatMul', nodeId), order: order++}, classes: 'declareBefore'},

                {group: 'edges', data: {source: formatId('k', nodeId), target: formatId('Index1', nodeId), parent: formatId('MatMul', nodeId), order: order++}, classes: 'declareBefore'},

                {group: 'edges', data: {source: formatId('Addition0', nodeId), target: formatId('Index0', nodeId), parent: formatId('MatMul', nodeId), order: order++, opType: 'Addition'}, classes: 'operation'},
                {group: 'edges', data: {source: formatId('displacementInMemory', nodeId), target: formatId('Index0', nodeId), parent: formatId('MatMul', nodeId), order: order++, value: typeSizeMap[type]}, classes: 'constant'},

                {group: 'edges', data: {source: formatId('displacementInMemory', nodeId), target: formatId('Index1', nodeId), parent: formatId('MatMul', nodeId), order: order++, value: typeSizeMap[type]}, classes: 'constant'},

                //store's index

                {group: 'nodes', data: {id: formatId('IndexRes', nodeId), parent: formatId('MatMul', nodeId), label: '*', opType: 'Multiplication'}, classes: 'operation'},

                {group: 'edges', data: {source: formatId('j', nodeId), target: formatId('IndexRes', nodeId), parent: formatId('MatMul', nodeId), order: order++}, classes: 'declareBefore'},
                {group: 'edges', data: {source: formatId('displacementInMemory', nodeId), target: formatId('IndexRes', nodeId), parent: formatId('MatMul', nodeId), order: order++, value: typeSizeMap[type]}, classes: 'constant'}


            ])
            break
        case '101':
            cy.add([
                {group: 'nodes', data: {id: formatId('k', nodeId), parent: formatId('MatMul', nodeId), label: 'k'}, classes: 'input'},
                {group: 'nodes', data: {id: formatId('displacementInMemory', nodeId), parent: formatId('MatMul', nodeId), label: 'displacement In Memory', value: typeSizeMap[type]}, classes: 'input'},
                {group: 'nodes', data: {id: formatId('Index0', nodeId), parent: formatId('MatMul', nodeId), label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: formatId('Index1', nodeId), parent: formatId('MatMul', nodeId), label: '*', opType: 'Multiplication'}, classes: 'operation'},

                {group: 'edges', data: {source: formatId('k', nodeId), target: formatId('Index0', nodeId), parent: formatId('MatMul', nodeId), order: order++}, classes: 'declareBefore'},
                {group: 'edges', data: {source: formatId('displacementInMemory', nodeId), target: formatId('Index0', nodeId), parent: formatId('MatMul', nodeId), order: order++, value: typeSizeMap[type]}, classes: 'constant'},
                {group: 'edges', data: {source: formatId('k', nodeId), target: formatId('Index1', nodeId), parent: formatId('MatMul', nodeId), order: order++}, classes: 'declareBefore'},
                {group: 'edges', data: {source: formatId('displacementInMemory', nodeId), target: formatId('Index1', nodeId), parent: formatId('MatMul', nodeId), order: order++, value: typeSizeMap[type]}, classes: 'constant'},

                //store's index

                {group: 'nodes', data: {id: formatId('IndexRes', nodeId), parent: formatId('MatMul', nodeId), label: '*', opType: 'Multiplication'}, classes: 'operation'},

                {group: 'edges', data: {source: formatId('displacementInMemory', nodeId), target: formatId('IndexRes', nodeId), parent: formatId('MatMul', nodeId), order: order++, value: typeSizeMap[type]}, classes: 'constant'}

            ])
            break
        case '010':
            cy.add([
                {group: 'nodes', data: {id: formatId('j', nodeId), parent: formatId('MatMul', nodeId), label: 'j'}, classes: 'input'},
                {group: 'nodes', data: {id: formatId('k', nodeId), parent: formatId('MatMul', nodeId), label: 'k'}, classes: 'input'},
                {group: 'nodes', data: {id: formatId('#columns1', nodeId), parent: formatId('MatMul', nodeId), label: '# of columns of ' + incomingEdges[1].data('source'), value: dimensions1[1].dimValue}, classes: 'input'},
                {group: 'nodes', data: {id: formatId('#rows1', nodeId), parent: formatId('MatMul', nodeId), label: '# of rows of ' + incomingEdges[1].data('source'), value: dimensions1[0].dimValue}, classes: 'input'},
                {group: 'nodes', data: {id: formatId('displacementInMemory', nodeId), parent: formatId('MatMul', nodeId), label: 'displacement In Memory', value: typeSizeMap[type]}, classes: 'input'},
                {group: 'nodes', data: {id: formatId('Index0', nodeId), parent: formatId('MatMul', nodeId), label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: formatId('Index1', nodeId), parent: formatId('MatMul', nodeId), label: '*', opType: 'Multiplication'}, classes: 'operation'},

                {group: 'edges', data: {source: formatId('j', nodeId), target: formatId('Index0', nodeId), parent: formatId('MatMul', nodeId), order: order++}, classes: 'declareBefore'},
                {group: 'edges', data: {source: formatId('displacementInMemory', nodeId), target: formatId('Index0', nodeId), parent: formatId('MatMul', nodeId), order: order++, value: typeSizeMap[type]}, classes: 'constant'},

                {group: 'edges', data: {source: formatId('k', nodeId), target: formatId('Index1', nodeId), parent: formatId('MatMul', nodeId), order: order++}, classes: 'declareBefore'},
                {group: 'edges', data: {source: formatId('displacementInMemory', nodeId), target: formatId('Index1', nodeId), parent: formatId('MatMul', nodeId), order: order++, value: typeSizeMap[type]}, classes: 'constant'},

                //store's index

                {group: 'nodes', data: {id: formatId('Multiplication5', nodeId), parent: formatId('MatMul', nodeId), label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: formatId('IndexRes', nodeId), parent: formatId('MatMul', nodeId), label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: formatId('Addition3', nodeId), parent: formatId('MatMul', nodeId), label: '+', opType: 'Addition'}, classes: 'operation'},

                {group: 'edges', data: {source: formatId('j', nodeId), target: formatId('Multiplication5', nodeId), parent: formatId('MatMul', nodeId), order: order++}, classes: 'declareBefore'},
                {group: 'edges', data: {source: formatId('#columns1', nodeId), target: formatId('Multiplication5', nodeId), parent: formatId('MatMul', nodeId), order: order++, value: dimensions1[1].dimValue }, classes: 'constant'},

                {group: 'edges', data: {source: formatId('k', nodeId), target: formatId('Addition3', nodeId), parent: formatId('MatMul', nodeId), order: order++}, classes: 'declareBefore'},
                {group: 'edges', data: {source: formatId('Multiplication5', nodeId), target: formatId('Addition3', nodeId), parent: formatId('MatMul', nodeId), order: order++, opType: 'Multiplication'}, classes: 'operation'},

                {group: 'edges', data: {source: formatId('Addition3', nodeId), target: formatId('IndexRes', nodeId), parent: formatId('MatMul', nodeId), order: order++, opType: 'Addition'}, classes: 'operation'},
                {group: 'edges', data: {source: formatId('displacementInMemory', nodeId), target: formatId('IndexRes', nodeId), parent: formatId('MatMul', nodeId), order: order++, value: typeSizeMap[type]}, classes: 'constant'}

            ])
            break
        case '110':
            cy.add([
                {group: 'nodes', data: {id: formatId('j', nodeId), parent: formatId('MatMul', nodeId), label: '1'}, classes: 'input'},
                {group: 'nodes', data: {id: formatId('IndexRes', nodeId), parent: formatId('MatMul', nodeId), label: 'IndexRes'}, classes: 'input'},
                {group: 'nodes', data: {id: formatId('#columns1', nodeId), parent: formatId('MatMul', nodeId), label: '# of columns of ' + incomingEdges[1].data('source'), value: dimensions1[1].dimValue}, classes: 'input'},
                {group: 'nodes', data: {id: formatId('#rows1', nodeId), parent: formatId('MatMul', nodeId), label: '# of rows of ' + incomingEdges[1].data('source'), value: dimensions1[0].dimValue}, classes: 'input'},
                {group: 'nodes', data: {id: formatId('displacementInMemory', nodeId), parent: formatId('MatMul', nodeId), label: 'displacement In Memory', value: typeSizeMap[type]}, classes: 'input'},
                {group: 'nodes', data: {id: formatId('Index0', nodeId), parent: formatId('MatMul', nodeId), label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: formatId('Index1', nodeId), parent: formatId('MatMul', nodeId), label: '*', opType: 'Multiplication'}, classes: 'operation'},

                {group: 'edges', data: {source: formatId('j', nodeId), target: formatId('Index0', nodeId), parent: formatId('MatMul', nodeId), order: order++}, classes: 'declareBefore'},
                {group: 'edges', data: {source: formatId('displacementInMemory', nodeId), target: formatId('Index0', nodeId), parent: formatId('MatMul', nodeId), order: order++, value: typeSizeMap[type]}, classes: 'constant'},

                {group: 'edges', data: {source: formatId('IndexRes', nodeId), target: formatId('Index1', nodeId), parent: formatId('MatMul', nodeId), order: order++}, classes: 'declareBefore'},
                {group: 'edges', data: {source: formatId('displacementInMemory', nodeId), target: formatId('Index1', nodeId), parent: formatId('MatMul', nodeId), order: order++, value: typeSizeMap[type]}, classes: 'constant'},
            ])
            break
        case '011':
            cy.add([
                {group: 'nodes', data: {id: formatId('j', nodeId), parent: formatId('MatMul', nodeId), label: '1'}, classes: 'input'},
                {group: 'nodes', data: {id: formatId('IndexRes', nodeId), parent: formatId('MatMul', nodeId), label: 'IndexRes'}, classes: 'input'},
                {group: 'nodes', data: {id: formatId('#columns1', nodeId), parent: formatId('MatMul', nodeId), label: '# of columns of ' + incomingEdges[1].data('source'), value: dimensions1[1].dimValue}, classes: 'input'},
                {group: 'nodes', data: {id: formatId('#rows1', nodeId), parent: formatId('MatMul', nodeId), label: '# of rows of ' + incomingEdges[1].data('source'), value: dimensions1[0].dimValue}, classes: 'input'},
                {group: 'nodes', data: {id: formatId('displacementInMemory', nodeId), parent: formatId('MatMul', nodeId), label: 'displacement In Memory', value: typeSizeMap[type]}, classes: 'input'},
                {group: 'nodes', data: {id: formatId('Index0', nodeId), parent: formatId('MatMul', nodeId), label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: formatId('Index1', nodeId), parent: formatId('MatMul', nodeId), label: '*', opType: 'Multiplication'}, classes: 'operation'},

                {group: 'edges', data: {source: formatId('IndexRes', nodeId), target: formatId('Index0', nodeId), parent: formatId('MatMul', nodeId), order: order++}, classes: 'declareBefore'},
                {group: 'edges', data: {source: formatId('displacementInMemory', nodeId), target: formatId('Index0', nodeId), parent: formatId('MatMul', nodeId), order: order++, value: typeSizeMap[type]}, classes: 'constant'},

                {group: 'edges', data: {source: formatId('j', nodeId), target: formatId('Index1', nodeId), parent: formatId('MatMul', nodeId), order: order++}, classes: 'declareBefore'},
                {group: 'edges', data: {source: formatId('displacementInMemory', nodeId), target: formatId('Index1', nodeId), parent: formatId('MatMul', nodeId), order: order++, value: typeSizeMap[type]}, classes: 'constant'},
            ])
            break
    }

    //multiplication of the loaded values and storing them in the result matrix (equal for any optimization)
    cy.add([
        {group: 'nodes', data: {id: formatId('1', nodeId), parent: formatId('MatMul', nodeId), label: '1'}, classes: 'constant'},
        {group: 'nodes', data: {id: formatId('index', nodeId), parent: formatId('MatMul', nodeId), label: 'index'}, classes: 'input'},
        {group: 'nodes', data: {id: formatId(incomingEdges[0].data('source'), nodeId), label: '&' + incomingEdges[0].data('source'), parent: formatId('MatMul', nodeId)}, classes: 'input'},
        {group: 'nodes', data: {id: formatId(incomingEdges[1].data('source'), nodeId), label: '&' + incomingEdges[1].data('source'), parent: formatId('MatMul', nodeId)}, classes: 'input'},
        {group: 'nodes', data: {id: formatId(`MatMul_${nodeId}`, nodeId), label: '&Result', parent: formatId('MatMul', nodeId)}, classes: 'output'},


        {group: 'nodes', data: {id: formatId('Multiplication4', nodeId), parent: formatId('MatMul', nodeId), label: '*', opType: 'Multiplication'}, classes: 'operation'},
        {group: 'nodes', data: {id: formatId('Addition2', nodeId), parent: formatId('MatMul', nodeId), label: '+', opType: 'Addition'}, classes: 'operation'},
        {group: 'nodes', data: {id: formatId('Addition8', nodeId), parent: formatId('MatMul', nodeId), label: '+', opType: 'Addition'}, classes: 'operation'},
        {group: 'nodes', data: {id: formatId('Load0', nodeId), parent: formatId('MatMul', nodeId), label: 'Load', opType: 'Load'}, classes: 'operation'},
        {group: 'nodes', data: {id: formatId('Load1', nodeId), parent: formatId('MatMul', nodeId), label: 'Load', opType: 'Load'}, classes: 'operation'},
        {group: 'nodes', data: {id: formatId('Load2', nodeId), parent: formatId('MatMul', nodeId), label: 'Load', opType: 'Load'}, classes: 'operation'},
        {group: 'nodes', data: {id: formatId('Store', nodeId), parent: formatId('MatMul', nodeId), label: 'Store', opType: 'Store'}, classes: 'operation'},


        {group: 'edges', data: {source: formatId(incomingEdges[0].data('source'), nodeId), target: formatId('Load0', nodeId), parent: formatId('MatMul', nodeId), order: order++}, classes: 'input'},
        {group: 'edges', data: {source: formatId('Index0', nodeId), target: formatId('Load0', nodeId), parent: formatId('MatMul', nodeId), order: order++, opType: 'Multiplication'}, classes: 'operation'},

        {group: 'edges', data: {source: formatId(incomingEdges[1].data('source'), nodeId), target: formatId('Load1', nodeId), parent: formatId('MatMul', nodeId), order: order++}, classes: 'input'},
        {group: 'edges', data: {source: formatId('Index1', nodeId), target: formatId('Load1', nodeId), parent: formatId('MatMul', nodeId), order: order++, opType: 'Multiplication'}, classes: 'operation'},


        {group: 'edges', data: {source: formatId('Load0', nodeId), target: formatId('Multiplication4', nodeId), parent: formatId('MatMul', nodeId), order: order++, opType: 'Load'}, classes: 'operation'},
        {group: 'edges', data: {source: formatId('Load1', nodeId), target: formatId('Multiplication4', nodeId), parent: formatId('MatMul', nodeId), order: order++, opType: 'Load'}, classes: 'operation'},

        {group: 'edges', data: {source: formatId(`MatMul_${nodeId}`, nodeId), target: formatId('Load2', nodeId), parent: formatId('MatMul', nodeId), order: order++}, classes: 'input'},
        {group: 'edges', data: {source: formatId('IndexRes', nodeId), target: formatId('Load2', nodeId), parent: formatId('MatMul', nodeId), order: order++, opType: 'Multiplication'}, classes: 'operation'},
        {group: 'edges', data: {source: formatId('Load2', nodeId), target: formatId('Addition2', nodeId), parent: formatId('MatMul', nodeId), order: order++, opType: 'Load'}, classes: 'operation'},

        {group: 'edges', data: {source: formatId('Multiplication4', nodeId), target: formatId('Addition2', nodeId), parent: formatId('MatMul', nodeId), order: order++, opType: 'Multiplication'}, classes: 'operation'},

        {group: 'edges', data: {source: formatId('Addition2', nodeId), target: formatId('Store', nodeId), parent: formatId('MatMul', nodeId), order: order++, opType: 'Addition'}, classes: 'operation'},

        {group: 'edges', data: {source: formatId('IndexRes', nodeId), target: formatId('Store', nodeId), parent: formatId('MatMul', nodeId), order: order++, opType: 'Multiplication'}, classes: 'operation'},

        {group: 'edges', data: {source: formatId('Store', nodeId), target: formatId(`MatMul_${nodeId}`, nodeId), parent: formatId('MatMul', nodeId), order: order++, opType: 'Store'}, classes: 'operation result'},

        {group: 'edges', data: {source: formatId('index', nodeId), target: formatId('Addition8', nodeId), parent: formatId('MatMul', nodeId), order: order++}, classes: 'declareBefore index'},
        {group: 'edges', data: {source: formatId('1', nodeId), target: formatId('Addition8', nodeId), parent: formatId('MatMul', nodeId), order: order++, value: 1}, classes: 'constant'},
        {group: 'edges', data: {source: formatId('Addition8', nodeId), target: formatId('index', nodeId), parent: formatId('MatMul', nodeId), order: order++, opType: 'Addition'}, classes: 'operation variable'},


    ])

    switch (pattern) {
        case '000':
            cy.add([
                {group: 'nodes', data: {id: formatId('Addition4', nodeId), parent: formatId('MatMul', nodeId), label: '+', opType: 'Addition'}, classes: 'operation'},
                {group: 'nodes', data: {id: formatId('Addition5', nodeId), parent: formatId('MatMul', nodeId), label: '+', opType: 'Addition'}, classes: 'operation'},
                {group: 'nodes', data: {id: formatId('Addition6', nodeId), parent: formatId('MatMul', nodeId), label: '+', opType: 'Addition'}, classes: 'operation'},
                {group: 'nodes', data: {id: formatId('Addition7', nodeId), parent: formatId('MatMul', nodeId), label: '+', opType: 'Addition'}, classes: 'operation'},
                {group: 'nodes', data: {id: formatId('Multiplication7', nodeId), parent: formatId('MatMul', nodeId), label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: formatId('Multiplication8', nodeId), parent: formatId('MatMul', nodeId), label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: formatId('Multiplication9', nodeId), parent: formatId('MatMul', nodeId), label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: formatId('Multiplication10', nodeId), parent: formatId('MatMul', nodeId), label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: formatId('Multiplication11', nodeId), parent: formatId('MatMul', nodeId), label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: formatId('Multiplication12', nodeId), parent: formatId('MatMul', nodeId), label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: formatId('Equality0', nodeId), parent: formatId('MatMul', nodeId), label: '==', opType: 'Equality'}, classes: 'operation'},
                {group: 'nodes', data: {id: formatId('Equality1', nodeId), parent: formatId('MatMul', nodeId), label: '==', opType: 'Equality'}, classes: 'operation'},
                {group: 'nodes', data: {id: formatId('Not0', nodeId), parent: formatId('MatMul', nodeId), label: '!', opType: 'Not'}, classes: 'operation'},
                {group: 'nodes', data: {id: formatId('Not1', nodeId), parent: formatId('MatMul', nodeId), label: '!', opType: 'Not'}, classes: 'operation'},


                {group: 'edges', data: {source: formatId('k', nodeId), target: formatId('Equality0', nodeId), parent: formatId('MatMul', nodeId), order: order++}, classes: 'declareBefore'},
                {group: 'edges', data: {source: formatId('#rows1', nodeId), target: formatId('Equality0', nodeId), parent: formatId('MatMul', nodeId), order: order++, value: dimensions1[0].dimValue}, classes: 'constant'},

                {group: 'edges', data: {source: formatId('1', nodeId), target: formatId('Addition4', nodeId), parent: formatId('MatMul', nodeId), order: order++, value: 1}, classes: 'constant'},
                {group: 'edges', data: {source: formatId('j', nodeId), target: formatId('Addition4', nodeId), parent: formatId('MatMul', nodeId), order: order++}, classes: 'declareBefore'},

                {group: 'edges', data: {source: formatId('Equality0', nodeId), target: formatId('Multiplication7', nodeId), parent: formatId('MatMul', nodeId), order: order++, opType: 'Equality'}, classes: 'operation'},
                {group: 'edges', data: {source: formatId('Addition4', nodeId), target: formatId('Multiplication7', nodeId), parent: formatId('MatMul', nodeId), order: order++, opType: 'Addition'}, classes: 'operation'},

                {group: 'edges', data: {source: formatId('Equality0', nodeId), target: formatId('Not0', nodeId), parent: formatId('MatMul', nodeId), order: order++, opType: 'Equality'}, classes: 'operation'},

                {group: 'edges', data: {source: formatId('Not0', nodeId), target: formatId('Multiplication8', nodeId), parent: formatId('MatMul', nodeId), order: order++, opType: 'Not'}, classes: 'operation'},
                {group: 'edges', data: {source: formatId('j', nodeId), target: formatId('Multiplication8', nodeId), parent: formatId('MatMul', nodeId), order: order++}, classes: 'declareBefore'},

                {group: 'edges', data: {source: formatId('Multiplication7', nodeId), target: formatId('Addition5', nodeId), parent: formatId('MatMul', nodeId), order: order++, opType: 'Multiplication'}, classes: 'operation'},
                {group: 'edges', data: {source: formatId('Multiplication8', nodeId), target: formatId('Addition5', nodeId), parent: formatId('MatMul', nodeId), order: order++, opType: 'Multiplication'}, classes: 'operation'},

                {group: 'edges', data: {source: formatId('Addition5', nodeId), target: formatId('j', nodeId), parent: formatId('MatMul', nodeId), order: order++, opType: 'Addition'}, classes: 'operation variable'},
                {group: 'edges', data: {source: formatId('Not0', nodeId), target: formatId('Multiplication9', nodeId), parent: formatId('MatMul', nodeId), order: order++, opType: 'Not'}, classes: 'operation'},
                {group: 'edges', data: {source: formatId('k', nodeId), target: formatId('Multiplication9', nodeId), parent: formatId('MatMul', nodeId), order: order++}, classes: 'declareBefore'},

                {group: 'edges', data: {source: formatId('Multiplication9', nodeId), target: formatId('k', nodeId), parent: formatId('MatMul', nodeId), order: order++, opType: 'Multiplication'}, classes: 'operation variable'},

                {group: 'edges', data: {source: formatId('j', nodeId), target: formatId('Equality1', nodeId), parent: formatId('MatMul', nodeId), order: order++}, classes: 'declareBefore'},
                {group: 'edges', data: {source: formatId('#columns1', nodeId), target: formatId('Equality1', nodeId), parent: formatId('MatMul', nodeId), order: order++, value: dimensions1[1].dimValue}, classes: 'constant'},

                {group: 'edges', data: {source: formatId('1', nodeId), target: formatId('Addition6', nodeId), parent: formatId('MatMul', nodeId), order: order++, value: 1}, classes: 'constant'},
                {group: 'edges', data: {source: formatId('i', nodeId), target: formatId('Addition6', nodeId), parent: formatId('MatMul', nodeId), order: order++}, classes: 'declareBefore'},

                {group: 'edges', data: {source: formatId('Equality1', nodeId), target: formatId('Multiplication10', nodeId), parent: formatId('MatMul', nodeId), order: order++, opType: 'Equality'}, classes: 'operation'},
                {group: 'edges', data: {source: formatId('Addition6', nodeId), target: formatId('Multiplication10', nodeId), parent: formatId('MatMul', nodeId), order: order++, opType: 'Addition'}, classes: 'operation'},

                {group: 'edges', data: {source: formatId('Equality1', nodeId), target: formatId('Not1', nodeId), parent: formatId('MatMul', nodeId), order: order++, opType: 'Equality'}, classes: 'operation'},

                {group: 'edges', data: {source: formatId('Not1', nodeId), target: formatId('Multiplication11', nodeId), parent: formatId('MatMul', nodeId), order: order++, opType: 'Not'}, classes: 'operation'},
                {group: 'edges', data: {source: formatId('i', nodeId), target: formatId('Multiplication11', nodeId), parent: formatId('MatMul', nodeId), order: order++}, classes: 'declareBefore'},

                {group: 'edges', data: {source: formatId('Multiplication10', nodeId), target: formatId('Addition7', nodeId), parent: formatId('MatMul', nodeId), order: order++, opType: 'Multiplication'}, classes: 'operation'},
                {group: 'edges', data: {source: formatId('Multiplication11', nodeId), target: formatId('Addition7', nodeId), parent: formatId('MatMul', nodeId), order: order++, opType: 'Multiplication'}, classes: 'operation'},

                {group: 'edges', data: {source: formatId('Addition7', nodeId), target: formatId('i', nodeId), parent: formatId('MatMul', nodeId), order: order++, opType: 'Addition'}, classes: 'operation variable'},

                {group: 'edges', data: {source: formatId('Not1', nodeId), target: formatId('Multiplication12', nodeId), parent: formatId('MatMul', nodeId), order: order++, opType: 'Not'}, classes: 'operation'},
                {group: 'edges', data: {source: formatId('j', nodeId), target: formatId('Multiplication12', nodeId), parent: formatId('MatMul', nodeId), order: order++}, classes: 'declareBefore'},

                {group: 'edges', data: {source: formatId('Multiplication12', nodeId), target: formatId('j', nodeId), parent: formatId('MatMul', nodeId), order: order++, opType: 'Multiplication'}, classes: 'operation variable'}
            ])
            break
        case '100':
        case '001':
        case '010':
            cy.add([
                {group: 'nodes', data: {id: formatId('Addition4', nodeId), parent: formatId('MatMul', nodeId), label: '+', opType: 'Addition'}, classes: 'operation'},
                {group: 'nodes', data: {id: formatId('Addition5', nodeId), parent: formatId('MatMul', nodeId), label: '+', opType: 'Addition'}, classes: 'operation'},
                {group: 'nodes', data: {id: formatId('Multiplication7', nodeId), parent: formatId('MatMul', nodeId), label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: formatId('Multiplication8', nodeId), parent: formatId('MatMul', nodeId), label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: formatId('Multiplication9', nodeId), parent: formatId('MatMul', nodeId), label: '*', opType: 'Multiplication'}, classes: 'operation'},
                {group: 'nodes', data: {id: formatId('Equality0', nodeId), parent: formatId('MatMul', nodeId), label: '==', opType: 'Equality'}, classes: 'operation'},
                {group: 'nodes', data: {id: formatId('Not0', nodeId), parent: formatId('MatMul', nodeId), label: '!', opType: 'Not'}, classes: 'operation'},

                {group: 'edges', data: {source: formatId('k', nodeId), target: formatId('Equality0', nodeId), parent: formatId('MatMul', nodeId), order: order++}, classes: 'declareBefore'},
                {group: 'edges', data: {source: formatId('#rows1', nodeId), target: formatId('Equality0', nodeId), parent: formatId('MatMul', nodeId), order: order++, value: dimensions1[0].dimValue}, classes: 'constant'},

                {group: 'edges', data: {source: formatId('1', nodeId), target: formatId('Addition4', nodeId), parent: formatId('MatMul', nodeId), order: order++, value: 1}, classes: 'constant'},
                {group: 'edges', data: {source: formatId('j', nodeId), target: formatId('Addition4', nodeId), parent: formatId('MatMul', nodeId), order: order++}, classes: 'declareBefore'},

                {group: 'edges', data: {source: formatId('Equality0', nodeId), target: formatId('Multiplication7', nodeId), parent: formatId('MatMul', nodeId), order: order++, opType: 'Equality'}, classes: 'operation'},
                {group: 'edges', data: {source: formatId('Addition4', nodeId), target: formatId('Multiplication7', nodeId), parent: formatId('MatMul', nodeId), order: order++, opType: 'Addition'}, classes: 'operation'},

                {group: 'edges', data: {source: formatId('Equality0', nodeId), target: formatId('Not0', nodeId), parent: formatId('MatMul', nodeId), order: order++, opType: 'Equality'}, classes: 'operation'},

                {group: 'edges', data: {source: formatId('Not0', nodeId), target: formatId('Multiplication8', nodeId), parent: formatId('MatMul', nodeId), order: order++, opType: 'Not'}, classes: 'operation'},
                {group: 'edges', data: {source: formatId('j', nodeId), target: formatId('Multiplication8', nodeId), parent: formatId('MatMul', nodeId), order: order++}, classes: 'declareBefore'},

                {group: 'edges', data: {source: formatId('Multiplication7', nodeId), target: formatId('Addition5', nodeId), parent: formatId('MatMul', nodeId), order: order++, opType: 'Multiplication'}, classes: 'operation'},
                {group: 'edges', data: {source: formatId('Multiplication8', nodeId), target: formatId('Addition5', nodeId), parent: formatId('MatMul', nodeId), order: order++, opType: 'Multiplication'}, classes: 'operation'},

                {group: 'edges', data: {source: formatId('Addition5', nodeId), target: formatId('j', nodeId), parent: formatId('MatMul', nodeId), order: order++, opType: 'Addition'}, classes: 'operation variable'},
                {group: 'edges', data: {source: formatId('Not0', nodeId), target: formatId('Multiplication9', nodeId), parent: formatId('MatMul', nodeId), order: order++, opType: 'Not'}, classes: 'operation'},
                {group: 'edges', data: {source: formatId('k', nodeId), target: formatId('Multiplication9', nodeId), parent: formatId('MatMul', nodeId), order: order++}, classes: 'declareBefore'},

                {group: 'edges', data: {source: formatId('Multiplication9', nodeId), target: formatId('k', nodeId), parent: formatId('MatMul', nodeId), order: order++, opType: 'Multiplication'}, classes: 'operation variable'}

            ])
            break
    }
    //checking of indexes


    outgoingEdges.forEach(edge => {
        cy.add({group: 'edges', data: {source: formatId('MatMul', nodeId) , target: edge.data('target'), dims: edge.data('dims'), elemType: edge.data('elemType'), order: edgeOrder.value++}, classes: 'compound variable'})
    })
    cy.remove(node)
}

export function transformOpps(cy) {
    let edgeOrder = {value : 0}
    cy.nodes('.operation').forEach(node => {
        if (node.data('opType') === 'MatMul') {
            transformMatMul(node, cy, edgeOrder)
        }
        else {
            transformSimpleLoopOperations(node, cy, edgeOrder)
        }
    })
}
