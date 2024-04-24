function transformAdd(node, cy) {
    const incomingEdges = node.incomers('edge')
    const outgoingEdges = node.outgoers('edge')
    const dimensions = incomingEdges[0].data('dims')

    if (dimensions[0].dimValue === '1' && dimensions.length === 1) {
        cy.add([
            {group: 'nodes', data: {id: node.data('id')+ ' Addition', label: '+', opType: 'Addition'}, classes: 'operation'},
            {group: 'edges', data: {source: incomingEdges[0].data('source'), target: node.data('id') + ' Addition', dims: incomingEdges[0].data('dims'), elemType: incomingEdges[0].data('elemType')}},
            {group: 'edges', data: {source: incomingEdges[1].data('source'), target: node.data('id') + ' Addition', dims: incomingEdges[1].data('dims'), elemType: incomingEdges[1].data('elemType')}}
        ])
        outgoingEdges.forEach(edge => {
            cy.add({group: 'edges', data: {source: node.data('id') + ' Addition', target: edge.data('target'), dims: edge.data('dims'), elemType: edge.data('elemType')}})
        })
    }
    else {
        let numberOfIterations = dimensions.reduce((total, dim) => total + parseInt(dim.dimValue), 0)
        cy.add([
            {group: 'nodes', data: {id: node.data('id') + 'LoopIterations', label: '# of loop iterations', value: numberOfIterations.toString()}, classes: 'input'},
            {group: 'nodes', data: {id: node.data('id') + 'Add', label: 'Add', opType: 'Add'}},
            {group: 'nodes', data: {id: node.data('id') + 'index', parent: node.data('id') +'Add', label: 'index'}, classes: 'input'},
            {group: 'nodes', data: {id: node.data('id') + 'displacementInMemory', parent: node.data('id') + 'Add', label: 'Displacement In Memory', value: 4}, classes: 'input'},
            {group: 'nodes', data: {id: node.data('id') + incomingEdges[0].data('source'), label: '&' + incomingEdges[0].data('source'), parent: node.data('id') + 'Add'}, classes: 'input'},
            {group: 'nodes', data: {id: node.data('id') + incomingEdges[1].data('source'), label: '&' + incomingEdges[1].data('source'), parent: node.data('id') + 'Add'}, classes: 'input'},
            {group: 'nodes', data: {id: node.data('id') + 'res', label: '&Result', parent: node.data('id') + 'Add'}, classes: 'output'},
            {group: 'nodes', data: {id: node.data('id') + 'Multiplication', parent: node.data('id') + 'Add', label: '*', opType: 'Multiplication'}, classes: 'operation'},
            {group: 'nodes', data: {id: node.data('id') + 'Load0', parent: node.data('id') + 'Add', label: 'Load', opType: 'Load'}, classes: 'operation'},
            {group: 'nodes', data: {id: node.data('id') + 'Load1', parent: node.data('id') + 'Add', label: 'Load', opType: 'Load'}, classes: 'operation'},
            {group: 'nodes', data: {id: node.data('id') + 'Addition', parent: node.data('id') + 'Add', label: '+', opType: 'Addition'}, classes: 'operation'},
            {group: 'nodes', data: {id: node.data('id') + 'Store', parent: node.data('id') + 'Add', label: 'Store', opType: 'Store'}, classes: 'operation'},

            {group: 'edges', data: {source: incomingEdges[0].data('source'), target: node.data('id') + 'Add', dims: incomingEdges[0].data('dims'), elemType: incomingEdges[0].data('elemType')}},
            {group: 'edges', data: {source: incomingEdges[1].data('source'), target: node.data('id') + 'Add', dims: incomingEdges[1].data('dims'), elemType: incomingEdges[1].data('elemType')}},
            {group: 'edges', data: {source: node.data('id') + 'LoopIterations', target: node.data('id') + 'Add'}},
            {group: 'edges', data: {source: node.data('id') + 'index', target: node.data('id') + 'Multiplication'}},
            {group: 'edges', data: {source: node.data('id') + 'displacementInMemory', target: node.data('id') + 'Multiplication'}},
            {group: 'edges', data: {source: node.data('id') + 'Multiplication', target: node.data('id') + 'Load0'}},
            {group: 'edges', data: {source: node.data('id') + 'Multiplication', target: node.data('id') + 'Load1'}},
            {group: 'edges', data: {source: node.data('id') + incomingEdges[0].data('source'), target: node.data('id') + 'Load0'}},
            {group: 'edges', data: {source: node.data('id') + incomingEdges[1].data('source'), target: node.data('id') + 'Load1'}},
            {group: 'edges', data: {source: node.data('id') + 'Load0', target: node.data('id') + 'Addition'}},
            {group: 'edges', data: {source: node.data('id') + 'Load1', target: node.data('id') + 'Addition'}},
            {group: 'edges', data: {source: node.data('id') + 'Addition', target: node.data('id') + 'Store'}},
            {group: 'edges', data: {source: node.data('id') + 'Multiplication', target: node.data('id') + 'Store'}},
            {group: 'edges', data: {source: node.data('id') + 'Store', target: node.data('id') + 'res'}}
        ])
        outgoingEdges.forEach(edge => {
            cy.add({group: 'edges', data: {source: node.data('id') + 'Add', target: edge.data('target'), dims: edge.data('dims'), elemType: edge.data('elemType')}})
        })
    }
    cy.remove(node)
}

function transformMatMul(node, cy) {
    const incomingEdges = node.incomers('edge')
    const outgoingEdges = node.outgoers('edge')
    const dimensions0 = incomingEdges[0].data('dims')
    const dimensions1 = incomingEdges[1].data('dims')
    console.log(incomingEdges[1].data())
    let numberOfIterations = dimensions0[0].dimValue * dimensions1[0].dimValue * dimensions1[1].dimValue

    //main body
    cy.add([
        {group: 'nodes', data: {id: node.data('id') + 'LoopIterations', label: '# of loop iterations', value: numberOfIterations.toString()}, classes: 'input'},
        {group: 'nodes', data: {id: node.data('id') + 'MatMul', label: 'MatMul', opType: 'MatMul'}},
        {group: 'nodes', data: {id: node.data('id') + 'i', parent: node.data('id') +'MatMul', label: 'i'}, classes: 'input'},
        {group: 'nodes', data: {id: node.data('id') + 'j', parent: node.data('id') +'MatMul', label: 'j'}, classes: 'input'},
        {group: 'nodes', data: {id: node.data('id') + 'k', parent: node.data('id') +'MatMul', label: 'k'}, classes: 'input'},
        {group: 'nodes', data: {id: node.data('id') + '#rows0', parent: node.data('id') +'MatMul', label: '# of rows of ' + incomingEdges[0].data('source'), value: dimensions0[0].dimValue}, classes: 'input'},
        {group: 'nodes', data: {id: node.data('id') + '#columns1', parent: node.data('id') +'MatMul', label: '# of columns of ' + incomingEdges[1].data('source'), value: dimensions1[1].dimValue}, classes: 'input'},
        {group: 'nodes', data: {id: node.data('id') + '#rows1', parent: node.data('id') +'MatMul', label: '# of rows of ' + incomingEdges[1].data('source'), value: dimensions1[0].dimValue}, classes: 'input'},
        {group: 'nodes', data: {id: node.data('id') + 'displacementInMemory', parent: node.data('id') + 'MatMul', label: 'Displacement In Memory', value: 4}, classes: 'input'},
        {group: 'nodes', data: {id: node.data('id') + incomingEdges[0].data('source'), label: '&' + incomingEdges[0].data('source'), parent: node.data('id') + 'MatMul'}, classes: 'input'},
        {group: 'nodes', data: {id: node.data('id') + incomingEdges[1].data('source'), label: '&' + incomingEdges[1].data('source'), parent: node.data('id') + 'MatMul'}, classes: 'input'},
        {group: 'nodes', data: {id: node.data('id') + 'res', label: '&Result', parent: node.data('id') + 'MatMul'}, classes: 'output'},

        {group: 'nodes', data: {id: node.data('id') + 'Multiplication0', parent: node.data('id') + 'MatMul', label: '*', opType: 'Multiplication'}, classes: 'operation'},
        {group: 'nodes', data: {id: node.data('id') + 'Multiplication1', parent: node.data('id') + 'MatMul', label: '*', opType: 'Multiplication'}, classes: 'operation'},
        {group: 'nodes', data: {id: node.data('id') + 'Multiplication2', parent: node.data('id') + 'MatMul', label: '*', opType: 'Multiplication'}, classes: 'operation'},
        {group: 'nodes', data: {id: node.data('id') + 'Multiplication3', parent: node.data('id') + 'MatMul', label: '*', opType: 'Multiplication'}, classes: 'operation'},
        {group: 'nodes', data: {id: node.data('id') + 'Multiplication4', parent: node.data('id') + 'MatMul', label: '*', opType: 'Multiplication'}, classes: 'operation'},
        {group: 'nodes', data: {id: node.data('id') + 'Multiplication5', parent: node.data('id') + 'MatMul', label: '*', opType: 'Multiplication'}, classes: 'operation'},
        {group: 'nodes', data: {id: node.data('id') + 'Multiplication6', parent: node.data('id') + 'MatMul', label: '*', opType: 'Multiplication'}, classes: 'operation'},
        {group: 'nodes', data: {id: node.data('id') + 'Addition0', parent: node.data('id') + 'MatMul', label: '+', opType: 'Addition'}, classes: 'operation'},
        {group: 'nodes', data: {id: node.data('id') + 'Addition1', parent: node.data('id') + 'MatMul', label: '+', opType: 'Addition'}, classes: 'operation'},
        {group: 'nodes', data: {id: node.data('id') + 'Addition2', parent: node.data('id') + 'MatMul', label: '+', opType: 'Addition'}, classes: 'operation'},
        {group: 'nodes', data: {id: node.data('id') + 'Addition3', parent: node.data('id') + 'MatMul', label: '+', opType: 'Addition'}, classes: 'operation'},
        {group: 'nodes', data: {id: node.data('id') + 'Load0', parent: node.data('id') + 'MatMul', label: 'Load', opType: 'Load'}, classes: 'operation'},
        {group: 'nodes', data: {id: node.data('id') + 'Load1', parent: node.data('id') + 'MatMul', label: 'Load', opType: 'Load'}, classes: 'operation'},
        {group: 'nodes', data: {id: node.data('id') + 'Store', parent: node.data('id') + 'MatMul', label: 'Store', opType: 'Store'}, classes: 'operation'},

        {group: 'edges', data: {source: incomingEdges[0].data('source'), target: node.data('id') + 'MatMul', dims: incomingEdges[0].data('dims'), elemType: incomingEdges[0].data('elemType')}},
        {group: 'edges', data: {source: incomingEdges[1].data('source'), target: node.data('id') + 'MatMul', dims: incomingEdges[0].data('dims'), elemType: incomingEdges[0].data('elemType')}},
        {group: 'edges', data: {source: node.data('id') + 'LoopIterations', target: node.data('id') + 'MatMul'}},
        {group: 'edges', data: {source: node.data('id') + 'i', target: node.data('id') + 'Multiplication0'}},
        {group: 'edges', data: {source: node.data('id') + '#rows1', target: node.data('id') + 'Multiplication0'}},
        {group: 'edges', data: {source: node.data('id') + 'k', target: node.data('id') + 'Multiplication1'}},
        {group: 'edges', data: {source: node.data('id') + '#columns1', target: node.data('id') + 'Multiplication1'}},
        {group: 'edges', data: {source: node.data('id') + 'Multiplication0', target: node.data('id') + 'Addition0'}},
        {group: 'edges', data: {source: node.data('id') + 'k', target: node.data('id') + 'Addition0'}},
        {group: 'edges', data: {source: node.data('id') + 'j', target: node.data('id') + 'Addition1'}},
        {group: 'edges', data: {source: node.data('id') + 'Multiplication1', target: node.data('id') + 'Addition1'}},
        {group: 'edges', data: {source: node.data('id') + 'Addition0', target: node.data('id') + 'Multiplication2'}},
        {group: 'edges', data: {source: node.data('id') + 'displacementInMemory', target: node.data('id') + 'Multiplication2'}},
        {group: 'edges', data: {source: node.data('id') + 'Addition1', target: node.data('id') + 'Multiplication3'}},
        {group: 'edges', data: {source: node.data('id') + 'displacementInMemory', target: node.data('id') + 'Multiplication3'}},
        {group: 'edges', data: {source: node.data('id') + 'Multiplication2', target: node.data('id') + 'Load0'}},
        {group: 'edges', data: {source: node.data('id') + 'Multiplication3', target: node.data('id') + 'Load1'}},
        {group: 'edges', data: {source: node.data('id') + incomingEdges[0].data('source'), target: node.data('id') + 'Load0'}},
        {group: 'edges', data: {source: node.data('id') + incomingEdges[1].data('source'), target: node.data('id') + 'Load1'}},
        {group: 'edges', data: {source: node.data('id') + 'Load0', target: node.data('id') + 'Multiplication4'}},
        {group: 'edges', data: {source: node.data('id') + 'Load1', target: node.data('id') + 'Multiplication4'}},
        {group: 'edges', data: {source: node.data('id') + 'Multiplication4', target: node.data('id') + 'Addition2'}},
        {group: 'edges', data: {source: node.data('id') + 'res', target: node.data('id') + 'Addition2'}},
        {group: 'edges', data: {source: node.data('id') + 'i', target: node.data('id') + 'Multiplication5'}},
        {group: 'edges', data: {source: node.data('id') + '#columns1', target: node.data('id') + 'Multiplication5'}},
        {group: 'edges', data: {source: node.data('id') + 'j', target: node.data('id') + 'Addition3'}},
        {group: 'edges', data: {source: node.data('id') + 'Multiplication5', target: node.data('id') + 'Addition3'}},
        {group: 'edges', data: {source: node.data('id') + 'Addition3', target: node.data('id') + 'Multiplication6'}},
        {group: 'edges', data: {source: node.data('id') + 'displacementInMemory', target: node.data('id') + 'Multiplication6'}},
        {group: 'edges', data: {source: node.data('id') + 'Addition2', target: node.data('id') + 'Store'}},
        {group: 'edges', data: {source: node.data('id') + 'Multiplication6', target: node.data('id') + 'Store'}},
        {group: 'edges', data: {source: node.data('id') + 'Store', target: node.data('id') + 'res'}},

    ])

    cy.add([
        {group: 'nodes', data: {id: node.data('id') + '1', parent: node.data('id') + 'MatMul', label: '1'}, classes: 'constant'},


        {group: 'nodes', data: {id: node.data('id') + 'Addition4', parent: node.data('id') + 'MatMul', label: '==', opType: 'Addition'}, classes: 'operation'},
        {group: 'nodes', data: {id: node.data('id') + 'Addition5', parent: node.data('id') + 'MatMul', label: '==', opType: 'Addition'}, classes: 'operation'},
        {group: 'nodes', data: {id: node.data('id') + 'Addition6', parent: node.data('id') + 'MatMul', label: '==', opType: 'Addition'}, classes: 'operation'},
        {group: 'nodes', data: {id: node.data('id') + 'Addition7', parent: node.data('id') + 'MatMul', label: '==', opType: 'Addition'}, classes: 'operation'},
        {group: 'nodes', data: {id: node.data('id') + 'Multiplication7', parent: node.data('id') + 'MatMul', label: '*', opType: 'Multiplication'}, classes: 'operation'},
        {group: 'nodes', data: {id: node.data('id') + 'Multiplication8', parent: node.data('id') + 'MatMul', label: '*', opType: 'Multiplication'}, classes: 'operation'},
        {group: 'nodes', data: {id: node.data('id') + 'Multiplication9', parent: node.data('id') + 'MatMul', label: '*', opType: 'Multiplication'}, classes: 'operation'},
        {group: 'nodes', data: {id: node.data('id') + 'Multiplication10', parent: node.data('id') + 'MatMul', label: '*', opType: 'Multiplication'}, classes: 'operation'},
        {group: 'nodes', data: {id: node.data('id') + 'Multiplication11', parent: node.data('id') + 'MatMul', label: '*', opType: 'Multiplication'}, classes: 'operation'},
        {group: 'nodes', data: {id: node.data('id') + 'Multiplication12', parent: node.data('id') + 'MatMul', label: '*', opType: 'Multiplication'}, classes: 'operation'},
        {group: 'nodes', data: {id: node.data('id') + 'EqualTo0', parent: node.data('id') + 'MatMul', label: '==', opType: 'EqualTo'}, classes: 'operation'},
        {group: 'nodes', data: {id: node.data('id') + 'EqualTo1', parent: node.data('id') + 'MatMul', label: '==', opType: 'EqualTo'}, classes: 'operation'},
        {group: 'nodes', data: {id: node.data('id') + 'Not0', parent: node.data('id') + 'MatMul', label: '!', opType: 'Not'}, classes: 'operation'},
        {group: 'nodes', data: {id: node.data('id') + 'Not1', parent: node.data('id') + 'MatMul', label: '!', opType: 'Not'}, classes: 'operation'},

        {group: 'edges', data: {source: node.data('id') + 'k', target: node.data('id') + 'EqualTo0'}},
        {group: 'edges', data: {source: node.data('id') + '#rows1', target: node.data('id') + 'EqualTo0'}},
        {group: 'edges', data: {source: node.data('id') + '1', target: node.data('id') + 'Addition4'}},
        {group: 'edges', data: {source: node.data('id') + 'j', target: node.data('id') + 'Addition4'}},
        {group: 'edges', data: {source: node.data('id') + 'EqualTo0', target: node.data('id') + 'Multiplication7'}},
        {group: 'edges', data: {source: node.data('id') + 'Addition4', target: node.data('id') + 'Multiplication7'}},
        {group: 'edges', data: {source: node.data('id') + 'EqualTo0', target: node.data('id') + 'Not0'}},
        {group: 'edges', data: {source: node.data('id') + 'Not0', target: node.data('id') + 'Multiplication8'}},
        {group: 'edges', data: {source: node.data('id') + 'j', target: node.data('id') + 'Multiplication8'}},
        {group: 'edges', data: {source: node.data('id') + 'Multiplication7', target: node.data('id') + 'Addition5'}},
        {group: 'edges', data: {source: node.data('id') + 'Multiplication8', target: node.data('id') + 'Addition5'}},
        {group: 'edges', data: {source: node.data('id') + 'Addition5', target: node.data('id') + 'j'}},
        {group: 'edges', data: {source: node.data('id') + 'Not0', target: node.data('id') + 'Multiplication9'}},
        {group: 'edges', data: {source: node.data('id') + 'k', target: node.data('id') + 'Multiplication9'}},
        {group: 'edges', data: {source: node.data('id') + 'Multiplication9', target: node.data('id') + 'k'}},

        {group: 'edges', data: {source: node.data('id') + 'j', target: node.data('id') + 'EqualTo1'}},
        {group: 'edges', data: {source: node.data('id') + '#columns1', target: node.data('id') + 'EqualTo1'}},
        {group: 'edges', data: {source: node.data('id') + '1', target: node.data('id') + 'Addition6'}},
        {group: 'edges', data: {source: node.data('id') + 'i', target: node.data('id') + 'Addition6'}},
        {group: 'edges', data: {source: node.data('id') + 'EqualTo1', target: node.data('id') + 'Multiplication10'}},
        {group: 'edges', data: {source: node.data('id') + 'Addition6', target: node.data('id') + 'Multiplication10'}},
        {group: 'edges', data: {source: node.data('id') + 'EqualTo1', target: node.data('id') + 'Not1'}},
        {group: 'edges', data: {source: node.data('id') + 'Not1', target: node.data('id') + 'Multiplication11'}},
        {group: 'edges', data: {source: node.data('id') + 'i', target: node.data('id') + 'Multiplication11'}},
        {group: 'edges', data: {source: node.data('id') + 'Multiplication10', target: node.data('id') + 'Addition7'}},
        {group: 'edges', data: {source: node.data('id') + 'Multiplication11', target: node.data('id') + 'Addition7'}},
        {group: 'edges', data: {source: node.data('id') + 'Addition7', target: node.data('id') + 'i'}},
        {group: 'edges', data: {source: node.data('id') + 'Not1', target: node.data('id') + 'Multiplication12'}},
        {group: 'edges', data: {source: node.data('id') + 'j', target: node.data('id') + 'Multiplication12'}},
        {group: 'edges', data: {source: node.data('id') + 'Multiplication12', target: node.data('id') + 'j'}}
    ])

    outgoingEdges.forEach(edge => {
        cy.add({group: 'edges', data: {source: node.data('id') + 'MatMul', target: edge.data('target'), dims: edge.data('dims'), elemType: edge.data('elemType')}})
    })

    cy.remove(node)
}

export function transformOpps(cy) {
    //ainda falta determinar o displacement em memoria de acordo como o tipo de input
    //verificar o matmul antes de prosseguir com otimizações
    cy.nodes('.operation').forEach(node => {
        if (node.data('opType') === 'Add'){
            transformAdd(node, cy)
        }
        else if (node.data('opType') === 'MatMul') {
            transformMatMul(node, cy)
        }
    })
}
