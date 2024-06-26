function sortByOrder(edge1, edge2) {

    let order1 = edge1.data('order');
    let order2 = edge2.data('order');

    if (order1 < order2) {
        return -1
    } else if (order1 > order2) {
        return 1
    } else {
        return 0
    }
}

function handleOpperation(edge, variables, operations, code) {

    const source = edge.data('source')
    const target = edge.data('target')

    switch (edge.data('opType')) {
        case 'Addition':
            if (!variables[source]) {
                variables[source] = `${variables[operations[source][0]]} + ${variables[operations[source][1]]}`
            }
            break
        case 'Subtraction':
            if (!variables[source]) {
                variables[source] = `${variables[operations[source][0]]} - ${variables[operations[source][1]]}`
            }
            break
        case 'Load':
            if (!variables[source]) {
                variables[source] = `${variables[operations[source][0]]}[${variables[operations[source][1]]}]`
            }
            break
        case 'Multiplication':
            if (!variables[source]) {
                variables[source] = `${variables[operations[source][0]]} * ${variables[operations[source][1]]}`
            }
            break
        case 'Store':
            if (!variables[source]) {
                variables[source] = `${target}[${variables[operations[source][1]]}] = ${variables[operations[source][0]]}`
                code.content += `       ${target}[${variables[operations[source][1]]}] = ${variables[operations[source][0]]}\n`
            }
            break
        case 'Equality':
            if (!variables[source]) {
                variables[source] = `${variables[operations[source][0]]} === ${variables[operations[source][1]]}`
            }
            if (edge.hasClass('variable')) {
                code.content += `   ${target} = ${variables[source]}\n`
            }
            break
        case 'Not':
            if (!variables[source]) {
                variables[source] = `!${variables[operations[source][0]]}`
            }
            break

    }
}

function handleEdge(edge, variables, operations, code, index) {
    const edgeClass = edge.classes()[0]
    let source = edge.data('source');
    let target = edge.data('target');

    switch (edgeClass) {
        case 'operation':
            handleOpperation(edge, variables, operations, code)
            if (edge.classes()[1]) {
                code.content += `       ${target} = ${variables[source]}\n`
            }
            break
        case 'index':
            if (index.id === "") {
                index.id = source
                variables[source] = index.id
                code.content = `  let ${index.id} = 0\n` + code.content
            }

            break
        case 'constant':
            if (!variables[source]) {
                variables[source] = `${edge.data('value')}`;
            }
            break
        case 'input':
            variables[source] = source.substring(1)
            break
    }

    if (operations[target]) {
        operations[target].push(edge.data('source'));
    } else {
        operations[target] = [edge.data('source')];
    }

}

function handleCompoundNode(compoundNode, cy, code) {
    //variables used in the loop
    let inputs = compoundNode.incomers('node')

    //the number of loop iterations
    let loopIterations = inputs.filter(node => node.data('label') === '# of loop iterations');

    //edges inside the loop
    let edgesInsideCompound = cy.edges().filter(edge => edge.data('parent') === compoundNode.data('id')).sort(sortByOrder)

    //index variable's name
    let index = {id : ""}

    let loopCode = {content : `   while (${index.id} < ${loopIterations.data('value')}) {\n`}

    let operations = {}
    let variables= {}

    edgesInsideCompound.forEach(edge => {
        handleEdge(edge, variables, operations, loopCode, index)
    })


    loopCode.content += '   }\n}'
    code.content += loopCode.content

}

//if the variable isn't declared, then I should declare it
//that only happens for intermediate variables
export function generateCode(cy, data) {
    let declaredVariables = []
    let code = {content: `function ${data.graph.name}(`}
    data.graph.input.forEach(input => {code.content += `${input.name}, `})
    data.graph.output.forEach(output => declaredVariables.push(output.name))
    code.content = code.content.slice(0, -2) +') {\n'

    let graphOperations = {}

    let edges = cy.edges().filter(edge => !edge.data('parent')).sort(sortByOrder)
    edges.forEach(edge => {
        let source = edge.data('source');
        let target = edge.data('target');
    })

    let compoundNodes = []
    cy.nodes().forEach(node => {
        if (node.isParent()) compoundNodes.push(node)
    })
    compoundNodes.forEach(compoundNode => {
        handleCompoundNode(compoundNode, cy, code)
    })
    console.log(code.content)
}
