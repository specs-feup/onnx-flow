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

function handleEdge() {

}


export function generateCode(cy, data) {
    let code = `function ${data.graph.name}(`
    data.graph.input.forEach(input => {code += `${input.name}, `})
    code = code.slice(0, -2) +')\n'

    let compoundNodes = []
    cy.nodes().forEach(node => {
        if (node.isParent()) compoundNodes.push(node)
    })
    compoundNodes.forEach(compoundNode => {
        //variables used in the loop
        let inputs = compoundNode.incomers('node')

        //the number of loop iterations
        let loopIterations = inputs.filter(node => node.data('label') === '# of loop iterations');

        //edges inside the loop
        let edgesInsideCompound = cy.edges().filter(edge => edge.data('parent') === compoundNode.data('id')).sort(sortByOrder)

        //index variable's name
        let indexId = ""

        let declareBeforeLoop = ``
        let loopCode = ``

        let operations = {}
        let variables= {}

        edgesInsideCompound.forEach(edge => {


            const edgeClass = edge.classes()[0]
            console.log(edgeClass)
            let source = edge.data('source');
            let target = edge.data('target');

            switch (edgeClass) {
                case 'operation':
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
                                loopCode += `       ${target}[${variables[operations[source][1]]}] = ${variables[operations[source][0]]}\n`
                            }
                            break
                        case 'Equality':
                            if (!variables[source]) {
                                variables[source] = `${variables[operations[source][0]]} === ${variables[operations[source][1]]}`
                            }
                            if (edge.classes()[1]) {
                                loopCode += `   ${target} = ${variables[source]}\n`
                            }
                            break
                        case 'Not':
                            if (!variables[source]) {
                                variables[source] = `!${variables[operations[source][0]]}`
                            }
                            break

                    }
                    if (edge.classes()[1]) {
                        loopCode += `       ${target} = ${variables[source]}\n`
                    }
                    break
                case 'index':
                    if (indexId === "") {
                        indexId = source
                        variables[source] = indexId
                        declareBeforeLoop += `  let ${indexId} = 0\n`
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

        })

        //initiate the loop
        code += declareBeforeLoop
        code += `   while (${indexId} < ${loopIterations.data('value')}) {\n`
        code+= loopCode
        code += '   }\n'
        code += '}'
        console.log(code)


        //adicionar ordem para cada 1 dos edges adicionados com uma variavel order e++ a seguir
        //depois percorrer, ver se a operação ja foi declarada e adicionar variavel.
        // se a operação esta completa (tem todos os inputs) realizamos quando é chamada?
        // declarar variavel resultado antes de cada loop (ainda tenho de guardar a informação das variaveis nos edges maybe)
        //
        //
    })

}
