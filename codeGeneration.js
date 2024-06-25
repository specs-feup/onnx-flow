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
    let code = ""
    /*let code = `
    function ${data.name}(A, B) {
    let result = [];
    for (let i = 0; i < A.length; i++) {
    result[i] = [];
        for (let j = 0; j < B[0].length; j++) {
            let sum = 0;
            for (let k = 0; k < A[0].length; k++) {
                sum += A[i][k] * B[k][j];
            }
            result[i][j] = sum;
        }
    }
    return result;
}\\n\`;
    }`*/
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
        let variables= []

        edgesInsideCompound.forEach(edge => {
            //identificar que variável é o target, enquanto isso vou coisando
            //console.log(edge.data())


            const edgeClass = edge.classes()[0]
            console.log(edgeClass)
            let source = edge.data('source');
            let target = edge.data('target');

            switch (edgeClass) {
                case 'operation':
                    switch (edge.data('opType')) {
                        case 'Addition':
                            if (!variables.includes(source)) {
                                variables.push(source)
                                loopCode += `   const ${source} = ${operations[source][0]} + ${operations[source][1]}\n`
                            }
                            break
                        case 'Subtraction':
                            if (!variables.includes(source)) {
                                variables.push(source)
                                loopCode += `   const ${source} = ${operations[source][0]} - ${operations[source][1]}\n`
                            }
                            break
                        case 'Load':
                            if (!variables.includes(source)) {
                                variables.push(source)
                                loopCode += `   const ${source} = ${operations[source][0]}[${operations[source][1]}]\n`
                            }
                            break
                        case 'Multiplication':
                            console.log(source)
                            console.log(variables)
                            if (!variables.includes(source)) {
                                variables.push(source)
                                loopCode += `   const ${source} = ${operations[source][0]} * ${operations[source][1]}\n`
                            }
                            break
                        case 'Store':
                            if (!variables.includes(source)) {
                                variables.push(source)
                                loopCode += `   ${target}[${[operations[source][1]]}] = ${operations[source][0]}\n`
                            }
                            break
                        case 'Equality':
                            if (!variables.includes(source)) {
                                variables.push(source)
                                loopCode += `   const ${source} = ${operations[source][0]} === ${operations[source][1]}\n`
                            }
                            break
                        case 'Not':
                            if (!variables.includes(source)) {
                                variables.push(source)
                                loopCode += `   const ${source} = !${operations[source][0]}\n`
                            }
                            break

                    }
                    break
                case 'index':
                    if (indexId === "") {
                        indexId = source
                        variables.push(indexId)
                        declareBeforeLoop += `let ${indexId} = 0\n`
                    }

                    break
                case 'constant':
                    if (!variables.includes(source)) {
                        declareBeforeLoop += `const ${source} = ${edge.data('value')}\n`;
                        variables.push(source);
                    }
                    break
                case 'input':
                    source = source.substring(1)
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
        code += `while (${indexId} < ${loopIterations.data('value')}) {\n`
        code+= loopCode
        code += '}'
        console.log(code)


        //adicionar ordem para cada 1 dos edges adicionados com uma variavel order e++ a seguir
        //depois percorrer, ver se a operação ja foi declarada e adicionar variavel.
        // se a operação esta completa (tem todos os inputs) realizamos quando é chamada?
        // declarar variavel resultado antes de cada loop (ainda tenho de guardar a informação das variaveis nos edges maybe)
        //
        //
        edgesInsideCompound.forEach(edge => {

        })
    })

}
