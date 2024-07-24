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

function getSubstringBeforeLastUnderscore(str) {
    let lastUnderscoreIndex = str.lastIndexOf('_');

    if (lastUnderscoreIndex !== -1) {
        return str.substring(0, lastUnderscoreIndex);
    }

    return str;
}


function handleOperation(edge, variables, operations, code) {

    const source = edge.data('source')
    const target = edge.data('target')

    switch (edge.data('opType')) {
        case 'Addition':
            if (!variables[source]) {
                variables[source] = `(${variables[operations[source][0]]} + ${variables[operations[source][1]]})`
            }

            break
        case 'Subtraction':
            if (!variables[source]) {
                variables[source] = `(${variables[operations[source][0]]} - ${variables[operations[source][1]]})`
            }
            break
        case 'Load':
            if (!variables[source]) {
                variables[source] = `${variables[operations[source][0]]}[${variables[operations[source][1]]}]`
            }
            break
        case 'Multiplication':
            if (!variables[source]) {
                variables[source] = `(${variables[operations[source][0]]} * ${variables[operations[source][1]]})`
            }
            break
        case 'Store':
            if (!variables[source]) {
                if (edge.hasClass('result')) {
                    variables[source] = `${getSubstringBeforeLastUnderscore(target)}[${variables[operations[source][1]]}] = ${variables[operations[source][0]]}`
                    code.content += `       ${getSubstringBeforeLastUnderscore(target)}[${variables[operations[source][1]]}] = ${variables[operations[source][0]]}\n`
                } else {
                    variables[source] = `${target}[${variables[operations[source][1]]}] = ${variables[operations[source][0]]}`
                    code.content += `       ${target}[${variables[operations[source][1]]}] = ${variables[operations[source][0]]}\n`
                }
            }
            break
        case 'Equality':
            if (!variables[source]) {
                variables[source] = `(${variables[operations[source][0]]} === ${variables[operations[source][1]]})`
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
function handleEdge(cy, edge, variables, operations, code, outputs) {

    const edgeClass = edge.classes()[0];
    let source = edge.data('source');
    let target = edge.data('target');

    const output = cy.$('#' + target);
    if (!output.outgoers('edge').length && !edge.data('parent')) {
        if (edge.hasClass('compound')) outputs.push(source)
        else outputs.push(target)
    }

    switch (edgeClass) {
        case 'operation':
            handleOperation(edge, variables, operations, code);
            if (edge.hasClass('variable')) {
                if (edge.hasClass('outer'))  code.content += `   let ${source} = ${variables[source]}\n`;
                else code.content += `       ${target} = ${variables[source]}\n`;
            }
            break;
        case 'constant':
            if (!variables[source]) {
                variables[source] = `${edge.data('value')}`;
            }
            break;
        case 'declareBefore':
            variables[source] = source;
            break;
        case 'input':
            variables[source] = getSubstringBeforeLastUnderscore(source);
            break;
        case 'compound':
            if (edge.hasClass('variable')) {
                code.content += `   let ${source} = {}\n`;
            }
            let edgesInsideCompound = cy.edges().filter(edge => edge.data('parent') === source).sort(sortByOrder);

            let declareBeforeLoop = new Set (edgesInsideCompound.filter(e => e.hasClass('declareBefore')).map(e => e.data('source')));
            let indexEdge = edgesInsideCompound.filter(e => e.hasClass('index')).map(e => e.data('source'))

            if (declareBeforeLoop) {
                let loopCode = {content: ''}
                declareBeforeLoop.forEach(source => {
                    loopCode.content +=    `   let ${source} = 0\n`
                })
                loopCode.content += `   while (${indexEdge} < ${variables[operations[source][0]]}) {\n`;
                edgesInsideCompound.forEach(edge => {
                    handleEdge(cy, edge, variables, operations, loopCode, outputs);
                });
                loopCode.content += '   }\n\n';
                code.content += loopCode.content;
            }
            break;
    }

    if (operations[target]) {
        operations[target].push(edge.data('source'));
    } else {
        operations[target] = [edge.data('source')];
    }
}

export function generateCode(cy, data) {
    let declaredVariables = [];
    let code = { content: `function ${data.graph.name}(` };
    data.graph.input.forEach(input => { code.content += `${input.name}, ` });
    data.graph.output.forEach(output => declaredVariables.push(output.name));
    code.content = code.content.slice(0, -2) + ') {\n\n';

    let variables = {};
    let operations = {};
    let outputs = [];

    let edges = cy.edges().filter(edge => !edge.data('parent')).sort(sortByOrder);
    edges.forEach(edge => {
        handleEdge(cy, edge, variables, operations, code, outputs);
    });

    if (outputs.length === 1) {
        code.content += `   return ${outputs[0]}`;
    } else {
        code.content += `   return {`
        outputs.forEach(out => {
            code.content += `${out}`
        })
        code.content += `}\n`
    }
    code.content += '\n}';
    console.log(code.content)
}
