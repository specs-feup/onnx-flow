export function layoutAndStyling(cy, sty){
    cy.style(sty)
    cy.layout({
        name: 'breadthfirst',
        directed: true,
        roots: '.input',
        padding: 10
    }).run()
    return cy
}