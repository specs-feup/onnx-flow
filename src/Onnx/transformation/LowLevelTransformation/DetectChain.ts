/**********************************************************************
 * Return a *topologically ordered* list of operations that form
 *   a linear chain of {Add,Sub,Mul,Div,MatMul}
 *********************************************************************/
import OperationNode from "../../OperationNode.js";

const SUP = new Set(["Add", "Sub", "Mul", "Div", "MatMul"]);

export function detectChain(seed: OperationNode.Class): OperationNode.Class[] {
  if (!SUP.has(seed.type)) return [];
  const todo = [seed];
  const seen = new Set<OperationNode.Class>();

  while (todo.length) {
    const op = todo.pop()!;
    if (seen.has(op) || !SUP.has(op.type)) continue;
    seen.add(op);

    /* walk forwards */
    op.getOutgoers.targets.filterIs(OperationNode).forEach(t => {
      if (t.getInputs()?.length === 1) todo.push(t);
    });

    /* walk backwards */
    op.getInputs()?.forEach(inp => {
      if (inp.is(OperationNode) && inp.outgoers.targets.length === 1) {
        todo.push(inp as OperationNode.Class);
      }
    });
  }

  /* very small topological sort – Kahn */
  const sorted: OperationNode.Class[] = [];
  const indeg = new Map<OperationNode.Class, number>();
  seen.forEach(n => indeg.set(n, 0));
  seen.forEach(n => {
    n.getOutgoers.targets.filterIs(OperationNode).forEach(t => {
      if (indeg.has(t)) indeg.set(t, indeg.get(t)! + 1);
    });
  });
  const q = [...[...indeg.entries()].filter(([, d]) => d === 0).map(([n]) => n)];
  while (q.length) {
    const n = q.shift()!;
    sorted.push(n);
    n.getOutgoers.targets.filterIs(OperationNode).forEach(t => {
      if (!indeg.has(t)) return;
      indeg.set(t, indeg.get(t)! - 1);
      if (indeg.get(t) === 0) q.push(t);
    });
  }
  return sorted;
}
