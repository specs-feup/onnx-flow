import CallEdge from "@specs-feup/flow/flow/CallEdge";
import ControlFlowEdge from "@specs-feup/flow/flow/ControlFlowEdge";
import ControlFlowNode from "@specs-feup/flow/flow/ControlFlowNode";
import FlowDotFormatter from "@specs-feup/flow/flow/dot/FlowDotFormatter";
import FlowGraph from "@specs-feup/flow/flow/FlowGraph";
import BaseNode from "@specs-feup/flow/graph/BaseNode";
import Graph from "@specs-feup/flow/graph/Graph";
import Node from "@specs-feup/flow/graph/Node";
//scratchData -> nao serializavel sao objetos complexos (com funções dentro, por exemplo. nao poderia guardar num ficheiro),
//diferente de strings, ints 
var myNode;
(function (myNode) {
    myNode.TAG = "__specs-onnx__my_Node";
    myNode.VERSION = "1";
    class Class extends BaseNode.Class {
        get name() {
            return this.data[myNode.TAG].name;
        }
    }
    myNode.Class = Class;
    class Builder {
        name;
        constructor(name) {
            this.name = name;
        }
        buildData(data) {
            return {
                ...data,
                [myNode.TAG]: {
                    version: myNode.VERSION,
                    name: this.name,
                },
            };
        }
        buildScratchData(scratchData) {
            return {
                ...scratchData,
            };
        }
    }
    myNode.Builder = Builder;
    myNode.TypeGuard = Node.TagTypeGuard(myNode.TAG, myNode.VERSION);
})(myNode || (myNode = {}));
const graph = Graph.create().init(new FlowGraph.Builder()).as(FlowGraph);
//init para inicializar a estrutura (dados), as para metodos
const f1 = graph.addFunction("f1");
const f2 = graph.addFunction("f2");
const c1 = graph.addEdge(f1, f2, "c1").init(new CallEdge.Builder());
const c2 = graph.addEdge(f2, f2, "c2").init(new CallEdge.Builder());
const cf1 = graph.addNode("cf1").init(new ControlFlowNode.Builder(f1));
const cf2 = graph.addNode("cf2").init(new ControlFlowNode.Builder(f1));
const cf3 = graph.addNode("cf3").init(new ControlFlowNode.Builder(f1));
const myN = graph.addNode("myN").init(new myNode.Builder("myN")).as(myNode);
console.log(myN.name);
const cfe1 = graph.addEdge(cf1, cf2, "cfe1").init(new ControlFlowEdge.Builder());
const cfe2 = graph.addEdge(cf2, cf3, "cfe2").init(new ControlFlowEdge.Builder().fake());
const cfe3 = graph.addEdge(cf2, cf2, "cfe3").init(new ControlFlowEdge.Builder());
f1.cfgEntryNode = cf1.as(ControlFlowNode);
const formatter = new FlowDotFormatter();
graph.toFile(formatter, "graph.dot");
//# sourceMappingURL=Graph.js.map