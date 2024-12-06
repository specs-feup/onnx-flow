import CallEdge from "@specs-feup/flow/flow/CallEdge";
import ControlFlowEdge from "@specs-feup/flow/flow/ControlFlowEdge";
import ControlFlowNode from "@specs-feup/flow/flow/ControlFlowNode";
import FlowDotFormatter from "@specs-feup/flow/flow/dot/FlowDotFormatter";
import FlowGraph from "@specs-feup/flow/flow/FlowGraph";
import BaseGraph from "@specs-feup/flow/graph/BaseGraph";
import BaseNode from "@specs-feup/flow/graph/BaseNode";
import Graph from "@specs-feup/flow/graph/Graph";
import Node from "@specs-feup/flow/graph/Node";


//scratchData -> nao serializavel sao objetos complexos (com funções dentro, por exemplo. nao poderia guardar num ficheiro),
//diferente de strings, ints 

namespace myNode {

    export const TAG = "__specs-onnx__my_Node";
    export const VERSION = "1";

    export class Class<
        D extends Data = Data,
        S extends ScratchData = ScratchData,
    > extends BaseNode.Class<D, S> {

        get name(): string {
            return this.data[TAG].name;
        }

    }   

    export class Builder implements Node.Builder<Data, ScratchData> {

        private name: string;

        constructor(name: string) {
            this.name = name;
        }

        buildData(data: BaseNode.Data): Data {
            return {
                ...data,
                [TAG]: {
                    version: VERSION,
                    name: this.name,
                },
            };
        }
        buildScratchData(scratchData: BaseNode.ScratchData): ScratchData {
            return {
                ...scratchData,
            };
        }
    }

    export const TypeGuard = Node.TagTypeGuard<Data, ScratchData>(TAG, VERSION);
    
    export interface Data extends BaseNode.Data {
        [TAG]: {
            version: typeof VERSION;
            name: string;
        };
    }

    export interface ScratchData extends BaseNode.ScratchData {}

}


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