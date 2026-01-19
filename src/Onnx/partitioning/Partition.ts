import Graph from "@specs-feup/flow/graph/Graph";
import BaseNode from "@specs-feup/flow/graph/BaseNode";
import OnnxGraph from "../OnnxGraph.js";
import TensorNode from "../TensorNode.js";
import OperationNode from "../OperationNode.js";
import OnnxEdge from "../OnnxEdge.js";
import { PartitionSets } from "./Strategies.js";

/**
 * Clones a TensorNode into the target graph.
 */
function cloneTensor(t: TensorNode.Class, targetGraph: OnnxGraph.Class): TensorNode.Class {
  return targetGraph.addNode(t.id).init(new TensorNode.Builder(
    t.literalType,
    t.shape,
    t.type,
    t.constantValue,
    t.originalInitializer,
    t.extraAttrs
  )).as(TensorNode);
}

/**
 * Clones an OperationNode into the target graph (WITHOUT inputs initially).
 */
function cloneOp(op: OperationNode.Class, targetGraph: OnnxGraph.Class): OperationNode.Class {
  return targetGraph.addNode(op.id).init(new OperationNode.Builder(
    op.type,
    [], // Inputs populated later to preserve order
    op.attributes,
    op.getSubgraphs()
  )).as(OperationNode);
}

/**
 * Updates the internal inputs list of an OperationNode.
 */
function setOpInputs(op: OperationNode.Class, inputs: BaseNode.Class[]) {
  (op.data as any)[OperationNode.TAG].inputs = inputs;
}

export function partitionGraph(originalGraph: OnnxGraph.Class, sets: PartitionSets): { head: OnnxGraph.Class, tail: OnnxGraph.Class } {
  const headGraph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
  const tailGraph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);

  const { head: headIds, tail: tailIds } = sets;

  const headMap = new Map<string, BaseNode.Class>();
  const tailMap = new Map<string, BaseNode.Class>();

  // 1. Initial Clone of All Nodes
  originalGraph.nodes.forEach(node => {
    if (headIds.has(node.id)) {
      if (node.is(TensorNode)) {
        headMap.set(node.id, cloneTensor(node.as(TensorNode), headGraph));
      } else if (node.is(OperationNode)) {
        headMap.set(node.id, cloneOp(node.as(OperationNode), headGraph));
      }
    } else if (tailIds.has(node.id)) {
      if (node.is(TensorNode)) {
        tailMap.set(node.id, cloneTensor(node.as(TensorNode), tailGraph));
      } else if (node.is(OperationNode)) {
        tailMap.set(node.id, cloneOp(node.as(OperationNode), tailGraph));
      }
    }
  });

  // 2. Handle Shared Initializers
  const headInitializers = new Set<string>();
  headMap.forEach((node, id) => {
    if (node.is(TensorNode)) {
      const t = node.as(TensorNode);
      if (t.type === 'initializer' || t.type === 'constant') headInitializers.add(id);
    }
  });

  // 3. Wiring Phase
  const ops = originalGraph.getOperationNodes();

  for (const originalOp of ops) {
    const originalInputs = originalOp.getInputs() ?? [];
    
    // --- Case A: Op is in HEAD ---
    if (headIds.has(originalOp.id)) {
      const clonedOp = headMap.get(originalOp.id)!.as(OperationNode);
      const newInputs: BaseNode.Class[] = [];

      // 3.1 Head Inputs (Tensor -> Op)
      for (const input of originalInputs) {
        if (!headMap.has(input.id)) {
           throw new Error(`[Partition] Op '${originalOp.id}' in Head depends on '${input.id}' which is not in Head.`);
        }
        
        const clonedInput = headMap.get(input.id)!;
        newInputs.push(clonedInput);
        
        if (clonedInput.is(TensorNode)) {
           const t = clonedInput.as(TensorNode);
           headGraph.addEdge(t, clonedOp).init(new OnnxEdge.Builder(t.literalType, t.shape)).as(OnnxEdge);
        }
      }
      setOpInputs(clonedOp, newInputs);

      // 3.2 Head Outputs (Op -> Tensor) - [FIX ADDED HERE]
      originalOp.outgoers.forEach(edge => {
        if (edge.target.is(TensorNode) && headIds.has(edge.target.id)) {
            const clonedT = headMap.get(edge.target.id)!.as(TensorNode);
            headGraph.addEdge(clonedOp, clonedT).init(new OnnxEdge.Builder(
                edge.data[OnnxEdge.TAG].literalType,
                edge.data[OnnxEdge.TAG].shape
            )).as(OnnxEdge);
        }
      });
    } 
    
    // --- Case B: Op is in TAIL ---
    else if (tailIds.has(originalOp.id)) {
      const clonedOp = tailMap.get(originalOp.id)!.as(OperationNode);
      const newInputs: BaseNode.Class[] = [];

      // 3.3 Tail Inputs (Tensor -> Op)
      for (const input of originalInputs) {
        // Option 1: Input exists in Tail (Internal flow)
        if (tailMap.has(input.id)) {
          const clonedInput = tailMap.get(input.id)!;
          newInputs.push(clonedInput);
          if (clonedInput.is(TensorNode)) {
             const t = clonedInput.as(TensorNode);
             tailGraph.addEdge(t, clonedOp).init(new OnnxEdge.Builder(t.literalType, t.shape)).as(OnnxEdge);
          }
          continue;
        }

        // Option 2: Input is in Head (Boundary Crossing) or Shared Initializer
        if (headIds.has(input.id)) {
          if (headInitializers.has(input.id)) {
             // Clone shared initializer into Tail if missing
             if (!tailMap.has(input.id)) {
                const origTensor = originalGraph.getNodeById(input.id).as(TensorNode);
                tailMap.set(input.id, cloneTensor(origTensor, tailGraph));
             }
             const clonedInput = tailMap.get(input.id)!;
             newInputs.push(clonedInput);
             
             const t = clonedInput.as(TensorNode);
             tailGraph.addEdge(t, clonedOp).init(new OnnxEdge.Builder(t.literalType, t.shape)).as(OnnxEdge);
             continue;
          }

          // Boundary Tensor
          const headNode = headMap.get(input.id)!.as(TensorNode);
          if (headNode.type !== 'constant' && headNode.type !== 'initializer') {
             (headNode.data as any)[TensorNode.TAG].type = 'output'; 
          }

          if (!tailMap.has(input.id)) {
             const origTensor = input.as(TensorNode);
             const ghost = tailGraph.addNode(input.id).init(new TensorNode.Builder(
               origTensor.literalType,
               origTensor.shape,
               'input'
             )).as(TensorNode);
             tailMap.set(input.id, ghost);
          }

          const ghostInput = tailMap.get(input.id)!;
          newInputs.push(ghostInput);
          
          const t = ghostInput.as(TensorNode);
          tailGraph.addEdge(t, clonedOp).init(new OnnxEdge.Builder(t.literalType, t.shape)).as(OnnxEdge);
        }
      }
      setOpInputs(clonedOp, newInputs);

      // 3.4 Tail Outputs (Op -> Tensor) - [FIX ADDED HERE]
      originalOp.outgoers.forEach(edge => {
        if (edge.target.is(TensorNode) && tailIds.has(edge.target.id)) {
            const clonedT = tailMap.get(edge.target.id)!.as(TensorNode);
            tailGraph.addEdge(clonedOp, clonedT).init(new OnnxEdge.Builder(
                edge.data[OnnxEdge.TAG].literalType,
                edge.data[OnnxEdge.TAG].shape
            )).as(OnnxEdge);
        }
      });
    }
  }

  return { head: headGraph, tail: tailGraph };
}