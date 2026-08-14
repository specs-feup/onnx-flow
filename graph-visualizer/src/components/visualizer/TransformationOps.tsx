/**
 * @file TransformationOps.tsx
 * @description Side panel component displaying discovered graph transformation recipes
 * (e.g. LowerGemmRecipe, LowerReluRecipe, LoopFusionMatcher). Allows applying transformations
 * with one click and records steps into the undo history stack.
 */

import type { CSSProperties } from "react";
import { useParams } from "react-router-dom";
import {
    type TransformationOpportunity,
    applyTransformation,
    fetchGraph,
    fetchTransformationOpportunities,
} from "@/api/api.ts";
import type { CytoscapeData } from "@/types/Cytoscape.ts";

/**
 * Properties for the TransformationOps component.
 */
interface TransformationOpsProps {
    /** CSS style properties */
    style: CSSProperties;
    /** Object containing array of detected transformation opportunities and setter */
    transformationOps: {
        ops: TransformationOpportunity[];
        setOps: (transformationOps: TransformationOpportunity[]) => void;
    };
    /** State setter for the active Cytoscape graph data */
    setCytoscapeData: (data: CytoscapeData | null) => void;
    /** Object containing undo/redo history stacks */
    transformationsHistory: {
        undo: { stack: string[]; setStack: (history: string[]) => void };
        redo: { stack: string[]; setStack: (history: string[]) => void };
    };
}

/**
 * Optimization and transformation opportunities action list component.
 *
 * @param props - TransformationOps properties
 * @returns JSX element containing the transformation buttons and recipe descriptions
 */
export default function TransformationOps(props: TransformationOpsProps) {
    const { sessionId } = useParams();


    return (
        <>
            {props.transformationOps.ops.map((op) => (
                <>
                <button
                    onClick={async () => {
                        const operationId: string = op.id;
                        props.transformationOps.setOps([]);
                        props.setCytoscapeData(null);
                        await applyTransformation(3000, sessionId!, operationId);
                        props.transformationsHistory.undo.setStack([
                            ...props.transformationsHistory.undo.stack,
                            operationId,
                        ]);
                        props.transformationsHistory.redo.setStack([]);
                        props.setCytoscapeData(await fetchGraph(3000, sessionId!));
                        props.transformationOps.setOps(
                            await fetchTransformationOpportunities(3000, sessionId!),
                        );
                    }}
                >
                    {op.recipeName} - {op.targetNodeId}
                </button>
                <span>{op.description}</span>
                </>
            ))}
        </>
    );
}
