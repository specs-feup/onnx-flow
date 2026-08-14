/**
 * @file DimensionBuilder.tsx
 * @description Interactive tensor shape builder component. Allows adding, removing,
 * and editing individual tensor rank dimensions with support for fixed integers,
 * dynamic/symbolic dimension strings, and undefined/unknown sizes.
 */

import React from "react";
import { type Shape, type KnownShape } from "@specs-feup/onnx-flow/Onnx/OnnxTypes";

/**
 * Properties for the DimensionBuilder component.
 */
interface DimensionBuilderProps {
    /** The active shape array (e.g. [1, 3, 224, 224] or ["batch", 3, undefined, undefined]) */
    value: Shape | KnownShape;
    /** Callback triggered when shape dimensions are modified, added, or removed */
    onChange: ((shape: KnownShape) => void) | ((shape: Shape) => void);
    /** Flag indicating whether the shape has validation errors */
    hasError?: boolean;
    /** Optional array of dimension indices that failed validation */
    errorIndices?: number[];
}

/**
 * Component for interactively configuring multi-dimensional tensor shapes.
 *
 * @param props - DimensionBuilder properties
 * @returns JSX element containing the shape dimension tag list and Add Dimension button
 */
export const DimensionBuilder: React.FC<DimensionBuilderProps> = ({ value = [], onChange, hasError = false, errorIndices = [] }) => {

    const shapeList = value || [];

    const addDimension = () => {
        onChange([...shapeList, undefined]);
    };

    const removeDimension = (indexToRemove: number) => {
        onChange(shapeList.filter((_, idx) => idx !== indexToRemove));
    };

    const updateDimension = (index: number, rawValue: string) => {
        const nextShape = [...shapeList];
        const trimmed = rawValue.trim();

        if (trimmed === "" || trimmed === "?" || trimmed === "undefined") {
            nextShape[index] = undefined;
        } else if (/^-?\d+$/.test(trimmed)) {
            nextShape[index] = Number(trimmed);
        } else {
            nextShape[index] = trimmed;
        }

        onChange(nextShape);
    };

    return (
        <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
                {shapeList.map((dim, idx) => {
                    const isDimError = hasError || errorIndices.includes(idx);
                    return (
                        <div
                            key={idx}
                            style={{
                                display: "flex",
                                alignItems: "center",
                                background: isDimError ? "#321d23" : "#2c2a30",
                                border: isDimError ? "1px solid #ff4d4f" : "1px solid rgb(95, 92, 102)",
                                borderRadius: "6px",
                                padding: "2px 6px",
                                gap: "4px",
                            }}
                        >
                            <span style={{ fontSize: "11px", color: isDimError ? "#ff7875" : "#888" }}>D{idx}:</span>
                            <input
                                type="text"
                                pattern="[0-9]+"
                                value={dim === undefined ? "" : String(dim)}
                                placeholder="?"
                                onChange={(e) => updateDimension(idx, e.target.value)}
                                style={{
                                    width: "60px",
                                    background: "transparent",
                                    border: "none",
                                    color: "white",
                                    fontSize: "13px",
                                    outline: "none",
                                }}
                            />
                            <button
                                type="button"
                                onClick={() => removeDimension(idx)}
                                style={{
                                    background: "transparent",
                                    border: "none",
                                    color: "#ff4d4f",
                                    cursor: "pointer",
                                    padding: "0 4px",
                                }}
                            >
                                ×
                            </button>
                        </div>
                    );
                })}
            </div>

            <button
                type="button"
                onClick={addDimension}
                style={{
                    padding: "6px",
                    borderRadius: "4px",
                    background: "#3e3c46",
                    color: "white",
                    border: "1px solid rgb(95, 92, 102)",
                    cursor: "pointer",
                    fontSize: "12px",
                }}
            >
                + Add Dim (Rank: {shapeList.length})
            </button>
        </div>
    );
};