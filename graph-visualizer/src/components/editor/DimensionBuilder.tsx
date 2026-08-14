import React from "react";

import { type Shape, type KnownShape } from "@specs-feup/onnx-flow/Onnx/OnnxTypes";


interface DimensionBuilderProps {
    value: Shape | KnownShape;
    onChange: ((shape: KnownShape) => void) | ((shape: Shape) => void);
    hasError?: boolean;
    errorIndices?: number[];
}

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