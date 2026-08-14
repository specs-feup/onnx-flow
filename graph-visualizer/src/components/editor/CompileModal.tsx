/**
 * @file CompileModal.tsx
 * @description Modal dialog component displaying the outcome of ONNX model compilation.
 * Renders success confirmation or detailed error stack trace diagnostics with themed visual feedback.
 */

import type { CSSProperties } from "react";

/**
 * Properties for the CompileModal component.
 */
interface CompileModalProps {
    /** Controls modal open/visible state */
    isOpen: boolean;
    /** Compilation result object containing success flag and informational or error message */
    result: { success: boolean; message: string } | null;
    /** Callback triggered to close the modal dialog */
    onClose: () => void;
}

/**
 * Modal dialog for displaying compiler feedback after compiling the edited in-memory graph.
 *
 * @param props - CompileModal properties
 * @returns JSX element containing the modal dialog or null if closed
 */
export default function CompileModal({ isOpen, result, onClose }: CompileModalProps) {

    if (!isOpen || !result) return null;

    const isSuccess = result.success;

    const overlayStyle: CSSProperties = {
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: "rgba(0, 0, 0, 0.7)",
        backdropFilter: "blur(4px)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 10000,
    };

    const dialogStyle: CSSProperties = {
        backgroundColor: "#1d1b20",
        border: `2px solid ${isSuccess ? "#27ae60" : "#e74c3c"}`,
        borderRadius: "10px",
        padding: "24px",
        width: "480px",
        maxWidth: "90vw",
        boxShadow: "0 8px 32px rgba(0, 0, 0, 0.6)",
        color: "#ffffff",
        display: "flex",
        flexDirection: "column",
        gap: "16px",
    };

    const headerStyle: CSSProperties = {
        fontSize: "1.2rem",
        fontWeight: "bold",
        color: isSuccess ? "#2ecc71" : "#e74c3c",
        display: "flex",
        alignItems: "center",
        gap: "10px",
    };

    const bodyStyle: CSSProperties = {
        fontSize: "0.95rem",
        color: "#ddd",
        lineHeight: "1.5",
        backgroundColor: isSuccess ? "rgba(46, 204, 113, 0.1)" : "rgba(231, 76, 60, 0.1)",
        border: `1px solid ${isSuccess ? "rgba(46, 204, 113, 0.3)" : "rgba(231, 76, 60, 0.3)"}`,
        borderRadius: "6px",
        padding: "12px",
        fontFamily: isSuccess ? "inherit" : "monospace",
        whiteSpace: "pre-wrap",
        maxHeight: "300px",
        overflowY: "auto",
    };

    const buttonGroupStyle: CSSProperties = {
        display: "flex",
        justifyContent: "flex-end",
        marginTop: "8px",
    };

    const buttonStyle: CSSProperties = {
        backgroundColor: isSuccess ? "#27ae60" : "#c0392b",
        color: "#ffffff",
        border: "none",
        borderRadius: "6px",
        padding: "8px 20px",
        fontWeight: 600,
        cursor: "pointer",
    };

    return (
        <div style={overlayStyle} onClick={onClose}>
            <div style={dialogStyle} onClick={(e) => e.stopPropagation()}>
                <div style={headerStyle}>
                    <span>{isSuccess ? "✓ Compilation Successful" : "⚠ Compilation Error"}</span>
                </div>
                <div style={bodyStyle}>
                    {result.message}
                </div>
                <div style={buttonGroupStyle}>
                    <button style={buttonStyle} onClick={onClose}>
                        {isSuccess ? "OK" : "Close"}
                    </button>
                </div>
            </div>
        </div>
    );
}
