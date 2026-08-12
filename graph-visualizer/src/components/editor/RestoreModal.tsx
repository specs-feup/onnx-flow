import type { CSSProperties } from "react";

interface RestoreModalProps {
    isOpen: boolean;
    onRestore: () => void;
    onDiscard: () => void;
}

export default function RestoreModal({ isOpen, onRestore, onDiscard }: RestoreModalProps) {
    if (!isOpen) return null;

    const overlayStyle: CSSProperties = {
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: "rgba(0, 0, 0, 0.65)",
        backdropFilter: "blur(3px)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 10000,
    };

    const dialogStyle: CSSProperties = {
        backgroundColor: "#1d1b20",
        border: "2px solid #533b6e",
        borderRadius: "10px",
        padding: "24px",
        width: "440px",
        maxWidth: "90vw",
        boxShadow: "0 8px 32px rgba(0, 0, 0, 0.5)",
        color: "#ffffff",
        display: "flex",
        flexDirection: "column",
        gap: "16px",
    };

    const headerStyle: CSSProperties = {
        fontSize: "1.25rem",
        fontWeight: "bold",
        color: "#e2d9f3",
        display: "flex",
        alignItems: "center",
        gap: "10px",
    };

    const bodyStyle: CSSProperties = {
        fontSize: "0.95rem",
        color: "#ccc",
        lineHeight: "1.5",
    };

    const buttonGroupStyle: CSSProperties = {
        display: "flex",
        justifyContent: "flex-end",
        gap: "12px",
        marginTop: "8px",
    };

    const restoreBtnStyle: CSSProperties = {
        backgroundColor: "#533b6e",
        color: "#ffffff",
        border: "none",
        borderRadius: "6px",
        padding: "8px 16px",
        fontWeight: 600,
        cursor: "pointer",
    };

    const discardBtnStyle: CSSProperties = {
        backgroundColor: "transparent",
        color: "#aaa",
        border: "1px solid #555",
        borderRadius: "6px",
        padding: "8px 16px",
        fontWeight: 500,
        cursor: "pointer",
    };

    return (
        <div style={overlayStyle} onClick={onDiscard}>
            <div style={dialogStyle} onClick={(e) => e.stopPropagation()}>
                <div style={headerStyle}>
                    <span>Restore Saved Graph?</span>
                </div>
                <div style={bodyStyle}>
                    Saved graph edits from a previous session were found in session storage. Would you like to restore your previous changes?
                </div>
                <div style={buttonGroupStyle}>
                    <button style={discardBtnStyle} onClick={onDiscard}>
                        Discard
                    </button>
                    <button style={restoreBtnStyle} onClick={onRestore}>
                        Restore
                    </button>
                </div>
            </div>
        </div>
    );
}
