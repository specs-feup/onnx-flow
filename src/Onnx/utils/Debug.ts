import fs from "fs";
import type BaseNode from "@specs-feup/flow/graph/BaseNode";
import ConstantNode from "../ConstantNode.js";
import TensorNode from "../TensorNode.js";

export function dbg(...args: unknown[]): void {
    console.log("[loop-debug]", ...args);
}

export function dbgTensor(label: string, t: BaseNode.Class | null | undefined): void {
    if (!t) return;
    if (t.is(TensorNode)) {
        const tn = t.as(TensorNode);
        dbg(label, { id: tn.id, kind: tn.type, elemType: tn.literalType, shape: tn.shape });
    } else if (t.is(ConstantNode)) {
        const cn = t.as(ConstantNode);
        dbg(label, { id: cn.id, kind: "constant", elemType: cn.literalType, shape: cn.shape });
    }
}

export function safeWriteJson(filePath: string, obj: unknown): void {
    const fd = fs.openSync(filePath, "w");
    const BUFFER_LIMIT = 1 << 20;
    let buffer = "";

    const flush = () => {
        if (buffer.length > 0) {
            fs.writeSync(fd, buffer);
            buffer = "";
        }
    };
    const write = (s: string) => {
        buffer += s;
        if (buffer.length >= BUFFER_LIMIT) flush();
    };

    const seen = new Set<unknown>();

    const writeValue = (value: unknown) => {
        if (value === null || value === undefined) {
            write("null");
            return;
        }
        const t = typeof value;
        if (t === "number") {
            write(Number.isFinite(value as number) ? String(value) : "null");
            return;
        }
        if (t === "bigint") {
            write(String(value));
            return;
        }
        if (t === "boolean") {
            write(String(value));
            return;
        }
        if (t === "string") {
            write(JSON.stringify(value));
            return;
        }

        if (Array.isArray(value)) {
            write("[");
            for (let i = 0; i < value.length; i++) {
                if (i > 0) write(",");
                writeValue(value[i]);
            }
            write("]");
            return;
        }
        if (t === "object") {
            if (seen.has(value)) throw new Error("safeWriteJson: cyclic reference");
            seen.add(value);
            const keys = Object.keys(value);
            write("{");
            for (let i = 0; i < keys.length; i++) {
                if (i > 0) write(",");
                write(JSON.stringify(keys[i]));
                write(":");
                writeValue((value as Record<string, unknown>)[keys[i]]);
            }
            write("}");
            seen.delete(value);
        }
    };

    try {
        writeValue(obj);
        flush();
    } finally {
        fs.closeSync(fd);
    }
}
