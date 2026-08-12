import express from "express";
import type { NextFunction, Request, Response } from "express";
import { ExplorerSession } from "./ExplorerSession.js";
import { generateUnifiedExplorerJson } from "../flow2json.js";
import { readdir, stat } from "node:fs/promises";
import path from "node:path";
import type { PathLike } from "node:fs";
import { parseOnnxFile } from "../index.js";
import { createGraph, createGraphFromCytoscape } from "../initGraph.js";
import { HistoryManager } from "../Onnx/transformation/tracking/HistoryManager.js";

const sessions = new Map<string, ExplorerSession>();

const pendingDeletions = new Map<string, NodeJS.Timeout>();
const GRACE_PERIOD_MS = 5000;

/**
 * Middleware auxiliar para validar a existência da sessão no mapa.
 * Anexa a sessão ao objeto Request via `res.locals.session`.
 */
function requireSession(req: Request, res: Response, next: NextFunction): void {
    const sessionId = req.params["sessionId"] as string;

    if (!sessionId || !sessions.has(sessionId)) {
        res.status(404).json({
            success: false,
            error: `Session '${sessionId}' not found or expired.`,
        });
        return;
    }

    if (pendingDeletions.has(sessionId)) {
        clearTimeout(pendingDeletions.get(sessionId));
        pendingDeletions.delete(sessionId);
        console.log(`[Keep-Alive] Session Elimination '${sessionId}' cancelled.`);
    }

    res.locals["session"] = sessions.get(sessionId);
    next();
}

export function startExplorerServer(
    initialSession: ExplorerSession | null,
    port: number = 3000,
): void {
    // Se for passada uma sessão inicial, registamo-la com um ID predefinido ("default")
    if (initialSession) {
        // sessions.set("default", initialSession);
    }

    const app = express();
    app.use(express.json({ limit: "150mb" }));

    // 1. CORS Middleware
    app.use((req: Request, res: Response, next: NextFunction) => {
        res.header("Access-Control-Allow-Origin", "*");
        res.header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
        res.header("Access-Control-Allow-Headers", "Content-Type");
        next();
    });

    app.get("/api/files", async (req: Request, res: Response) => {
        try {
            const folderPath: PathLike = "./examples/onnx";
            const entries = await readdir(folderPath, {
                encoding: "utf8",
                withFileTypes: true,
                recursive: false,
            });
            const onnxFiles = entries
                .filter((entry) => entry.isFile() && entry.name.endsWith(".onnx"))
                .map(async (entry) => {
                    const fullPath = path.join(folderPath, entry.name);
                    const fileStat = await stat(fullPath);
                    return {
                        name: entry.name,
                        size: fileStat.size,
                        lastModified: fileStat.mtime,
                    };
                });
            const filesMetadata = await Promise.all(onnxFiles);
            res.json({ success: true, files: filesMetadata });
        } catch (error) {
            res.status(500).json({ success: false, error: String(error) });
        }
    });

    // ============================================================================
    // GESTÃO DE SESSÕES
    // ============================================================================

    // 2. CRIAR NOVA SESSÃO
    app.post("/api/sessions", async (req: Request, res: Response) => {
        try {
            const { onnxFilename } = req.body;
            const sessionId = onnxFilename;

            if (sessions.has(sessionId)) {
                res.status(409).json({
                    success: false,
                    error: `Session '${sessionId}' already exists.`,
                });
                return;
            }

            const onnxObject = await parseOnnxFile("./examples/onnx/" + onnxFilename);
            const graphData = createGraph(onnxObject);

            const newSession = new ExplorerSession(graphData, {
                canonicalize: true,
                fuse: true,
                recurse: true,
                coalesce: true,
                decomposeForCgra: true,
                loopLowering: true,
            });

            sessions.set(sessionId, newSession);

            res.status(201).json({
                success: true,
                sessionId,
                message: "New session started successfully.",
                graph: generateUnifiedExplorerJson(newSession.graph),
            });
        } catch (error) {
            res.status(500).json({ success: false, error: String(error) });
        }
    });

    // 3. LISTAR TODAS AS SESSÕES ATIVAS
    app.get("/api/sessions", (_req: Request, res: Response) => {
        res.json({
            success: true,
            activeSessions: Array.from(sessions.keys()),
        });
    });

    // 4. ELIMINAR UMA SESSÃO
    app.delete("/api/sessions/:sessionId", (req: Request, res: Response) => {
        const sessionId = req.params["sessionId"] as string;
        const deleted = sessions.delete(sessionId);
        if (!deleted) {
            res.status(404).json({ success: false, error: `Session '${sessionId}' not found.` });
            return;
        }
        res.json({ success: true, message: `Session '${sessionId}' deleted.` });
    });

    // 4.5 MARCAR SESSÃO PARA ELIMINAÇÃO (Via Beacon do Frontend)
    app.post("/api/sessions/:sessionId/end", (req: Request, res: Response) => {
        const sessionId = req.params["sessionId"] as string;

        if (!sessions.has(sessionId)) {
            res.status(404).json({ success: false, error: `Session '${sessionId}' not found.` });
            return;
        }

        if (!pendingDeletions.has(sessionId)) {
            const timeoutId = setTimeout(() => {
                sessions.delete(sessionId);
                pendingDeletions.delete(sessionId);
                console.log(`[Cleanup] Session '${sessionId}' eliminated after the grace period.`);
            }, GRACE_PERIOD_MS);

            pendingDeletions.set(sessionId, timeoutId);
            console.log(
                `[Grace Period] Session '${sessionId}' marked for deletion in ${GRACE_PERIOD_MS}ms.`,
            );
        }

        res.status(200).json({
            success: true,
            message: `Session '${sessionId}' marked for deletion.`,
        });
    });

    // ============================================================================
    // OPERAÇÕES SOBRE UMA SESSÃO ESPECÍFICA (via requireSession)
    // ============================================================================

    // 5. Graph Endpoint
    app.get("/api/sessions/:sessionId/graph", requireSession, (_req: Request, res: Response) => {
        try {
            const session = res.locals["session"] as ExplorerSession;
            const payload = generateUnifiedExplorerJson(session.graph);
            res.json(payload);
        } catch (error) {
            res.status(500).json({ error: String(error) });
        }
    });

    // 5.5 Compile Edited Graph Endpoint
    app.post("/api/sessions/:sessionId/compile", requireSession, (req: Request, res: Response) => {
        try {
            const cyJson = req.body.graph || req.body;
            if (!cyJson || !cyJson.elements) {
                res.status(400).json({
                    success: false,
                    error: "Invalid request payload: missing graph elements.",
                });
                return;
            }

            const newGraph = createGraphFromCytoscape(cyJson);

            const session = res.locals["session"] as ExplorerSession;
            session.graph = newGraph;
            session.history = new HistoryManager(newGraph);

            const updatedPayload = generateUnifiedExplorerJson(session.graph);
            res.json({
                success: true,
                message: "ONNX Model compiled successfully!",
                graph: updatedPayload,
            });
        } catch (error) {
            console.error("Compilation error:", error);
            res.status(400).json({
                success: false,
                error: error instanceof Error ? error.message : String(error),
            });
        }
    });

    // 6. Opportunities Endpoint
    app.get("/api/sessions/:sessionId/opportunities", requireSession, (_req: Request, res: Response) => {
        try {
            const session = res.locals["session"] as ExplorerSession;
            const opps = session.getOpportunities().map((opp) => ({
                id: opp.id,
                description: opp.description,
                recipeName: opp.recipeName,
                targetNodeId: opp.targetNodeId,
            }));
            res.json(opps);
        } catch (error) {
            res.status(500).json({
                error: String(error),
                stack: error instanceof Error ? error.stack : undefined,
            });
        }
    });

    // 7. Trigger an optimization
    app.post("/api/sessions/:sessionId/apply/:id", requireSession, (req: Request, res: Response) => {
        try {
            const session = res.locals["session"] as ExplorerSession;
            const oppId = req.params["id"] as string;
            const success = session.applyOpportunity(oppId);

            if (success) {
                res.json({
                    success: true,
                    graph: generateUnifiedExplorerJson(session.graph),
                });
            } else {
                res.status(404).json({ success: false, error: "Opportunity no longer valid." });
            }
        } catch (error) {
            res.status(500).json({ success: false, error: String(error) });
        }
    });

    // 8. Time-Travel: Undo
    app.post("/api/sessions/:sessionId/undo", requireSession, (_req: Request, res: Response) => {
        try {
            const session = res.locals["session"] as ExplorerSession;
            const undone = session.undo();
            res.json({
                success: !!undone,
                patch: undone,
                graph: generateUnifiedExplorerJson(session.graph),
            });
        } catch (error) {
            res.status(500).json({ success: false, error: String(error) });
        }
    });

    // 9. Time-Travel: Redo
    app.post("/api/sessions/:sessionId/redo", requireSession, (_req: Request, res: Response) => {
        try {
            const session = res.locals["session"] as ExplorerSession;
            const redone = session.redo();
            res.json({
                success: !!redone,
                patch: redone,
                graph: generateUnifiedExplorerJson(session.graph),
            });
        } catch (error) {
            res.status(500).json({ success: false, error: String(error) });
        }
    });

    // 10. Update Compiler Settings dynamically
    app.put("/api/sessions/:sessionId/config", requireSession, (req: Request, res: Response) => {
        try {
            const session = res.locals["session"] as ExplorerSession;
            session.options = { ...session.options, ...req.body };
            res.json({ success: true, options: session.options });
        } catch (error) {
            res.status(500).json({ success: false, error: String(error) });
        }
    });

    // 11. Download standard ONNX JSON (.json)
    app.get("/api/sessions/:sessionId/export/onnx-json", requireSession, (_req: Request, res: Response) => {
        try {
            const session = res.locals["session"] as ExplorerSession;
            const jsonProto = session.getOutputOnnxJson();

            res.setHeader("Content-Disposition", "attachment; filename=onnxflow_output.json");
            res.setHeader("Content-Type", "application/json");
            res.send(JSON.stringify(jsonProto, null, 2));
        } catch (error) {
            console.error("Error exporting ONNX JSON:", error);
            res.status(500).json({ success: false, error: String(error) });
        }
    });

    // 12. Download Cytoscape Unified JSON (.json)
    app.get("/api/sessions/:sessionId/export/unified-json", requireSession, (_req: Request, res: Response) => {
        try {
            const session = res.locals["session"] as ExplorerSession;
            const cyJson = session.getOutputUnifiedJson();

            res.setHeader("Content-Disposition", "attachment; filename=onnxflow_output_unified.json");
            res.setHeader("Content-Type", "application/json");
            res.send(JSON.stringify(cyJson, null, 2));
        } catch (error) {
            console.error("\Error exporting Unified JSON:", error);
            res.status(500).json({ success: false, error: String(error) });
        }
    });

    // ============================================================================
    // SERVIDOR & SHUTDOWN
    // ============================================================================

    const server = app.listen(port, () => {
        console.log(`======================================================`);
        console.log(`ONNX-Flow Explorer Server running on port ${port}`);
        console.log(`API Endpoints available:`);
        console.log(`   GET    http://localhost:${port}/api/files`);
        console.log(`   POST   http://localhost:${port}/api/sessions`);
        console.log(`   GET    http://localhost:${port}/api/sessions`);
        console.log(`   DELETE http://localhost:${port}/api/sessions/:sessionId`);
        console.log(`   GET    http://localhost:${port}/api/sessions/:sessionId/graph`);
        console.log(`   GET    http://localhost:${port}/api/sessions/:sessionId/opportunities`);
        console.log(`   POST   http://localhost:${port}/api/sessions/:sessionId/apply/:id`);
        console.log(`   POST   http://localhost:${port}/api/sessions/:sessionId/undo`);
        console.log(`   POST   http://localhost:${port}/api/sessions/:sessionId/redo`);
        console.log(`   PUT    http://localhost:${port}/api/sessions/:sessionId/config`);
        console.log(`   GET    http://localhost:${port}/api/sessions/:sessionId/export/onnx-json`);
        console.log(`   GET    http://localhost:${port}/api/sessions/:sessionId/export/unified-json`);
        console.log(`======================================================\n`);
    });

    app.post("/api/shutdown", (_req: Request, res: Response) => {
        res.json({ success: true, message: "Server shutting down..." });
        console.log("Received shutdown signal. Closing server...");

        setTimeout(() => {
            server.close(() => {
                console.log("Server stopped cleanly. Exiting process.");
                process.exit(0);
            });
        }, 500);
    });
}