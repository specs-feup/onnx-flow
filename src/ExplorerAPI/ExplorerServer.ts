import express from "express";
import type { Request, Response } from "express";
import { ExplorerSession } from "./ExplorerSession.js";
import { generateUnifiedExplorerJson } from "../flow2json.js";

let activeSession: ExplorerSession | null = null;

export function startExplorerServer(
    initialSession: ExplorerSession | null,
    port: number = 3000,
): void {
    if (initialSession) {
        // activeSession = initialSession;
    }

    const app = express();
    app.use(express.json({ limit: "150mb" }));

    // 1. CORS Middleware (Essential for local frontend development)
    app.use((req: Request, res: Response, next) => {
        res.header("Access-Control-Allow-Origin", "*");
        res.header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE");
        res.header("Access-Control-Allow-Headers", "Content-Type");
        next();
    });

    // 2. DYNAMICALLY LOAD / RESET SESSION
    app.post("/api/session/start", (req: Request, res: Response) => {
        try {
            const { graphData, options } = req.body;
            // Instantiates a brand new session dynamically
            activeSession = new ExplorerSession(graphData, options);
            res.json({
                success: true,
                message: "New session started successfully.",
                graph: generateUnifiedExplorerJson(activeSession.graph),
            });
        } catch (error) {
            res.status(500).json({ success: false, error: String(error) });
        }
    });

    // 3. Graph Endpoint
    app.get("/api/graph", (req: Request, res: Response) => {
        if (!activeSession) {
            res.status(500).json({ error: "There is no active session!" });
        } else {
            try {
                const payload = generateUnifiedExplorerJson(activeSession.graph);
                res.json(payload);
            } catch (error) {
                res.status(500).json({ error: String(error) });
            }
        }
    });

    // 4. Opportunities Endpoint
    app.get("/api/opportunities", (req: Request, res: Response) => {
        if (!activeSession) {
            res.status(500).json({ error: "There is no active session!" });
        } else {
            try {
                const opps = activeSession.getOpportunities().map((opp) => ({
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
        }
    });

    // 5. Trigger an optimization
    app.post("/api/apply/:id", (req: Request, res: Response) => {
        if (!activeSession) {
            res.status(500).json({ error: "There is no active session!" });
        } else {
            try {
                const success = activeSession.applyOpportunity(req.params["id"]);
                if (success) {
                    // Instantly return the updated graph so the UI can re-render
                    res.json({
                        success: true,
                        graph: generateUnifiedExplorerJson(activeSession.graph),
                    });
                } else {
                    res.status(404).json({ success: false, error: "Opportunity no longer valid." });
                }
            } catch (error) {
                res.status(500).json({ success: false, error: String(error) });
            }
        }
    });

    // 6. Time-Travel: Undo
    app.post("/api/undo", (req: Request, res: Response) => {
        if (!activeSession) {
            res.status(500).json({ error: "There is no active session!" });
        } else {
            try {
                const undone = activeSession.undo();
                res.json({
                    success: !!undone,
                    patch: undone,
                    graph: generateUnifiedExplorerJson(activeSession.graph),
                });
            } catch (error) {
                res.status(500).json({ success: false, error: String(error) });
            }
        }
    });

    // 7. Time-Travel: Redo
    app.post("/api/redo", (req: Request, res: Response) => {
        if (!activeSession) {
            res.status(500).json({ error: "There is no active session!" });
        } else {
            try {
                const redone = activeSession.redo();
                res.json({
                    success: !!redone,
                    patch: redone,
                    graph: generateUnifiedExplorerJson(activeSession.graph),
                });
            } catch (error) {
                res.status(500).json({ success: false, error: String(error) });
            }
        }
    });

    // 8. Update Compiler Settings dynamically
    app.put("/api/config", (req: Request, res: Response) => {
        if (!activeSession) {
            res.status(500).json({ error: "There is no active session!" });
        } else {
            try {
                // Merges new options (e.g., { fuse: false }) into the session
                activeSession.options = { ...activeSession.options, ...req.body };
                res.json({ success: true, options: activeSession.options });
            } catch (error) {
                res.status(500).json({ success: false, error: String(error) });
            }
        }
    });

    // 9. Download standard ONNX JSON (.json)
    app.get("/api/export/onnx-json", (req: Request, res: Response) => {
        if (!activeSession) {
            res.status(500).json({ error: "There is no active session!" });
        } else {
            try {
                // Get the raw JSON object from the session
                const jsonProto = activeSession.getOutputOnnxJson();

                // Tell the browser to open a "Save File" dialog
                res.setHeader("Content-Disposition", "attachment; filename=onnxflow_output.json");
                res.setHeader("Content-Type", "application/json");

                // Send the data over the network
                res.send(JSON.stringify(jsonProto, null, 2));
            } catch (error) {
                console.error("\n💥 Error exporting ONNX JSON:", error);
                res.status(500).json({ success: false, error: String(error) });
            }
        }
    });

    // 10. Download Cytoscape Unified JSON (.json)
    app.get("/api/export/unified-json", (req: Request, res: Response) => {
        if (!activeSession) {
            res.status(500).json({ error: "There is no active session!" });
        } else {
            try {
                // Get the Cytoscape-ready object
                const cyJson = activeSession.getOutputUnifiedJson();

                res.setHeader(
                    "Content-Disposition",
                    "attachment; filename=onnxflow_output_unified.json",
                );
                res.setHeader("Content-Type", "application/json");

                res.send(JSON.stringify(cyJson, null, 2));
            } catch (error) {
                console.error("\n💥 Error exporting Unified JSON:", error);
                res.status(500).json({ success: false, error: String(error) });
            }
        }
    });

    //Start the API
    const server = app.listen(port, () => {
        console.log(`\n======================================================`);
        console.log(`🚀 ONNX-Flow Explorer Server running on port ${port}`);
        console.log(`📡 API Endpoints available:`);
        console.log(`   POST http://localhost:${port}/api/session/start`);
        console.log(`   GET  http://localhost:${port}/api/graph`);
        console.log(`   GET  http://localhost:${port}/api/opportunities`);
        console.log(`   POST http://localhost:${port}/api/apply/:id`);
        console.log(`   POST http://localhost:${port}/api/undo`);
        console.log(`   POST http://localhost:${port}/api/redo`);
        console.log(`   GET http://localhost:${port}/api/export/onnx-json`);
        console.log(`   GET http://localhost:${port}/api/export/unified-json`);
        console.log(`======================================================\n`);
    });

    app.post("/api/shutdown", (req: Request, res: Response) => {
        res.json({ success: true, message: "Server shutting down..." });

        console.log("\n🛑 Received shutdown signal. Closing server...");

        // Delay slightly so the HTTP response finishes sending to the browser
        setTimeout(() => {
            server.close(() => {
                console.log("👋 Server stopped cleanly. Exiting process.");
                process.exit(0); // Terminate the Node.js process
            });
        }, 500);
    });
}
