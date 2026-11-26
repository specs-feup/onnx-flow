import express from "express";
import type { Request, Response } from "express";
import { graphviz } from "node-graphviz";
import type OnnxGraph from "../onnx/OnnxGraph.js";
import type OnnxDotFormatter from "../onnx/dot/OnnxDotFormatter.js";
import type CgraDotFormatter from "../onnx/dot/CgraDotFormatter.js";

/**
 * Handles DOT-to-SVG rendering, Graphviz Online URL generation, partitioned DOT
 * cluster formatting, and the local Express visualization server.
 */
export class GraphvizVisualizer {
    /** Renders a Graphviz DOT string to SVG markup using `node-graphviz`. */
    public static async renderDotToSVG(dotGraph: string): Promise<string> {
        return await graphviz.layout(dotGraph);
    }

    /** Builds a shareable Graphviz Online URL encoding the given DOT source. */
    public static generateOnlineLink(dotGraph: string): string {
        return "https://dreampuf.github.io/GraphvizOnline/#" + encodeURIComponent(dotGraph);
    }

    /** Combines head and tail graph partitions into a single clustered DOT digraph. */
    public static buildPartitionedDot(
        head: OnnxGraph.Class,
        tail: OnnxGraph.Class,
        dotFormatter: OnnxDotFormatter | CgraDotFormatter,
    ): string {
        const stripWrapper = (g: OnnxGraph.Class): string =>
            g
                .toString(dotFormatter)
                .replace(/^digraph[^{]*\{/, "")
                .replace(/\}\s*$/, "");

        return `digraph G {
            rankdir=LR;
            compound=true;
            subgraph cluster_head {
                label="HEAD PARTITION";
                color=blue;
                style=dashed;
                ${stripWrapper(head)}
            }
            subgraph cluster_tail {
                label="TAIL PARTITION";
                color=red;
                style=dashed;
                ${stripWrapper(tail)}
            }
        }`;
    }

    /** Starts a lightweight Express HTTP server displaying the rendered SVG graph. */
    public static startServer(getDot: () => string, port: number = 8080): void {
        const app = express();

        app.get("/", async (_req: Request, res: Response) => {
            try {
                const svgContent = await GraphvizVisualizer.renderDotToSVG(getDot());
                res.setHeader("Content-Type", "text/html");
                res.send(GraphvizVisualizer.buildHtmlPage(svgContent));
            } catch (error) {
                console.error("Error rendering graph:", error);
                res.status(500).send(error);
            }
        });

        app.listen(port, () => {
            console.log(`Graphviz Visualization server running at http://localhost:${port}`);
        });
    }

    private static buildHtmlPage(svgContent: string): string {
        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>onnx-flow Graphviz Visualization</title>
    <style>
        body, html {
            margin: 0;
            padding: 0;
            height: 100vh;
            overflow: hidden;
            font-family: Arial, sans-serif;
            background-color: #f9f9f9;
            text-align: center;
        }
        h1 {
            margin: 20px 0;
            font-size: 2.5em;
            color: #333;
        }
        .graph-container {
            max-width: 90%;
            height: calc(100vh - 160px);
            margin: 0 auto;
            overflow: auto;
            border: 1px solid #ddd;
            padding: 10px;
            background-color: #fff;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            border-radius: 8px;
        }
        svg {
            max-width: 100%;
            height: auto;
        }
    </style>
</head>
<body>
    <h1>onnx-flow Graphviz Visualization</h1>
    <div class="graph-container">${svgContent}</div>
</body>
</html>`;
    }
}
