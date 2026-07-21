import type { DecompositionOptions } from "../../../src/DecompositionOptions.js";
import { createGraph } from "../../../src/initGraph.js";
import { ExplorerSession } from "../../../src/ExplorerAPI/ExplorerSession.js";
import { startExplorerServer } from "../../../src/ExplorerAPI/ExplorerServer.js";
import { parseOnnxFile } from "../../../src/index.js";
import fs from "fs";


export async function ExplorerAPI(inputFilePath: string): Promise<void> {
    let onnxObject;

    if (inputFilePath.endsWith(".json")) {
        onnxObject = JSON.parse(fs.readFileSync(inputFilePath, "utf8"));
    } else {
        onnxObject = await parseOnnxFile(inputFilePath);
    }

    const graph = createGraph(onnxObject);

    const decompOptions: DecompositionOptions = {
        canonicalize: true,
        fuse: true,
        recurse: true,
        coalesce: true,
        decomposeForCgra: true,
        loopLowering: true,
    };

    const session = new ExplorerSession(graph, decompOptions);
    startExplorerServer(session, 3000);
    return;
}