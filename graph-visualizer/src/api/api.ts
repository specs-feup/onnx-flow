import express, { type Request, type Response } from "express";
import { spawn, type ChildProcessWithoutNullStreams } from "child_process";

const app = express();
app.use(express.json());
const PORT = 4000;

let childProcess: ChildProcessWithoutNullStreams | null = null;

app.post('/server/start/:filename', (req: Request, res: Response) => {
    if (childProcess) {
        res.status(400).send('Server is already running. Stop the existing server before starting a new one.');
    }
    else {
        childProcess = spawn('onnx-flow', [req.params["filename"], '--i']);
        console.log('Request received to start server with filename:', req.params["filename"]);
        console.log('Child process started with PID:', childProcess.pid);
        res.status(200).send('Server initialized.');
    }
})

app.post('/server/stop', (req: Request, res: Response) => {
    if (childProcess) {
        childProcess.kill();
        childProcess = null;
        console.log('Request received to stop server.');
        res.status(200).send('Server stopped.');
    } else {
        console.log('No server process to stop.');
        res.status(400).send('No server process to stop.');
    }
});

app.listen(PORT, () => {
    console.log(`Backend Server is running on port ${PORT}`);
    console.log(`Use POST http://localhost:${PORT}/server/start/:filename to start the server and POST http://localhost:${PORT}/server/stop to stop it.`);
});