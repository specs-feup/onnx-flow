/**
 * @file main.tsx
 * @description Application entry point for the ONNX Graph Visualizer & Editor web application.
 * Initializes the React 19 root, mounts the root routes, and applies global stylesheets.
 */

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "@/styles/index.css";
import MainRoutes from "@/routes/MainRoutes.tsx";

// Mount the React application to the DOM root container with StrictMode enabled
createRoot(document.getElementById("root")!).render(
    <StrictMode>
        <MainRoutes />
    </StrictMode>,
);

