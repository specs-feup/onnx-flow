import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "@/styles/index.css";
import MainRoutes from "@/routes/MainRoutes.tsx";

createRoot(document.getElementById("root")!).render(
    <StrictMode>
        <MainRoutes />
    </StrictMode>,
);
