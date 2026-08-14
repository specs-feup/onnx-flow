/**
 * @file MainRoutes.tsx
 * @description Defines the client-side routing hierarchy for the ONNX Graph Visualizer application
 * using React Router v6 createBrowserRouter. Configures the Homepage ('/'), the App visualizer/editor
 * workspace ('/app/:sessionId'), and runtime error boundaries.
 */

import { createBrowserRouter, RouterProvider, useRouteError } from "react-router-dom";
import App from "@/pages/App";
import Home from "@/pages/Home";

/**
 * Error boundary fallback component that renders when an unhandled route error occurs.
 *
 * @returns JSX element containing the error description and status details
 */
function ErrorFallback() {
    const error = useRouteError() as any;
    return (
        <div style={{ padding: "2rem", color: "red" }}>
            <h2>Something went wrong!</h2>
            <pre>{error?.message || error?.statusText || JSON.stringify(error)}</pre>
        </div>
    );
}

/**
 * React Router browser router configuration.
 * - `/` : Home page featuring the file explorer and model picker
 * - `/app/:sessionId` : Interactive visualizer and editor workspace for a loaded ONNX session
 */
const router = createBrowserRouter([
    {
        path: "/",
        element: <Home />,
        errorElement: <ErrorFallback />,
    },
    {
        path: "/app/:sessionId",
        element: <App />,
        errorElement: <ErrorFallback />,
    },
]);

/**
 * Root routing component that provides the router context to the application.
 *
 * @returns JSX.Element wrapping RouterProvider with the configured router
 */
function MainRoutes() {
    return <RouterProvider router={router} />;
}

export default MainRoutes;