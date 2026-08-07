import { createBrowserRouter, RouterProvider, useRouteError } from "react-router-dom";
import App from "@/pages/App";
import Home from "@/pages/Home";

// 1. Create a quick error display component
function ErrorFallback() {
    const error = useRouteError() as any;
    return (
        <div style={{ padding: "2rem", color: "red" }}>
            <h2>Something went wrong!</h2>
            <pre>{error?.message || error?.statusText || JSON.stringify(error)}</pre>
        </div>
    );
}

// 2. Attach it to your routes
const router = createBrowserRouter([
    {
        path: "/",
        element: <Home />,
        errorElement: <ErrorFallback />, // <-- Catches crashes in <Home />
    },
    {
        path: "/app/:sessionId",
        element: <App />,
        errorElement: <ErrorFallback />, // <-- Catches crashes in <App />
    },
]);

function MainRoutes() {
    return <RouterProvider router={router} />;
}

export default MainRoutes;