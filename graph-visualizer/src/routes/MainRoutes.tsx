import { createBrowserRouter, RouterProvider, useRouteError } from "react-router-dom";
import App from "@/pages/App";
import Home from "@/pages/Home";

function ErrorFallback() {
    const error = useRouteError() as any;
    return (
        <div style={{ padding: "2rem", color: "red" }}>
            <h2>Something went wrong!</h2>
            <pre>{error?.message || error?.statusText || JSON.stringify(error)}</pre>
        </div>
    );
}

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

function MainRoutes() {
    return <RouterProvider router={router} />;
}

export default MainRoutes;