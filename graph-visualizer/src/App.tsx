import { BrowserRouter, Routes, Route } from "react-router-dom";
import Visualizer from "./App/page.tsx";
import Home from "./Home/page.tsx";
function App() {
    return (
        <BrowserRouter>
            <Routes>
                <Route path="/" element={<Home />} />
                <Route path="/app" element={<Visualizer />} />
            </Routes>
        </BrowserRouter>
    );
}

export default App;