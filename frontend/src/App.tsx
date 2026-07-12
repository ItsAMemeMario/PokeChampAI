import { BrowserRouter, Route, Routes } from "react-router-dom";
import { BattleDashboard } from "./pages/BattleDashboard";
import { TeamSetup } from "./pages/TeamSetup";

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-slate-950 text-slate-100">
        <Routes>
          <Route path="/" element={<TeamSetup />} />
          <Route path="/battle" element={<BattleDashboard />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

export default App;
