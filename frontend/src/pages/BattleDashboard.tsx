import { Link } from "react-router-dom";
import { SuggestionPanel } from "../components/SuggestionPanel";

export function BattleDashboard() {
  return (
    <div className="mx-auto max-w-5xl space-y-6 p-6">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-100">Battle Dashboard</h1>
          <p className="mt-1 text-sm text-slate-400">Phase: Idle</p>
        </div>
        <Link
          to="/"
          className="text-sm text-indigo-400 hover:text-indigo-300"
        >
          Team Setup
        </Link>
      </header>

      <div className="grid gap-4 md:grid-cols-2">
        <SuggestionPanel title="Team Preview" />
        <SuggestionPanel title="Turn Suggestion" />
      </div>

      <SuggestionPanel title="Battle Log">
        <p className="text-sm text-slate-400">
          Live battle events will appear here via WebSocket.
        </p>
      </SuggestionPanel>
    </div>
  );
}
