import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import { ActiveFieldPanel } from "../components/ActiveFieldPanel";
import { BattleLogPanel } from "../components/BattleLogPanel";
import { PhaseBanner } from "../components/PhaseBanner";
import { TeamPreviewPanel } from "../components/TeamPreviewPanel";
import { TurnSuggestionPanel } from "../components/TurnSuggestionPanel";
import { useBattleSocket } from "../hooks/useBattleSocket";

export function BattleDashboard() {
  const live = useBattleSocket();
  const queryClient = useQueryClient();

  const stopMutation = useMutation({
    mutationFn: api.stopSession,
    onSuccess: (status) => {
      queryClient.setQueryData(["session"], status);
    },
  });

  const phase = live.session?.phase ?? null;
  const turnNumber = live.session?.turn_number ?? 0;

  return (
    <div className="mx-auto max-w-6xl space-y-6 p-6">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold text-slate-100">Battle Dashboard</h1>
          <p className="mt-1 text-sm text-slate-400">
            Live state and Gemini suggestions streamed over WebSocket.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          {live.session?.cv_running && (
            <button
              type="button"
              className="rounded-md border border-slate-600 px-3 py-1.5 text-sm text-slate-300 hover:border-slate-500 hover:text-slate-100 disabled:opacity-50"
              disabled={stopMutation.isPending}
              onClick={() => stopMutation.mutate()}
            >
              {stopMutation.isPending ? "Stopping…" : "Stop Monitoring"}
            </button>
          )}
          <Link
            to="/"
            className="text-sm text-indigo-400 hover:text-indigo-300"
          >
            Team Setup
          </Link>
        </div>
      </header>

      <PhaseBanner
        phase={phase}
        turnNumber={turnNumber}
        connected={live.connected}
        cvRunning={live.session?.cv_running ?? false}
        adbConnected={live.session?.adb_connected ?? false}
      />

      <div className="grid gap-4 lg:grid-cols-2">
        <TeamPreviewPanel
          opponentSpecies={live.opponentTeamSpecies}
          playerSelectedSpecies={live.playerSelectedSpecies}
          suggestion={live.teamPreviewSuggestion}
        />
        <TurnSuggestionPanel suggestion={live.turnSuggestion} />
      </div>

      <ActiveFieldPanel gameState={live.gameState} />

      <BattleLogPanel events={live.battleLogs} />
    </div>
  );
}
