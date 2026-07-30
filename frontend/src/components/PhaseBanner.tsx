import type { BattlePhase } from "../types/team";

type PhaseBannerProps = {
  phase: BattlePhase | null;
  turnNumber: number;
  connected: boolean;
  cvRunning: boolean;
  adbConnected: boolean;
};

const PHASE_LABELS: Record<BattlePhase, string> = {
  idle: "Idle",
  team_preview: "Team Preview",
  team_selected: "Team Selected",
  battle_animation: "Battle Animation",
  action_selection: "Action Selection",
  ended: "Ended",
};

export function PhaseBanner({
  phase,
  turnNumber,
  connected,
  cvRunning,
  adbConnected,
}: PhaseBannerProps) {
  const phaseLabel = phase ? PHASE_LABELS[phase] : "Connecting…";
  const showTurn =
    turnNumber > 0 &&
    (phase === "action_selection" ||
      phase === "battle_animation" ||
      phase === "ended");

  return (
    <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-slate-700 bg-slate-900/60 px-4 py-3">
      <div>
        <p className="text-xs uppercase tracking-wide text-slate-500">Phase</p>
        <p className="text-lg font-semibold text-slate-100">
          {phaseLabel}
          {showTurn ? ` · Turn ${turnNumber}` : null}
        </p>
      </div>
      <div className="flex flex-wrap items-center gap-3 text-sm text-slate-400">
        <StatusDot
          label="WebSocket"
          ok={connected}
          okText="live"
          badText="reconnecting"
        />
        <StatusDot
          label="CV"
          ok={cvRunning}
          okText="monitoring"
          badText="stopped"
        />
        <StatusDot
          label="ADB"
          ok={adbConnected}
          okText="connected"
          badText="disconnected"
        />
      </div>
    </div>
  );
}

function StatusDot({
  label,
  ok,
  okText,
  badText,
}: {
  label: string;
  ok: boolean;
  okText: string;
  badText: string;
}) {
  return (
    <span className="flex items-center gap-2">
      <span
        className={`inline-block h-2 w-2 rounded-full ${ok ? "bg-emerald-500" : "bg-amber-500"}`}
        aria-hidden
      />
      {label}: {ok ? okText : badText}
    </span>
  );
}
