import type { TurnAction, TurnSuggestion } from "../types/battle";
import { isSwitchAction } from "../types/battle";
import { SuggestionPanel } from "./SuggestionPanel";

type TurnSuggestionPanelProps = {
  suggestion: TurnSuggestion | null;
};

function formatAction(action: TurnAction): string {
  const body = action.action;
  if (isSwitchAction(body)) {
    return `Switch ${body.switch_out.species} → ${body.switch_in.species}`;
  }
  const targets =
    body.targets.length > 0
      ? ` → ${body.targets.map((t) => t.species).join(", ")}`
      : "";
  const mega = body.mega ? " (Mega)" : "";
  return `${body.actor.species}: ${body.move}${mega}${targets}`;
}

export function TurnSuggestionPanel({ suggestion }: TurnSuggestionPanelProps) {
  return (
    <SuggestionPanel title="Turn Suggestion">
      {!suggestion ? (
        <p className="text-sm text-slate-400">
          Per-turn actions appear when FIGHT is detected (action selection).
        </p>
      ) : (
        <div className="space-y-4">
          <p className="text-sm text-slate-300">
            Turn {suggestion.turn_number}
          </p>
          <ol className="space-y-3">
            {suggestion.actions.map((action, index) => (
              <li
                key={index}
                className="rounded-md border border-slate-700 bg-slate-950/50 p-3"
              >
                <p className="font-medium text-slate-100">
                  {formatAction(action)}
                </p>
                <p className="mt-1 text-sm text-slate-400">{action.reasoning}</p>
              </li>
            ))}
          </ol>
          <p className="text-sm leading-relaxed text-slate-400">
            {suggestion.overall_reasoning}
          </p>
        </div>
      )}
    </SuggestionPanel>
  );
}
