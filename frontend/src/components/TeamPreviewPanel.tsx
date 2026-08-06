import type { TeamPreviewSuggestion } from "../types/battle";
import { SuggestionPanel } from "./SuggestionPanel";

type TeamPreviewPanelProps = {
  opponentSpecies: string[] | null;
  playerSelectedSpecies: string[] | null;
  suggestion: TeamPreviewSuggestion | null;
};

function SpeciesList({
  label,
  species,
  highlight,
}: {
  label: string;
  species: string[];
  highlight?: string[];
}) {
  const highlightSet = new Set(highlight ?? []);
  return (
    <div>
      <p className="mb-1 text-xs uppercase tracking-wide text-slate-500">{label}</p>
      <ul className="flex flex-wrap gap-1.5">
        {species.map((name) => (
          <li
            key={name}
            className={
              highlightSet.has(name)
                ? "rounded bg-indigo-900/50 px-2 py-0.5 text-sm text-indigo-200"
                : "rounded bg-slate-800 px-2 py-0.5 text-sm text-slate-300"
            }
          >
            {name}
          </li>
        ))}
      </ul>
    </div>
  );
}

export function TeamPreviewPanel({
  opponentSpecies,
  playerSelectedSpecies,
  suggestion,
}: TeamPreviewPanelProps) {
  const empty =
    !opponentSpecies && !playerSelectedSpecies && !suggestion;

  return (
    <SuggestionPanel title="Team Preview">
      {empty ? (
        <p className="text-sm text-slate-400">
          Opponent roster and bring suggestions appear during team preview.
        </p>
      ) : (
        <div className="space-y-4">
          {opponentSpecies && opponentSpecies.length > 0 && (
            <SpeciesList
              label="Opponent 6"
              species={opponentSpecies}
              highlight={suggestion?.predicted_opponent_bring}
            />
          )}
          {suggestion && (
            <>
              <SpeciesList
                label="Predicted opponent bring"
                species={suggestion.predicted_opponent_bring}
                highlight={[...suggestion.predicted_opponent_lead_pair]}
              />
              <SpeciesList
                label="Suggested player bring"
                species={suggestion.suggested_player_bring}
                highlight={[...suggestion.suggested_player_lead_pair]}
              />
              <div>
                <p className="mb-1 text-xs uppercase tracking-wide text-slate-500">
                  Lead pair
                </p>
                <p className="text-sm text-slate-200">
                  {suggestion.suggested_player_lead_pair.join(" + ")}
                </p>
              </div>
              <p className="text-sm leading-relaxed text-slate-400">
                {suggestion.reasoning}
              </p>
            </>
          )}
          {playerSelectedSpecies && playerSelectedSpecies.length > 0 && (
            <SpeciesList
              label="Your selected bring"
              species={playerSelectedSpecies}
            />
          )}
        </div>
      )}
    </SuggestionPanel>
  );
}
