import type { ActivePokemon, GameState, SideState } from "../types/battle";
import { SuggestionPanel } from "./SuggestionPanel";

type ActiveFieldPanelProps = {
  gameState: GameState | null;
};

function hpBarColor(pct: number): string {
  if (pct > 50) return "bg-emerald-500";
  if (pct > 20) return "bg-amber-500";
  return "bg-red-500";
}

function SlotCard({
  label,
  pokemon,
}: {
  label: string;
  pokemon: ActivePokemon | null;
}) {
  if (!pokemon) {
    return (
      <div className="rounded-md border border-dashed border-slate-700 bg-slate-950/40 p-3">
        <p className="text-xs text-slate-500">{label}</p>
        <p className="mt-1 text-sm text-slate-600">Empty</p>
      </div>
    );
  }

  return (
    <div className="rounded-md border border-slate-700 bg-slate-950/60 p-3">
      <div className="flex items-start justify-between gap-2">
        <div>
          <p className="text-xs text-slate-500">{label}</p>
          <p className="font-medium text-slate-100">{pokemon.species}</p>
        </div>
        <p className="text-sm font-semibold tabular-nums text-slate-200">
          {pokemon.hp_percentage}%
        </p>
      </div>
      <div className="mt-2 h-2 overflow-hidden rounded bg-slate-800">
        <div
          className={`h-full transition-[width] duration-300 ${hpBarColor(pokemon.hp_percentage)}`}
          style={{ width: `${pokemon.hp_percentage}%` }}
        />
      </div>
      <div className="mt-2 flex flex-wrap gap-2 text-xs text-slate-400">
        {pokemon.status_condition !== "none" && (
          <span className="rounded bg-slate-800 px-1.5 py-0.5 uppercase">
            {pokemon.status_condition}
          </span>
        )}
        {pokemon.revealed_ability && (
          <span>Ability: {pokemon.revealed_ability}</span>
        )}
        {pokemon.revealed_item && <span>Item: {pokemon.revealed_item}</span>}
      </div>
    </div>
  );
}

function SideColumn({ title, side }: { title: string; side: SideState }) {
  return (
    <div className="space-y-2">
      <h3 className="text-sm font-medium text-slate-300">{title}</h3>
      <SlotCard label="Slot 1" pokemon={side.slot_1} />
      <SlotCard label="Slot 2" pokemon={side.slot_2} />
    </div>
  );
}

export function ActiveFieldPanel({ gameState }: ActiveFieldPanelProps) {
  return (
    <SuggestionPanel title="Active Field">
      {!gameState ? (
        <p className="text-sm text-slate-400">
          Waiting for battle to begin. Game state appears after team preview.
        </p>
      ) : (
        <div className="space-y-4">
          <div className="flex flex-wrap gap-3 text-xs text-slate-400">
            <span>Weather: {gameState.field.weather}</span>
            <span>Terrain: {gameState.field.terrain}</span>
            <span>
              Trick Room:{" "}
              {gameState.field.trick_room_turns > 0
                ? `${gameState.field.trick_room_turns} turns`
                : "off"}
            </span>
          </div>
          <div className="grid gap-4 sm:grid-cols-2">
            <SideColumn title="Opponent" side={gameState.opponent} />
            <SideColumn title="Player" side={gameState.player} />
          </div>
        </div>
      )}
    </SuggestionPanel>
  );
}
