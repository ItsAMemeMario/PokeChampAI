import type { PlayerPokemon } from "../types/team";

type PokemonCardProps = {
  pokemon: PlayerPokemon;
  index: number;
};

function formatEvs(evs: Record<string, number>): string {
  return Object.entries(evs)
    .map(([stat, value]) => `${value} ${stat}`)
    .join(" / ");
}

export function PokemonCard({ pokemon, index }: PokemonCardProps) {
  return (
    <article className="rounded-lg border border-slate-700 bg-slate-950/80 p-4">
      <header className="mb-2">
        <h3 className="font-semibold text-slate-100">
          {index + 1}. {pokemon.species}
        </h3>
      </header>

      <dl className="space-y-1 text-sm">
        <div className="flex gap-2">
          <dt className="w-16 shrink-0 text-slate-500">Item</dt>
          <dd className="text-slate-300">{pokemon.item}</dd>
        </div>
        <div className="flex gap-2">
          <dt className="w-16 shrink-0 text-slate-500">Ability</dt>
          <dd className="text-slate-300">{pokemon.ability}</dd>
        </div>
        <div className="flex gap-2">
          <dt className="w-16 shrink-0 text-slate-500">Nature</dt>
          <dd className="text-slate-300">{pokemon.nature}</dd>
        </div>
        <div className="flex gap-2">
          <dt className="w-16 shrink-0 text-slate-500">EVs</dt>
          <dd className="text-slate-300">{formatEvs(pokemon.evs)}</dd>
        </div>
        <div className="flex gap-2">
          <dt className="w-16 shrink-0 text-slate-500">Moves</dt>
          <dd className="text-slate-300">{pokemon.moves.join(", ")}</dd>
        </div>
      </dl>
    </article>
  );
}
