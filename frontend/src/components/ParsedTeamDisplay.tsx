import type { PlayerTeam } from "../types/team";
import { PokemonCard } from "./PokemonCard";

type ParsedTeamDisplayProps = {
  team: PlayerTeam;
};

export function ParsedTeamDisplay({ team }: ParsedTeamDisplayProps) {
  return (
    <section className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-slate-100">Parsed Team</h2>
        <span className="rounded-full bg-indigo-900/50 px-2.5 py-0.5 text-xs font-medium text-indigo-300">
          Regulation {team.regulation}
        </span>
      </div>
      <div className="grid gap-3 sm:grid-cols-2">
        {team.pokemon.map((pokemon, index) => (
          <PokemonCard key={`${pokemon.species}-${index}`} pokemon={pokemon} index={index} />
        ))}
      </div>
    </section>
  );
}
