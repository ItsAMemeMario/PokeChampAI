export type PlayerPokemon = {
  species: string;
  item: string;
  ability: string;
  evs: Record<string, number>;
  nature: string;
  moves: string[];
};

export type PlayerTeam = {
  pokemon: PlayerPokemon[];
  regulation: "M-B";
};

export type BattlePhase =
  | "idle"
  | "team_preview"
  | "battle_animation"
  | "action_selection"
  | "ended";

export type SessionStatus = {
  phase: BattlePhase;
  turn_number: number;
  cv_running: boolean;
  team_loaded: boolean;
  adb_connected: boolean;
};
