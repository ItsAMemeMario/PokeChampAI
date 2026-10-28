import type { BattlePhase, SessionStatus } from "./team";

export type Side = "player" | "opponent";
export type Slot = 1 | 2;

export type PokemonRef = {
  species: string;
  side: Side;
  slot: Slot;
};

export type StatStages = {
  atk: number;
  def: number;
  spa: number;
  spd: number;
  spe: number;
  evasion: number;
  accuracy: number;
};

export type MoveRestrictionState = {
  restriction: "no_pp" | "forced_move" | "cooldown" | "unusable";
  move: string | null;
  source_item: string | null;
  clears_on_switch: boolean;
};

export type ActivePokemon = {
  species: string;
  hp_percentage: number;
  status_condition: "none" | "brn" | "par" | "slp" | "psn" | "tox" | "frz";
  stat_stages: StatStages;
  volatile_statuses: string[];
  is_protected_this_turn: boolean;
  is_protected_last_turn: boolean;
  perish_turns: number;
  item_state: "unknown" | "held" | "consumed" | "lost";
  move_restrictions: MoveRestrictionState[];
  revealed_ability: string | null;
  revealed_item: string | null;
  revealed_moves: string[];
};

export type BenchedPokemon = {
  species: string;
  hp_percentage: number;
  status_condition: "none" | "brn" | "par" | "slp" | "psn" | "tox" | "frz";
  item_state: "unknown" | "held" | "consumed" | "lost";
  revealed_ability: string | null;
  revealed_item: string | null;
  revealed_moves: string[];
};

export type Hazards = {
  spikes: 0 | 1 | 2 | 3;
  toxic_spikes: 0 | 1 | 2;
  stealth_rocks: 0 | 1;
  sticky_web: 0 | 1;
};

export type SideState = {
  slot_1: ActivePokemon | null;
  slot_2: ActivePokemon | null;
  benched: BenchedPokemon[];
  mega_used: boolean;
  tailwind_turns: number;
  reflect_turns: number;
  light_screen_turns: number;
  aurora_veil_turns: number;
  safeguard_turns: number;
  hazards: Hazards;
};

export type FieldState = {
  weather: "none" | "sun" | "rain" | "sand" | "snow";
  weather_turns: number;
  weather_suppressed: boolean;
  terrain: "none" | "electric" | "grassy" | "misty" | "psychic";
  terrain_turns: number;
  trick_room_turns: number;
  gravity_turns: number;
  magic_room_turns: number;
  wonder_room_turns: number;
  fairy_lock_turns: number;
};

export type GameState = {
  turn_number: number;
  field: FieldState;
  player: SideState;
  opponent: SideState;
};

export type TeamPreviewSuggestion = {
  predicted_opponent_bring: string[];
  predicted_opponent_lead_pair: [string, string];
  suggested_player_bring: string[];
  suggested_player_lead_pair: [string, string];
  reasoning: string;
};

export type TurnMoveAction = {
  actor: PokemonRef;
  mega: boolean;
  move: string;
  targets: PokemonRef[];
};

export type TurnSwitchAction = {
  switch_out: PokemonRef;
  switch_in: PokemonRef;
};

export type TurnAction = {
  action: TurnMoveAction | TurnSwitchAction;
  reasoning: string;
};

export type TurnSuggestion = {
  turn_number: number;
  actions: TurnAction[];
  overall_reasoning: string;
};

export type BattleLogEvent = {
  type: string;
  raw_text: string;
  timestamp: string;
  [key: string]: unknown;
};

export type TeamPreviewPayload = {
  opponent_species: string[] | null;
  player_selected_species: string[] | null;
  suggestion: TeamPreviewSuggestion | null;
};

export type BattleSnapshot = {
  session: SessionStatus;
  game_state: GameState | null;
  battle_logs: BattleLogEvent[];
  opponent_team_species: string[] | null;
  player_selected_species: string[] | null;
  team_preview_suggestion: TeamPreviewSuggestion | null;
  turn_suggestion: TurnSuggestion | null;
};

export type WsMessage =
  | { type: "snapshot"; payload: BattleSnapshot }
  | { type: "session"; payload: SessionStatus }
  | { type: "phase"; payload: { phase: BattlePhase; turn_number: number } }
  | { type: "state"; payload: GameState | null }
  | { type: "log"; payload: BattleLogEvent }
  | {
      type: "log_patched";
      payload: { turn: number; index: number; event: BattleLogEvent };
    }
  | { type: "team_preview"; payload: TeamPreviewPayload }
  | { type: "turn_suggestion"; payload: TurnSuggestion | null };

export type BattleLiveState = {
  connected: boolean;
  session: SessionStatus | null;
  gameState: GameState | null;
  battleLogs: BattleLogEvent[];
  opponentTeamSpecies: string[] | null;
  playerSelectedSpecies: string[] | null;
  teamPreviewSuggestion: TeamPreviewSuggestion | null;
  turnSuggestion: TurnSuggestion | null;
};

export function isSwitchAction(
  action: TurnMoveAction | TurnSwitchAction,
): action is TurnSwitchAction {
  return "switch_out" in action;
}
