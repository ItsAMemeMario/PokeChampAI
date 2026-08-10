import { useEffect, useState } from "react";
import { getWebSocketUrl } from "../api/client";
import type {
  BattleLiveState,
  BattleLogEvent,
  BattleSnapshot,
  WsMessage,
} from "../types/battle";

const MAX_LOG_EVENTS = 150;
const RECONNECT_MS = 2000;

const EMPTY_STATE: BattleLiveState = {
  connected: false,
  session: null,
  gameState: null,
  battleLogs: [],
  opponentTeamSpecies: null,
  playerSelectedSpecies: null,
  teamPreviewSuggestion: null,
  turnSuggestion: null,
};

function applySnapshot(snapshot: BattleSnapshot): BattleLiveState {
  return {
    connected: true,
    session: snapshot.session,
    gameState: snapshot.game_state,
    battleLogs: snapshot.battle_logs.slice(-MAX_LOG_EVENTS),
    opponentTeamSpecies: snapshot.opponent_team_species,
    playerSelectedSpecies: snapshot.player_selected_species,
    teamPreviewSuggestion: snapshot.team_preview_suggestion,
    turnSuggestion: snapshot.turn_suggestion,
  };
}

function appendLog(
  logs: BattleLogEvent[],
  event: BattleLogEvent,
): BattleLogEvent[] {
  const last = logs[logs.length - 1];
  if (last && isOcrReread(last, event)) {
    return [...logs.slice(0, -1), event];
  }
  const next = [...logs, event];
  return next.length > MAX_LOG_EVENTS ? next.slice(-MAX_LOG_EVENTS) : next;
}

function eventSide(event: BattleLogEvent): string | undefined {
  const pokemon = event.pokemon as { side?: string; species?: string } | undefined;
  const actor = event.actor as { side?: string; species?: string } | undefined;
  return pokemon?.side ?? actor?.side;
}

function eventSpecies(event: BattleLogEvent): string | undefined {
  const pokemon = event.pokemon as { species?: string } | undefined;
  const actor = event.actor as { species?: string } | undefined;
  return pokemon?.species ?? actor?.species;
}

/** Collapse consecutive OCR re-reads of the same on-screen message. */
function isOcrReread(previous: BattleLogEvent, next: BattleLogEvent): boolean {
  if (previous.type !== next.type) {
    return false;
  }
  if (next.type === "move_used") {
    return eventSide(previous) === eventSide(next);
  }
  if (
    next.type === "lead_in" ||
    next.type === "switch_in" ||
    next.type === "switch_out" ||
    next.type === "faint" ||
    next.type === "item_used" ||
    next.type === "stat_change" ||
    next.type === "status_applied" ||
    next.type === "volatile_applied"
  ) {
    if (next.type === "lead_in") {
      const prevSide = (previous as { side?: string }).side;
      const nextSide = (next as { side?: string }).side;
      return prevSide === nextSide;
    }
    return (
      eventSide(previous) === eventSide(next) &&
      eventSpecies(previous) === eventSpecies(next)
    );
  }
  return false;
}

function reduceMessage(
  prev: BattleLiveState,
  message: WsMessage,
): BattleLiveState {
  switch (message.type) {
    case "snapshot":
      return applySnapshot(message.payload);
    case "session":
      return { ...prev, session: message.payload, connected: true };
    case "phase":
      return {
        ...prev,
        connected: true,
        session: prev.session
          ? {
              ...prev.session,
              phase: message.payload.phase,
              turn_number: message.payload.turn_number,
            }
          : {
              phase: message.payload.phase,
              turn_number: message.payload.turn_number,
              cv_running: false,
              team_loaded: false,
              adb_connected: false,
            },
      };
    case "state":
      return { ...prev, connected: true, gameState: message.payload };
    case "log":
      return {
        ...prev,
        connected: true,
        battleLogs: appendLog(prev.battleLogs, message.payload),
      };
    case "log_patched":
      // Completer rewrites structured fields in place on the server. The live
      // log already received the event via "log" (often already patched); do
      // not append a second row that looks like a duplicate.
      return { ...prev, connected: true };
    case "team_preview":
      return {
        ...prev,
        connected: true,
        opponentTeamSpecies: message.payload.opponent_species,
        playerSelectedSpecies: message.payload.player_selected_species,
        teamPreviewSuggestion: message.payload.suggestion,
      };
    case "turn_suggestion":
      return {
        ...prev,
        connected: true,
        turnSuggestion: message.payload,
      };
    default:
      return prev;
  }
}

export function useBattleSocket(): BattleLiveState {
  const [state, setState] = useState<BattleLiveState>(EMPTY_STATE);

  useEffect(() => {
    let cancelled = false;
    let socket: WebSocket | null = null;
    let reconnectTimer: number | null = null;

    const clearReconnect = () => {
      if (reconnectTimer !== null) {
        window.clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
    };

    const scheduleReconnect = () => {
      clearReconnect();
      if (cancelled) {
        return;
      }
      reconnectTimer = window.setTimeout(connect, RECONNECT_MS);
    };

    const connect = () => {
      if (cancelled) {
        return;
      }
      // Ensure only one live socket from this effect (drop orphans before open).
      if (socket !== null) {
        const prev = socket;
        socket = null;
        prev.close();
      }

      const next = new WebSocket(getWebSocketUrl());
      socket = next;

      next.onopen = () => {
        if (cancelled || socket !== next) {
          return;
        }
        setState((prev) => ({ ...prev, connected: true }));
      };

      next.onmessage = (event) => {
        if (cancelled || socket !== next) {
          return;
        }
        try {
          const message = JSON.parse(event.data as string) as WsMessage;
          setState((prev) => reduceMessage(prev, message));
        } catch {
          // Ignore malformed frames.
        }
      };

      next.onclose = () => {
        if (socket === next) {
          socket = null;
        }
        if (cancelled) {
          return;
        }
        setState((prev) => ({ ...prev, connected: false }));
        scheduleReconnect();
      };

      next.onerror = () => {
        next.close();
      };
    };

    connect();

    return () => {
      cancelled = true;
      clearReconnect();
      const active = socket;
      socket = null;
      active?.close();
    };
  }, []);

  return state;
}
