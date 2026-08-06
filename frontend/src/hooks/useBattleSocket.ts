import { useEffect, useRef, useState } from "react";
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
  const next = [...logs, event];
  return next.length > MAX_LOG_EVENTS ? next.slice(-MAX_LOG_EVENTS) : next;
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
      // Patches rewrite earlier events; keep the stream readable by appending a note.
      return {
        ...prev,
        connected: true,
        battleLogs: appendLog(prev.battleLogs, {
          ...message.payload.event,
          raw_text: `[patched T${message.payload.turn}] ${message.payload.event.raw_text}`,
        }),
      };
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
  const reconnectTimer = useRef<number | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const disposedRef = useRef(false);

  useEffect(() => {
    disposedRef.current = false;

    const clearReconnect = () => {
      if (reconnectTimer.current !== null) {
        window.clearTimeout(reconnectTimer.current);
        reconnectTimer.current = null;
      }
    };

    const scheduleReconnect = () => {
      clearReconnect();
      if (disposedRef.current) {
        return;
      }
      reconnectTimer.current = window.setTimeout(connect, RECONNECT_MS);
    };

    const connect = () => {
      if (disposedRef.current) {
        return;
      }
      const socket = new WebSocket(getWebSocketUrl());
      socketRef.current = socket;

      socket.onopen = () => {
        setState((prev) => ({ ...prev, connected: true }));
      };

      socket.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data as string) as WsMessage;
          setState((prev) => reduceMessage(prev, message));
        } catch {
          // Ignore malformed frames.
        }
      };

      socket.onclose = () => {
        setState((prev) => ({ ...prev, connected: false }));
        socketRef.current = null;
        scheduleReconnect();
      };

      socket.onerror = () => {
        socket.close();
      };
    };

    connect();

    return () => {
      disposedRef.current = true;
      clearReconnect();
      socketRef.current?.close();
      socketRef.current = null;
    };
  }, []);

  return state;
}
