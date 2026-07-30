import type { PlayerTeam, SessionStatus } from "../types/team";
import type {
  BattleLogEvent,
  GameState,
  TeamPreviewPayload,
  TurnSuggestion,
} from "../types/battle";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function parseErrorMessage(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: string | { msg: string }[] };
    if (typeof body.detail === "string") {
      return body.detail;
    }
    if (Array.isArray(body.detail)) {
      return body.detail.map((entry) => entry.msg).join(", ");
    }
  } catch {
    const text = await response.text();
    if (text) {
      return text;
    }
  }
  return `Request failed: ${response.status}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
    ...init,
  });

  if (!response.ok) {
    const message = await parseErrorMessage(response);
    throw new ApiError(message, response.status);
  }

  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<{ status: string }>("/api/health"),

  submitTeam: (pokepaste: string) =>
    request<PlayerTeam>("/api/team", {
      method: "POST",
      body: JSON.stringify({ pokepaste }),
    }),

  getTeam: async (): Promise<PlayerTeam | null> => {
    const response = await fetch(`${API_BASE}/api/team`, {
      headers: { "Content-Type": "application/json" },
    });
    if (response.status === 404) {
      return null;
    }
    if (!response.ok) {
      const message = await parseErrorMessage(response);
      throw new ApiError(message, response.status);
    }
    return response.json() as Promise<PlayerTeam>;
  },

  getSession: () => request<SessionStatus>("/api/session"),

  startSession: () =>
    request<SessionStatus>("/api/session/start", { method: "POST" }),

  stopSession: () =>
    request<SessionStatus>("/api/session/stop", { method: "POST" }),

  getState: () => request<{ game_state: GameState | null }>("/api/state"),

  getLogs: (limit = 100) =>
    request<{ events: BattleLogEvent[] }>(`/api/logs?limit=${limit}`),

  getTeamPreviewSuggestion: () =>
    request<TeamPreviewPayload>("/api/suggestions/team-preview"),

  getTurnSuggestion: () =>
    request<{ suggestion: TurnSuggestion | null }>("/api/suggestions/turn"),
};

export function getWebSocketUrl(): string {
  const base = API_BASE.replace(/^http/, "ws");
  return `${base}/ws`;
}
