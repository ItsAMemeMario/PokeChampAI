import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../api/client";
import { ParsedTeamDisplay } from "../components/ParsedTeamDisplay";

export function TeamSetup() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [pokepaste, setPokepaste] = useState("");
  const [validateError, setValidateError] = useState<string | null>(null);

  const { data: health, isLoading: healthLoading, isError: healthError } = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
  });

  const { data: savedTeam, isLoading: teamLoading } = useQuery({
    queryKey: ["team"],
    queryFn: api.getTeam,
  });

  const { data: session } = useQuery({
    queryKey: ["session"],
    queryFn: api.getSession,
    refetchInterval: 5000,
  });

  const validateMutation = useMutation({
    mutationFn: () => api.submitTeam(pokepaste),
    onMutate: () => setValidateError(null),
    onSuccess: (team) => {
      queryClient.setQueryData(["team"], team);
    },
    onError: (error: Error) => {
      setValidateError(error.message);
    },
  });

  const startMutation = useMutation({
    mutationFn: api.startSession,
    onSuccess: (status) => {
      queryClient.setQueryData(["session"], status);
      navigate("/battle");
    },
    onError: (error: Error) => {
      setValidateError(error.message);
    },
  });

  const team = savedTeam ?? null;
  const canValidate = pokepaste.trim().length > 0 && !validateMutation.isPending;
  const canStart = team !== null && !startMutation.isPending;

  return (
    <div className="mx-auto max-w-4xl space-y-6 p-6">
      <header>
        <h1 className="text-2xl font-bold text-slate-100">Team Setup</h1>
        <p className="mt-1 text-sm text-slate-400">
          Paste your Showdown pokepaste and start monitoring.
        </p>
      </header>

      <div className="flex flex-wrap items-center gap-4 text-sm">
        <StatusIndicator
          label="Backend"
          status={
            healthLoading ? "checking" : healthError ? "error" : health?.status === "ok" ? "ok" : "unknown"
          }
        />
        <StatusIndicator
          label="ADB"
          status={
            session === undefined
              ? "checking"
              : session.adb_connected
                ? "ok"
                : "error"
          }
        />
        {session?.cv_running && (
          <span className="rounded-full bg-emerald-900/40 px-2.5 py-0.5 text-xs font-medium text-emerald-300">
            Monitoring active
          </span>
        )}
      </div>

      <div className="rounded-lg border border-slate-700 bg-slate-900/60 p-4">
        <label htmlFor="pokepaste" className="mb-2 block text-sm font-medium text-slate-300">
          Pokepaste
        </label>
        <textarea
          id="pokepaste"
          className="h-56 w-full rounded-md border border-slate-600 bg-slate-950 p-3 font-mono text-sm text-slate-100 placeholder:text-slate-600 focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
          placeholder={`Garchomp @ Lum Berry\nAbility: Rough Skin\nEVs: 252 Atk / 4 SpD / 252 Spe\nJolly Nature\n- Earthquake\n- Dragon Claw\n- Swords Dance\n- Protect`}
          value={pokepaste}
          onChange={(event) => setPokepaste(event.target.value)}
        />
        <p className="mt-2 text-xs text-slate-500">
          Paste a full 6-Pokémon Showdown team. Each Pokémon block is separated by a blank line.
        </p>
      </div>

      {validateError && (
        <div className="rounded-md border border-red-800 bg-red-950/40 px-4 py-3 text-sm text-red-300">
          {validateError}
        </div>
      )}

      <div className="flex flex-wrap items-center gap-3">
        <button
          type="button"
          className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-50"
          disabled={!canValidate}
          onClick={() => validateMutation.mutate()}
        >
          {validateMutation.isPending ? "Validating…" : "Validate Team"}
        </button>
        <button
          type="button"
          className="rounded-md bg-emerald-600 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-500 disabled:cursor-not-allowed disabled:opacity-50"
          disabled={!canStart}
          onClick={() => startMutation.mutate()}
        >
          {startMutation.isPending ? "Starting…" : "Start Monitoring"}
        </button>
        <Link to="/battle" className="text-sm text-indigo-400 hover:text-indigo-300">
          Go to Battle Dashboard
        </Link>
      </div>

      {teamLoading ? (
        <p className="text-sm text-slate-400">Loading saved team…</p>
      ) : team ? (
        <ParsedTeamDisplay team={team} />
      ) : (
        <p className="text-sm text-slate-500">
          No team validated yet. Paste your pokepaste and click Validate Team.
        </p>
      )}
    </div>
  );
}

type StatusIndicatorProps = {
  label: string;
  status: "ok" | "error" | "checking" | "unknown";
};

function StatusIndicator({ label, status }: StatusIndicatorProps) {
  const color =
    status === "ok"
      ? "bg-emerald-500"
      : status === "checking"
        ? "bg-amber-500"
        : "bg-red-500";

  const text =
    status === "ok"
      ? "connected"
      : status === "checking"
        ? "checking…"
        : status === "unknown"
          ? "unknown"
          : "disconnected";

  return (
    <span className="flex items-center gap-2 text-slate-400">
      <span className={`inline-block h-2 w-2 rounded-full ${color}`} aria-hidden />
      {label}: {text}
    </span>
  );
}
