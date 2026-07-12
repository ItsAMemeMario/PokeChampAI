import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "../api/client";

export function TeamSetup() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
  });

  return (
    <div className="mx-auto max-w-3xl space-y-6 p-6">
      <header>
        <h1 className="text-2xl font-bold text-slate-100">Team Setup</h1>
        <p className="mt-1 text-sm text-slate-400">
          Paste your Showdown pokepaste and start monitoring.
        </p>
      </header>

      <div className="rounded-lg border border-slate-700 bg-slate-900/60 p-4">
        <label className="mb-2 block text-sm font-medium text-slate-300">
          Pokepaste
        </label>
        <textarea
          className="h-48 w-full rounded-md border border-slate-600 bg-slate-950 p-3 text-sm text-slate-100"
          placeholder="Paste your team pokepaste here..."
          readOnly
        />
        <p className="mt-2 text-xs text-slate-500">
          Team parsing will be wired up in a later milestone.
        </p>
      </div>

      <div className="flex items-center gap-4">
        <button
          type="button"
          className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-500"
          disabled
        >
          Validate Team
        </button>
        <Link
          to="/battle"
          className="text-sm text-indigo-400 hover:text-indigo-300"
        >
          Go to Battle Dashboard
        </Link>
      </div>

      <p className="text-sm text-slate-400">
        Backend:{" "}
        {isLoading
          ? "checking..."
          : isError
            ? "unreachable"
            : data?.status === "ok"
              ? "connected"
              : "unknown"}
      </p>
    </div>
  );
}
