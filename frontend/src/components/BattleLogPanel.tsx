import { useEffect, useRef } from "react";
import type { BattleLogEvent } from "../types/battle";
import { SuggestionPanel } from "./SuggestionPanel";

type BattleLogPanelProps = {
  events: BattleLogEvent[];
};

function formatEvent(event: BattleLogEvent): string {
  if (event.raw_text) {
    return String(event.raw_text);
  }
  return event.type;
}

export function BattleLogPanel({ events }: BattleLogPanelProps) {
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [events.length]);

  return (
    <SuggestionPanel title="Battle Log">
      {events.length === 0 ? (
        <p className="text-sm text-slate-400">
          Live battle events will appear here via WebSocket.
        </p>
      ) : (
        <div className="max-h-72 overflow-y-auto rounded-md border border-slate-800 bg-slate-950/50">
          <ul className="divide-y divide-slate-800 font-mono text-xs">
            {events.map((event, index) => (
              <li key={`${event.timestamp}-${index}`} className="px-3 py-2">
                <span className="mr-2 text-slate-500">{event.type}</span>
                <span className="text-slate-300">{formatEvent(event)}</span>
              </li>
            ))}
          </ul>
          <div ref={bottomRef} />
        </div>
      )}
    </SuggestionPanel>
  );
}
