type SuggestionPanelProps = {
  title: string;
  children?: React.ReactNode;
};

export function SuggestionPanel({ title, children }: SuggestionPanelProps) {
  return (
    <section className="rounded-lg border border-slate-700 bg-slate-900/60 p-4">
      <h2 className="mb-3 text-lg font-semibold text-slate-100">{title}</h2>
      {children ?? (
        <p className="text-sm text-slate-400">No suggestions yet.</p>
      )}
    </section>
  );
}
