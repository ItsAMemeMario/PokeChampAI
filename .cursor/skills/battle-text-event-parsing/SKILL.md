---
name: battle-text-event-parsing
description: Design battle text event parsing with RapidFuzz fuzzy matching against Pokemon Showdown default.ts strings. Use when designing event parsing for any battle text event, adding OCR battle-log parsers, or matching Showdown battle messages with RapidFuzz.
---

# Battle text event parsing

## When to use this skill

Use this skill when designing event parsing for any battle text event using RapidFuzz fuzzy matching.

## What to do

- Look up its relevant text in https://github.com/smogon/pokemon-showdown/blob/master/data/text/default.ts. Ignore emboldening (the Markdown command **).
- For texts where a large portion are not fixed tokens (e.g. "[POKEMON] used [MOVE]!"), devise a way to locate the substitutable tokens then look for an case-specific way to further locate any small fixed tokens.
- For texts with large fixed tokens (e.g. "[POKEMON] is paralyzed! It may be unable to move!"), fuzzy-match a large token (e.g. "is paralyzed! It may be unable to move!").

## Note

- **Upon looking up a showdown text, ask me if it looks correct.** I will tell you to proceed, or correct your finding by giving you the right attribute name instead.

## Project anchors

- Parser: `backend/app/cv/event_parser.py`
- RapidFuzz name snapping: `backend/app/util/legal_snap.py` (`snap_to_legal`, `fuzz.WRatio`)
- Tests: `backend/tests/test_event_parser.py`
