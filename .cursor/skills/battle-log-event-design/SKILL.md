---
name: battle-log-event-design
description: Design new BattleLogEvent subtypes for Pokemon Champions Regulation M-B from Showdown default.ts texts, scoped for Gemini battle suggestions. Use when designing a new battle log event type or adding BattleLogEvent schema.
---

# Battle log event design

## When to use this skill

Use this skill when designing a new battle log event type.

## What to do

- Look at https://github.com/smogon/pokemon-showdown/blob/master/data/text/default.ts.
- Locate any game feature relevant to Pokemon Champions Regulation M-B (e.g. no dynamax, no z-moves) and its related texts. Some features may have multiple texts, such as status conditions being covered by a combined `StatusAppliedEvent`.
- Design a `BattleLogEvent` sub-type that makes sense, taking into account what Gemini needs to know to make good suggestions.

## Note

- **When you've drafted an event type, explain what gameplay feature covers and ask me if it makes sense.** I will greenlight the design or give more instructions/corrections.

## Project anchors

- Event schema / union: `backend/app/schema/battle_log.py`
- Reducer: `backend/app/services/gamestate_reducer.py`
- Completer (partial OCR → fields Gemini sees): `backend/app/services/battle_log_completer.py`
- Parsing implementation (after design is approved): skill `battle-text-event-parsing`
