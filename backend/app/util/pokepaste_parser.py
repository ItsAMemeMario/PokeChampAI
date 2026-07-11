import json
import re

# re.VERBOSE allows us to write multi-line regex with comments
POKEPASTE_RE = re.compile(r"""
    ^(?P<species>.+?)\s*@\s*(?P<item>.+?)\s*\n     # Capture Species and Item
    Ability:\s*(?P<ability>.+?)\s*\n               # Capture Ability
    Level:\s*(?P<level>\d+)\s*\n                   # Capture Level
    EVs:\s*(?P<evs>.+?)\s*\n                       # Capture EVs string
    (?P<nature>[A-Za-z]+)\s+Nature\s*\n            # Capture Nature
    (?P<moves>(?:-\s+.+(?:\n|$))+)                 # Capture all moves as a single block
""", re.VERBOSE | re.MULTILINE)

def parse(pokepaste):
    # Split the paste into individual Pokemon blocks (separated by double newlines)
    blocks = [block.strip() for block in pokepaste.strip().split('\n\n') if block.strip()]
    parsed_team = []

    for block in blocks:
        match = POKEPASTE_RE.search(block)
        
        if not match:
            print(f"Failed to parse block:\n{block}\n")
            continue
            
        # Extract the named capture groups into a dictionary
        pokemon = match.groupdict()
        
        # 1. Clean up the Level into an integer
        pokemon['level'] = int(pokemon['level'])
        
        # 2. Convert the Moves block into a list of strings
        # Splits "- Earthquake\n- Protect" into ["Earthquake", "Protect"]
        raw_moves = pokemon['moves'].strip().split('\n')
        pokemon['moves'] = [move.replace('- ', '').strip() for move in raw_moves]
        
        # 3. Convert the EVs string into a dictionary
        # Splits "252 Atk / 4 SpD / 252 Spe" into {"Atk": 252, "SpD": 4, "Spe": 252}
        ev_dict = {}
        for stat_chunk in pokemon['evs'].split('/'):
            amount, stat_name = stat_chunk.strip().split(' ')
            ev_dict[stat_name] = int(amount)
        pokemon['evs'] = ev_dict

        parsed_team.append(pokemon)
        
    return parsed_team

# --- Example Usage ---
# strict_paste = """
# Garchomp @ Lum Berry
# Ability: Rough Skin
# Level: 50
# EVs: 252 Atk / 4 SpD / 252 Spe
# Jolly Nature
# - Earthquake
# - Dragon Claw
# - Swords Dance
# - Protect

# Incineroar @ Sitrus Berry
# Ability: Intimidate
# Level: 50
# EVs: 252 HP / 68 Def / 140 SpD
# Careful Nature
# - Flare Blitz
# - Parting Shot
# - Knock Off
# """

# print(json.dumps(parse(strict_paste), indent=2))
#
# Output:
# [
#   {
#     "species": "Garchomp",
#     "item": "Lum Berry",
#     "ability": "Rough Skin",
#     "level": 50,
#     "evs": {
#       "Atk": 252,
#       "SpD": 4,
#       "Spe": 252
#     },
#     "nature": "Jolly",
#     "moves": [
#       "Earthquake",
#       "Dragon Claw",
#       "Swords Dance",
#       "Protect"
#     ]
#   },
#   {
#     "species": "Incineroar",
#     "item": "Sitrus Berry",
#     "ability": "Intimidate",
#     "level": 50,
#     "evs": {
#       "HP": 252,
#       "Def": 68,
#       "SpD": 140
#     },
#     "nature": "Careful",
#     "moves": [
#       "Flare Blitz",
#       "Parting Shot",
#       "Knock Off"
#     ]
#   }
# ]