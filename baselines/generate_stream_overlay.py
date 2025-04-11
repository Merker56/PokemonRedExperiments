import json
from pathlib import Path

OVERLAY_PATH = Path("./overlay")
OVERLAY_PATH.mkdir(exist_ok=True)

def update_agent_overlay(agent_id, party_info, location=None, badges=None, steps=0):
    """
    Update the overlay JSON file for a specific agent.
    
    Parameters:
    - party_info: Dictionary of Pokémon information from PyBoy memory reading
    """
    data = {"party": []}
    for party_key, pokemon_data in party_info.items():
        # Unpack the tuple: (name, level, m1, m2, m3, m4, hp_str, current_hp_int, max_hp_int)
        # Indices 7 and 8 are the new integer values
        pokemon_entry = {
            "name": pokemon_data[0],
            "level": pokemon_data[1],
            "moves": [pokemon_data[2], pokemon_data[3], pokemon_data[4], pokemon_data[5]],
            "hp_string": pokemon_data[6], # Keep original string if needed elsewhere
            "current_hp": pokemon_data[7], # Use CORRECT integer value
            "max_hp": pokemon_data[8]      # Use CORRECT integer value
        }
        data["party"].append(pokemon_entry)

    # Add other data...
    if location is not None: data["location"] = location
    if badges is not None: data["badges"] = badges
    data["steps"] = steps

    # Write JSON...
    output_file = OVERLAY_PATH / f"agent{agent_id}_data.json"
    try:
        with open(output_file, "w") as f:
            json.dump(data, f, indent=4)
    except IOError as e:
        print(f"Error writing overlay file for Agent {agent_id}: {e}")

def prepare_overlay_data(self, location=None, badges=None):
    """
    Method to be called in your main PyBoy environment class
    Prepares data for overlay updates
    """
    # Update Pokémon info first
    self.update_pokemon_info()
    
    # You can add additional data collection here
    overlay_data = {
        "party_info": self.party_info,
        "location": location,  # You'll need to implement this method
        "badges": badges, 
        "steps": self.step_count  # Track steps in your environment
    }
    
    # Update the overlay
    update_agent_overlay(
        agent_id=self.agent_id,  # Add an agent_id attribute to your environment
        **overlay_data
    )