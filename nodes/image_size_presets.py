import json
import os
from ..utils.colored_print import color, style
script_directory = os.path.dirname(os.path.abspath(__file__))

class Y7Nodes_ImageSizePresets:
    @classmethod
    def INPUT_TYPES(cls):
        try:
            print(f"script_directory= {script_directory}", color.ORANGE)
            with open(os.path.join(script_directory, 'custom_dimensions.json')) as f:
                dimensions_dict = json.load(f)
        except FileNotFoundError:
            print(f"custom_dimensions.json not found in {script_directory}. Using default.", color.ORANGE)
            # Provide a default entry if the file is not found
            dimensions_dict = [
                {"label": "◼︎ (1:1) - 1024x1024", "value": "1024x1024"}
            ]

        # Use the full label as the dropdown option
        presets = [d["label"] for d in dimensions_dict]
        
        # Add a "Custom" option
        if not any(preset.startswith("Custom") for preset in presets):
            presets.append("Custom")

        return {
        "required": {
            "preset": (presets,),
            "custom_w": ("INT", {
                    "default": 1024,
                    "min": 512,
                    "max": 2048,
                    "step": 1,
                    "tooltip": "custom width"
                }),
            "custom_h": ("INT", {
                    "default": 1024,
                    "min": 512,
                    "max": 2048,
                    "step": 1,
                    "tooltip": "custom height"
                }),


        },
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("Width", "Height")
    FUNCTION = "generate"
    CATEGORY = "Y7Nodes/Utilss"    

    def generate(self, preset, custom_w, custom_h):
        try:
            # Check if the custom option is selected
            if preset == "Custom":
                return (custom_w, custom_h)
            
            # Find the corresponding value in the JSON data
            try:
                with open(os.path.join(script_directory, 'custom_dimensions.json')) as f:
                    dimensions_dict = json.load(f)
                
                # Find the matching preset
                dimension_entry = next((d for d in dimensions_dict if d["label"] == preset), None)
                
                if dimension_entry:
                    # Get the value from the matched entry
                    value = dimension_entry["value"]
                    
                    # Split the value into width and height
                    width, height = [int(x.strip()) for x in value.split('x')]
                    return (width, height)
                else:
                    print(f"Preset not found in dimensions_dict: {preset}", color.RED)
                    return (1024, 1024)  # Default fallback
                    
            except FileNotFoundError:
                print(f"custom_dimensions.json not found in {script_directory}", color.RED)
                return (1024, 1024)  # Default fallback
            except json.JSONDecodeError:
                print(f"Error parsing custom_dimensions.json", color.RED)
                return (1024, 1024)  # Default fallback
                
        except Exception as e:
            # Catch any unexpected errors
            print(f"Error processing preset: {str(e)}", color.RED)
            return (1024, 1024)  # Default fallback
