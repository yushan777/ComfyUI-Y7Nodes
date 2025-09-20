import json
import os
from ..utils.colored_print import color, style
script_directory = os.path.dirname(os.path.abspath(__file__))

default_dims = [
                    {"label": "◼︎ (1:1) - 1024x1024", "value": "1024x1024"},
                    { "label": "🁢 (3:4) - 896x1152", "value": "896x1152" },
                    { "label": "🁢 (5:8) - 832x1216", "value": "832x1216" },
                    { "label": "🁢 (9:16) - 768x1344", "value": "768x1344" },
                    { "label": "🁢 (9:21) - 640x1536", "value": "640x1536" },
                    { "label": "🁢 HD 720p (9:16) - 720x1280", "value": "720x1280" },
                    { "label": "🁢 FHD 1080p (9:16) - 1080x1920", "value": "1080x1920" },
                    { "label": "🁢 UHD 1440p (9:16) - 1440x2560", "value": "1440x2560" },
                    { "label": "🀰 (4:3) - 1152x896", "value": "1152x896" },
                    { "label": "🀰 (3:2) - 1216x832", "value": "1216x832" },
                    { "label": "🀰 (16:9) - 1344x768", "value": "1344x768" },
                    { "label": "🀰 (21:9) - 1536x640", "value": "1536x640" },
                    { "label": "🀰 HD 720p (16:9) - 1280x720", "value": "1280x720" },
                    { "label": "🀰 FHD 1080p (16:9) - 1920x1080", "value": "1920x1080" },
                    { "label": "🀰 UHD 1440p (16:9) - 2560x1440", "value": "2560x1440" }                    
                ]

class Y7Nodes_ImageSizePresets:

    # Class variable to store the loaded dimensions
    dimensions_dict = None

    @classmethod
    def load_dimensions(cls):
        """Load dimensions from JSON file once and cache the result"""
        if cls.dimensions_dict is None:
            try:
                with open(os.path.join(script_directory, 'custom_dimensions.json')) as f:
                    cls.dimensions_dict = json.load(f)
            except FileNotFoundError:
                print(f"custom_dimensions.json not found in {script_directory}. Using defaults.", color.ORANGE)
                # Provide a default entry if the file is not found
                cls.dimensions_dict = default_dims

            except json.JSONDecodeError:
                print(f"Error parsing custom_dimensions.json. Using defaults.", color.RED)
                cls.dimensions_dict = default_dims

        return cls.dimensions_dict
        
    @classmethod
    def INPUT_TYPES(cls):
        dimensions_dict = cls.load_dimensions()

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
                    "min": 120,
                    "max": 7680,
                    "step": 1,
                    "tooltip": "custom width"
                }),
            "custom_h": ("INT", {
                    "default": 1024,
                    "min": 120,
                    "max": 7680,
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
            
            # Use the already loaded dimensions
            dimensions_dict = self.load_dimensions()
            
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
                
        except Exception as e:
            # Catch any unexpected errors
            print(f"Error processing preset: {str(e)}", color.RED)
            return (1024, 1024)  # Default fallback
