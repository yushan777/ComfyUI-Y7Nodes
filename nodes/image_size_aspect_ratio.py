import json
import os
script_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Y7Nodes_ImageSizePresets:
    @classmethod
    def INPUT_TYPES(cls):
        try:
            with open(os.path.join(script_directory, 'custom_dimensions.json')) as f:
                dimensions_dict = json.load(f)
        except FileNotFoundError:
            dimensions_dict = []
        return {
        "required": {
            "preset": (
                 [f"{d['label']} - {d['value']}" for d in dimensions_dict],
            ),
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
    DESCRIPTION = """
Generates an empty latent image with the specified dimensions.  
The choices are loaded from 'custom_dimensions.json' in the nodes folder.
"""


    def generate(self, preset, custom_w, custom_h):

        # Split the string into label and value
        label, value = preset.split(' - ')
        # Split the value into width and height
        width, height = [x.strip() for x in value.split('x')]
      
        # if preset is set to "custom" then use the custom values
        return (int(width), int(height),)