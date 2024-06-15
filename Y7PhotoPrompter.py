import random
import json
import re
import os
import logging

# ==================================================================
# Function to load data from a JSON file
def load_json_file(file_name):
    # Construct the absolute path to the data file
    file_path = os.path.join(os.path.dirname(__file__), "data", file_name)
    with open(file_path, "r") as file:
        return json.load(file)
    

COLOR_MODE_01 = load_json_file("01_color_mode.json")
TECHNIQUE_02 = load_json_file("02_technique.json")
SUBJECT_STYLE_03 = load_json_file("03_subject_style.json")

class PhotoPromptGenerator:
    logging.basicConfig(level=logging.DEBUG)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 1125899906842624}),
                "custom": ("STRING", {}),
                "color_mode": (
                    ["random"] + COLOR_MODE_01,
                    {"default": "color"},            
                ),
                "technique": (
                    ["disabled"] + ["random"] + TECHNIQUE_02,
                    {"default": "disabled"},               
                ),
                "subject_style": (
                    ["disabled"] + ["random"] + SUBJECT_STYLE_03,
                    {"default": "disabled"},              
                ),

            },
        }

    RETURN_TYPES = (
        "STRING",
        "INT",
        # "STRING",
        # "STRING",
        # "STRING"
    )
    RETURN_NAMES = (
        "prompt",
        "seed",
        # "t5xxl",
        # "clip_l",
        # "clip_g",
    )    

    FUNCTION = "generate_prompt"
    CATEGORY = "Y7/PromptGenerator"
    
    def generate_prompt(self, **kwargs):
        # get seed. if not provided then default to 0
        seed = kwargs.get("seed", 0)
        # self.rng set to a new instance of random.Random, init with the seed value
        self.rng = random.Random(seed)

        # components hold all the separate prompt parts together
        components = []

        # ------------------------------------------------------------
        # CUSTOM
        # get custom string, if nothing is passed in, default to ""
        custom = kwargs.get("custom", "")
        # if not empty, append custom string to components list
        if custom != "":
            components.append(custom)

        # ------------------------------------------------------------
        # COLOR MODE
        # get color_mode, if nothing is passed in, default to "color"
        color_mode = kwargs.get("color_mode", "color")
        components.append(f'{color_mode}, ')
        # ------------------------------------------------------------
        # TECHNIQUE
        # get technique, if nothing is passed in, default to "disabled"
        technique = kwargs.get("technique", "disabled")
        components.append(f'{technique}, ')        
        # ------------------------------------------------------------
        # SUBJECT STYLE
        subject_style = kwargs.get("subject_style", "portrait")
        components.append(f'{subject_style}, ')        
        # ------------------------------------------------------------
        # ------------------------------------------------------------
        # ------------------------------------------------------------
        # ------------------------------------------------------------
        # ------------------------------------------------------------
        # ------------------------------------------------------------
        # ------------------------------------------------------------
        # ------------------------------------------------------------
        # ------------------------------------------------------------
        # ------------------------------------------------------------
        # ------------------------------------------------------------
        # ------------------------------------------------------------
        # ------------------------------------------------------------
        # ------------------------------------------------------------
        # ------------------------------------------------------------
        # ------------------------------------------------------------
        # ------------------------------------------------------------
        # ------------------------------------------------------------
        # ------------------------------------------------------------
        # ------------------------------------------------------------
        # ------------------------------------------------------------
        # ------------------------------------------------------------

        prompt = " ".join(components)

        return (prompt, seed)


        
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
    
NODE_CLASS_MAPPINGS = {
    "PhotoPrompter_(Y7)": PhotoPromptGenerator
}

# Human readable names for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "PhotoPrompter_(Y7)": "Auto Prompter (Y7)"
}        