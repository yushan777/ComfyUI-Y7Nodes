import random
import json
import re
import os
import logging

# ==================================================================

class PhotoPromptGenerator:
    logging.basicConfig(level=logging.DEBUG)

    # Load JSON as class variables
    COLOR_MODE_01 = None
    STYLE_TYPE_02 = None
    SUBJECT_CLASS_03 = None


    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    @classmethod
    def initialize_class_variables(cls):
        cls.COLOR_MODE_01 = cls.load_json_file("01_color_mode.json")
        cls.STYLE_TYPE_02 = cls.load_json_file("02_style_type.json")
        cls.SUBJECT_CLASS_03 = cls.load_json_file("03_subject_class.json")

    @classmethod
    def load_json_file(cls, file_name):
        # This method should be updated to handle class level file access if necessary
        file_path = os.path.join(os.path.dirname(__file__), "data", file_name)
        with open(file_path, "r") as file:
            return json.load(file)
        
    @classmethod
    def INPUT_TYPES(cls):
        cls.initialize_class_variables()  # Ensure variables are loaded

        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 1125899906842624}),
                "custom": ("STRING", {}),
                "color_mode": (
                    ["disabled"] + ["random"] + cls.COLOR_MODE_01,
                    {"default": "color"},            
                ),
                "style_type": (
                    ["disabled"] + ["random"] + cls.STYLE_TYPE_02,
                    {"default": "disabled"},              
                ),
                "subject_class": (
                    ["disabled"] + ["random"] + cls.SUBJECT_CLASS_03,
                    {"default": "a man"},              
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
            components.append(f'{custom},')

        components.append(f'a')  
        # ------------------------------------------------------------
        # COLOR MODE
        # get color_mode, if nothing is passed in, default to "color"
        color_mode = kwargs.get("color_mode", "color")
        components.append(f'{color_mode}')
        # ------------------------------------------------------------
        # STYLE/TYPE
        # get style/type, if nothing is passed in, default to "disabled"
        style_or_type = kwargs.get("style_type", "disabled")
        components.append(f'{style_or_type}')  
        components.append(f'photo of')  
        # ------------------------------------------------------------
        # SUBJECT / CLASS
        subject_or_class = kwargs.get("subject_class", "a man")
        components.append(f'{subject_or_class}')          
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

        # concatenate a list of strings with each element separated by a space
        prompt = " ".join(components)

        return prompt, seed


        
    
NODE_CLASS_MAPPINGS = {
    "PhotoPrompter_(Y7)": PhotoPromptGenerator
}

# Human readable names for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "PhotoPrompter_(Y7)": "Photo Prompter (Y7)"
}        