import random
import json
import re
import os
import logging


# ==================================================================

class PhotoPromptGenerator:
    logging.basicConfig(level=logging.DEBUG)

    
    DISABLED = "disabled"
    RANDOM = "random"

    # Load JSON as class variables
    STYLE_AND_FRAMING = None
    SUBJECT_CLASS = None
    ROLE = None  # if active, it will override hairstyle, bodyshape, clothes, accessories
    HAIRSTYLE = None
    BODY_SHAPE = None
    CLOTHING_PRESETS = None
    CLOTHING_UPPER = None
    CLOTHING_LOWER = None
    FOOTWEAR = None
    ACCESSORIES = None
    PRIMARY_ACTION = None 
    GAZE = None 
    HANDS = None 
    LOCATION_INTERIOR = None
    LOCATION_EXTERIOR = None
    TIME_OF_DAY = None 
    LIGHTING = None
    WEATHER = None
    CAMERA_OR_FILM = None
    PHOTOGRAPHER = None

    ADDITIONAL_DETAILS = None

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return ""
    
    @classmethod
    def initialize_class_variables(cls):
        
        cls.STYLE_AND_FRAMING = cls.load_json_file("style_and_framing.json")
        cls.SUBJECT_CLASS = cls.load_json_file("subject_class.json")
        cls.ROLE = cls.load_json_file("roles.json")
        cls.HAIRSTYLE = cls.load_json_file("hairstyle.json")
        cls.BODY_SHAPE = cls.load_json_file("body_shape.json")
        cls.CLOTHING_UPPER = cls.load_json_file("clothing_top.json")
        cls.CLOTHING_LOWER = cls.load_json_file("clothing_lower.json")
        cls.FOOTWEAR = cls.load_json_file("footwear.json")
        cls.ACCESSORIES = cls.load_json_file("accessories.json")
        cls.PRIMARY_ACTION = cls.load_json_file("primary_action.json")
        cls.GAZE = cls.load_json_file("gaze.json")
        cls.HANDS = cls.load_json_file("hands.json")
        cls.LOCATION_INTERIOR = cls.load_json_file("location_interior.json") # this json, each object has 2 properties (description and detail) for longer descriptions)
        cls.LOCATION_EXTERIOR = cls.load_json_file("location_exterior.json") 
        cls.LIGHTING = cls.load_json_file("lighting.json")       
        cls.TIME_OF_DAY = cls.load_json_file("time_of_day.json")
        cls.WEATHER = cls.load_json_file("weather.json")
        cls.CAMERA_OR_FILM = cls.load_json_file("camera_or_film.json")
        cls.PHOTOGRAPHER = cls.load_json_file("photographer.json")

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
                "style_and_framing": (
                    ["disabled", "random"] + cls.STYLE_AND_FRAMING,
                    {"default": "random"},            
                ),
                "subject_class": (
                    ["disabled", "random"] + cls.SUBJECT_CLASS,
                    {"default": "a man"},              
                ),
                "role": (
                    ["disabled", "random"] + cls.ROLE,
                    {"default": "disabled"},              
                ),
                # use a dictionary comprehension to extract only the "style_name" values for the list: 
                "hairstyle": (
                    ["disabled", "random"] + [hairstyle["style_name"] for hairstyle in cls.HAIRSTYLE],
                    {"default": "random"},              
                ),
                # use a dictionary comprehension to extract only the "description" values for the list: 
                "body_shape": (
                    ["disabled", "random"] + [body_shape["description"] for body_shape in cls.BODY_SHAPE],
                    {"default": "random"},              
                ),                
                # use a dictionary comprehension to extract only the "item" values for the list: 
                "clothing_upper": (
                    ["disabled", "random"] + [clothing_upper["item"] for clothing_upper in cls.CLOTHING_UPPER],
                    {"default": "random"},
                ),
                # use a dictionary comprehension to extract only the "item" values for the list: 
                "clothing_lower": (
                    ["disabled", "random"] + [clothing_lower["item"] for clothing_lower in cls.CLOTHING_LOWER],
                    {"default": "random"},
                ),
                # use a dictionary comprehension to extract only the "item" values for the list: 
                "footwear": (
                    ["disabled", "random"] + [footwear["item"] for footwear in cls.FOOTWEAR], 
                    {"default": "random"},
                ),
                "accessories": (
                    ["disabled", "random"] + cls.ACCESSORIES,
                    {"default": "random"},
                ),
                "primary_action": (
                    ["disabled", "random"] + cls.PRIMARY_ACTION,
                    {"default": "random"},
                ),    
                "gaze": (
                    ["disabled", "random"] + cls.GAZE,
                    {"default": "random"},
                ),
                "hands": (
                    ["disabled", "random"] + cls.HANDS,
                    {"default": "random"},
                ),   
                "show_detailed_location": (
                    "BOOLEAN", {"default": True}
                ),                    
                # use a dictionary comprehension to extract only the "description" values for the list: 
                "location_interior": (
                    ["disabled", "random"] + [location["description"] for location in cls.LOCATION_INTERIOR],
                    {"default": "random"},
                ),
                # use a dictionary comprehension to extract only the "description" values for the list: 
                "location_exterior": (
                    ["disabled", "random"] + [location["description"] for location in cls.LOCATION_EXTERIOR],
                    {"default": "random"},
                ),           
                "lighting": (
                    ["disabled", "random"] + cls.LIGHTING,
                    {"default": "random"},
                ),                  
                "time_of_day": (
                    ["disabled", "random"] + cls.TIME_OF_DAY,
                    {"default": "random"},
                ),  
                "weather": (
                    ["disabled", "random"] + cls.WEATHER,
                    {"default": "random"},
                ),   
                "camera_or_film": (
                    ["disabled", "random"] + cls.CAMERA_OR_FILM,
                    {"default": "random"},
                ),   
                "photographer": (
                    ["disabled", "random"] + cls.PHOTOGRAPHER,
                    {"default": "random"},
                ),   
                "remove_commas_periods": (
                    "BOOLEAN", {"default": False}
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
    DESCRIPTION = ""
    
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

                           
        # ------------------------------------------------------------
        # STYLE_AND_FRAMING 
        style_and_framing = kwargs.get("style_and_framing", "disabled")
        if style_and_framing == "disabled":
            style_and_framing = ""
        elif style_and_framing == "random":
            style_and_framing = self.select_random_choice(self.STYLE_AND_FRAMING)      

        # Only append if truthy (is not an empty string)
        if style_and_framing:
            # if framing string begins with a vowel
            if self.begins_with_vowel(style_and_framing) == True:
                # set either "a" or "an"
                components.append(f'an')  
            else:
                components.append(f'a')  

            # now add framing 
            components.append(f'{style_and_framing}')  

        components.append(f'photo of')  
        # ------------------------------------------------------------
        # SUBJECT / CLASS
        subject_or_class = kwargs.get("subject_class", "a man")
        if subject_or_class == "disabled":
            subject_or_class = ""
        elif subject_or_class == "random":
            subject_or_class = self.select_random_choice(self.SUBJECT_CLASS)  

        if subject_or_class:       
            components.append(f'{subject_or_class}')          
        # ------------------------------------------------------------
        # ROLE
        role = kwargs.get("role", "random")
        if role == "disabled":
            role = ""
        elif role == "random":
            role = self.select_random_choice(self.ROLE)    

        if role:
            article = 'an' if self.begins_with_vowel(role) else 'a'
            components.append(f'as {article} {role}') 

        # ------------------------------------------------------------
        # HAIR STYLE
        hairstyle = kwargs.get("hairstyle", "random")
        if hairstyle == "disabled":
            hairstyle = ""
        elif hairstyle == "random":
            hairstyle = self.select_random_choice(self.HAIRSTYLE)    
        else:
            # get whole hairstyle obj so we have access to whole string
            for hstyle in self.HAIRSTYLE:
                if hstyle['style_name'] == hairstyle:
                    hairstyle = hstyle
                    break

        if hairstyle: # if not empty
            full_string = hairstyle.get('full_string')
            components.append(f'{full_string},')           
        # ------------------------------------------------------------   
        # BODY SHAPE
        body_shape = kwargs.get("body_shape", "random")
        if body_shape == "disabled":
            body_shape = ""
        elif body_shape == "random":
            body_shape = self.select_random_choice(self.BODY_SHAPE)    
        else:
            # if user has made selection, then we only have the 'description' value, not the whole obj that also contains the "detail" attribute, so
            # we look for the whole object based on the 'description'
            for bs in self.BODY_SHAPE:
                if bs["description"] == body_shape:
                    body_shape = bs
                    break

        if body_shape: # if not an empty 
            desc = body_shape.get('description')
            detail = body_shape.get('detail')
            article = 'an' if self.begins_with_vowel(desc) else 'a'
            bs_string = f'with {article} {desc}, {detail}'
            components.append(f'{bs_string},')                             
        # ------------------------------------------------------------
        # CLOTHING UPPER
        clothing_upper = kwargs.get("clothing_upper", "random")
        if clothing_upper == "disabled":
            clothing_upper = ""
        elif clothing_upper == "random":
            clothing_upper = self.select_random_choice(self.CLOTHING_UPPER)          
        else:
            # if user has made selection, then we only have 'item' property, not the whole obj that also contains the "default_color" attribute, so
            # we the whole selected location object based on the "item" attribute
            for cloth in self.CLOTHING_UPPER:
                if cloth["item"] == clothing_upper:
                    clothing_upper = cloth
                    break

        if clothing_upper: # if not an empty
            item = clothing_upper.get('item')  # defaults to 'none' if key does not exist
            color = clothing_upper.get('default_color')  # defaults to 'none' if key does not exist
            article = 'an' if self.begins_with_vowel(item) else 'a'
            clothing_string = f'wearing {article} {color} {item}'
            components.append(f'{clothing_string},')      
        # ------------------------------------------------------------
        # CLOTHING LOWER 
        clothing_lower = kwargs.get("clothing_lower", "random")
        if clothing_lower == "disabled":
            clothing_lower = ""
        elif clothing_lower == "random":
            clothing_lower = self.select_random_choice(self.CLOTHING_LOWER)   
        else:
            # if user has made selection, then we only have 'item' property, not the whole obj that also contains the "default_color" attribute, so
            # we the whole selected location object based on the "item" attribute
            for cloth in self.CLOTHING_LOWER:
                if cloth["item"] == clothing_lower:
                    clothing_lower = cloth
                    break         

        if clothing_lower: # if not an empty      
            item = clothing_lower.get('item')  # defaults to 'none' if key does not exist
            color = clothing_lower.get('default_color')  # defaults to 'none' if key does not exist    
            article = ""
            # if no upper clothing 
            if not clothing_upper:
                article = 'an' if self.begins_with_vowel(item) else 'a'
                clothing_string = f'wearing {article} {color} {item}' 
            else:
                clothing_string = f'{color} {item}' 

            components.append(f'{clothing_string},')            
        # ------------------------------------------------------------       
        # ------------------------------------------------------------
        # FOOTWEAR
        footwear = kwargs.get("footwear", "random")
        if footwear == "disabled":
            footwear = ""
        elif footwear == "random":
            footwear = self.select_random_choice(self.FOOTWEAR) 
        else: 
            # if user has made selection, then we only have 'item' property, not the whole obj that also contains the "default_color" attribute, so
            # we the whole selected location object based on the "item" attribute
            for fw in self.CLOTHING_LOWER:
                if footwear["item"] == fw:
                    footwear = fw
                    break         
        
        if footwear:  # if not an empty   
            item = footwear.get('item')  # defaults to 'none' if key does not exist
            color = footwear.get('default_color')  # defaults to 'none' if key does not exist    
            article = ""
            # if no upper clothing AND no lower clothing
            if not clothing_upper and not clothing_lower:
                article = 'an' if self.begins_with_vowel(item) else 'a'
                footwear_string = f'wearing {article} {color} {item}' 
            else:
                footwear_string = f'{color} {item}' 
            components.append(f'{footwear_string},')            
        # ------------------------------------------------------------
        # ACCESSORIES
        accessories = kwargs.get("accessories", "random")
        if accessories == self.DISABLED:
            accessories = ""
        elif accessories == "random":
            accessories = self.select_random_choice(self.ACCESSORIES)      

        # Only append if truthy (is not an empty string) 
        if accessories:
            components.append(f'and a {accessories}.')            
        # ------------------------------------------------------------
        # PRIMARY ACTION
        primary_action = kwargs.get("primary_action", "random")
        if primary_action == self.DISABLED:
            primary_action = ""
        elif primary_action == self.RANDOM:
            primary_action = self.select_random_choice(self.PRIMARY_ACTION)   

        # Only append if truthy (is not an empty string) 
        if primary_action:
            if re.search(r'\bman\b', subject_or_class.lower()):
                components.append("He is")
            elif re.search(r'\bwoman\b', subject_or_class.lower()):
                components.append("She is")
            
            components.append(f'{primary_action},') 
        # ------------------------------------------------------------
        # GAZE
        gaze = kwargs.get("gaze", "random")
        if gaze == self.DISABLED:
            gaze = ""
        elif gaze == self.RANDOM:
            gaze = self.select_random_choice(self.GAZE)   

        # Only append if truthy (is not an empty string) 
        if gaze:
            components.append(f'{gaze}') 

        # ------------------------------------------------------------
        # HANDS
        hands = kwargs.get("hands", "random")
        if hands == self.DISABLED:
            hands = ""
        elif hands == self.RANDOM:
            hands = self.select_random_choice(self.HANDS)   

        # Only append if truthy (is not an empty string) 
        if hands:
            components.append(f'with') 
            components.append(f'{hands}.') 
        # ------------------------------------------------------------
        # SHOW DETAILED LOCATION DESCRIPTION (OR NOT)
        show_detailed_location = kwargs.get("show_detailed_location", True)

        # ------------------------------------------------------------
        # LOCATION - INT 
        # get initial selection froom drop down
        location_interior = kwargs.get("location_interior", "random")        
        if location_interior == self.DISABLED:
            location_interior = ""
        elif location_interior == self.RANDOM:
            location_interior = self.select_random_choice(self.LOCATION_INTERIOR)   
        else: # else is a selected value                    
            # Find the whole selected location object based on the description
            for loc in self.LOCATION_INTERIOR:
                if loc["description"] == location_interior:
                    location_interior = loc
                    break

        # Only append if truthy (is not an empty) 
        if location_interior:
            if re.search(r'\bman\b', subject_or_class.lower()):
                components.append("He is")
            elif re.search(r'\bwoman\b', subject_or_class.lower()):
                components.append("She is")

            if show_detailed_location:
                location_string = f'{location_interior["description"]}, {location_interior["detail"]}'
            else:
                location_string = location_interior[0]
            components.append(f'{location_string},') 

        # ELSE IF INT LOCATION IS EMPTY, THEN WE LOOK AT EXTERNAL LOCATION
        else: # LOCATION - EXT
            print('interior is empty - looking at exterior....')
            # get initial selection froom drop down
            location_exterior = kwargs.get("location_exterior", "random")
            if location_exterior == self.DISABLED:
                location_exterior = ""
            elif location_exterior == self.RANDOM:
                location_exterior = self.select_random_choice(self.LOCATION_EXTERIOR)   
            else:
                # Find the whole selected location object based on the description
                for loc in self.LOCATION_EXTERIOR:
                    if loc["description"] == location_exterior:
                        location_exterior = loc
                        break

            # Only append if truthy (is not an empty string) 
            if location_exterior:
                if re.search(r'\bman\b', subject_or_class.lower()):
                    components.append("He is")
                elif re.search(r'\bwoman\b', subject_or_class.lower()):
                    components.append("She is")

                if show_detailed_location:
                    location_string = f'{location_exterior["description"]}, {location_exterior["detail"]}'
                else:
                    location_string = location_exterior["description"]     

                components.append(f'{location_string},') 

        # ------------------------------------------------------------
        # LIGHTING
        lighting = kwargs.get("lighting", "random")
        if lighting == self.DISABLED:
            lighting = ""
        elif lighting == self.RANDOM:    
            lighting = self.select_random_choice(self.LIGHTING) 
        
        # Only append if truthy (is not an empty string) 
        if lighting:
            components.append(f'{lighting},')
        # ------------------------------------------------------------
        # TIME-OF-DAY
        time_of_day = kwargs.get("time_of_day", "random")
        if time_of_day == self.DISABLED:
            time_of_day = ""
        elif time_of_day == self.RANDOM:
            time_of_day = self.select_random_choice(self.TIME_OF_DAY)   

        # Only append if truthy (is not an empty string) 
        if time_of_day:
            components.append(f'it is {time_of_day}')
        # ------------------------------------------------------------
        # WEATHER
        weather = kwargs.get("weather", "random")
        if weather == self.DISABLED:
            weather = ""
        elif weather == self.RANDOM:
            weather = self.select_random_choice(self.WEATHER)   

        # Only append if truthy (is not an empty string) 
        if weather:
            if location_interior:
                components.append(f'and outside the window the weather is {weather}.')        
            else:
                components.append(f'and the weather is {weather}.')        
        # ------------------------------------------------------------
        # CAMERA OR FILM
        camera_or_film = kwargs.get("camera_or_film", "random")
        if camera_or_film == self.DISABLED:
            camera_or_film = ""
        elif camera_or_film == self.RANDOM:
            camera_or_film = self.select_random_choice(self.CAMERA_OR_FILM)   

        # Only append if truthy (is not an empty string) 
        if camera_or_film:
            components.append(f'{camera_or_film}.') 
        # ------------------------------------------------------------
        # PHOTOGRAPHER
        photographer = kwargs.get("photographer", "random")
        if photographer == self.DISABLED:
            photographer = ""
        elif photographer == self.RANDOM:
            photographer = self.select_random_choice(self.PHOTOGRAPHER)   

        # Only append if truthy (is not an empty string) 
        if photographer:
            components.append(f'Shot by {photographer}.')         
        # ------------------------------------------------------------
        # SHOW DETAILED LOCATION DESCRIPTION (OR NOT)
        remove_commas_periods = kwargs.get("remove_commas_periods", True)
        # ------------------------------------------------------------
        # ------------------------------------------------------------
        # ------------------------------------------------------------
        # ------------------------------------------------------------
        # ------------------------------------------------------------
        # ------------------------------------------------------------

        # concatenate a list of strings with each element separated by a space
        prompt = " ".join(components)

        if remove_commas_periods:
            # Remove commas and periods using regular expressions
            prompt = re.sub(r'[,.]', '', prompt)

        
        return prompt, seed

    # ==============================================================================================================
    def select_random_choice(self, available_choices):
        return self.rng.choices(available_choices, k=1)[0]
    
    # ==============================================================================================================        
    def begins_with_vowel(self, str_input):
        VOWELS = "AEIOUaeiou"
        if str_input[0] in VOWELS:
            return True
        else:
            return False 
    
NODE_CLASS_MAPPINGS = {
    "PhotoPrompter_(Y7)": PhotoPromptGenerator
}

# Human readable names for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "PhotoPrompter_(Y7)": "Photo Prompter (Y7)"
}        