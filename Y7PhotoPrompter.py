import random
import json
import re
import os
import logging
from enum import Enum

class SUBJECT_TYPE(Enum):
    MAN = 1
    WOMAN = 2
    OTHER = 3

# ==================================================================

class PhotoPromptGenerator:
    logging.basicConfig(level=logging.DEBUG)
    
    DISABLED = "disabled"
    RANDOM = "random"
    
    # load config file
    CONFIG_DATA = None 
    
    # Load JSON as class variables
    STYLE_AND_FRAMING = None
    SUBJECT_CLASS = None
    ROLE = None  # if active, it will override hairstyle, bodyshape, clothes, accessories
    HAIRSTYLE = None
    BODY_SHAPE = None
    CLOTHING_PRESETS = None
    CLOTHING_UPPER = None
    CLOTHING_UNDERGARMENT = None
    CLOTHING_LOWER = None    
    FOOTWEAR = None
    ACCESSORIES_HEAD = None
    ACCESSORIES_OTHER = None
    ACTION = None 
    GAZE = None 
    HANDS = None 
    SCENE_INDOOR = None
    SCENE_OUTDOOR = None
    TIME_OF_DAY = None 
    LIGHTING = None
    WEATHER = None
    CAMERA_OR_FILM = None
    PHOTOGRAPHER = None

    COLORS = None # Not seen by user, invisible list

    # ===========================================================
    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    # ===========================================================
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return ""
    
    # ===========================================================
    @classmethod
    def initialize_class_variables(cls):
        
        cls.CONFIG_DATA = cls.load_config()

        cls.STYLE_AND_FRAMING = cls.load_data_files("style_and_framing.json")
        cls.SUBJECT_CLASS = cls.load_data_files("subject_class.json")
        cls.ROLE = cls.load_data_files("roles.json")
        cls.HAIRSTYLE = cls.load_data_files("hairstyle.json")
        cls.BODY_SHAPE = cls.load_data_files("body_shape.json")
        cls.CLOTHING_UPPER = cls.load_data_files("clothing_top.json") # (item, default color, gender, category)        
        cls.CLOTHING_LOWER = cls.load_data_files("clothing_lower.json") # (item, default color, gender, category)
        cls.CLOTHING_UNDERGARMENT = cls.load_data_files("clothing_undergarment.json") # (item, default color, gender, category)
        cls.FOOTWEAR = cls.load_data_files("footwear.json")
        cls.ACCESSORIES_HEAD = cls.load_data_files("accessories_head.json")
        cls.ACCESSORIES_OTHER = cls.load_data_files("accessories_other.json")
        cls.ACTION = cls.load_data_files("action.json")
        cls.GAZE = cls.load_data_files("gaze.json")
        cls.HANDS = cls.load_data_files("hands.json")
        cls.SCENE_INDOOR = cls.load_data_files("scene_indoor.json") # this json, each object has 3 properties (description, detail, preposition)
        cls.SCENE_OUTDOOR = cls.load_data_files("scene_outdoor.json") # this json, each object has 3 properties (description, detail, preposition)
        cls.LIGHTING = cls.load_data_files("lighting.json")       
        cls.TIME_OF_DAY = cls.load_data_files("time_of_day.json")
        cls.WEATHER = cls.load_data_files("weather.json")
        cls.CAMERA_OR_FILM = cls.load_data_files("camera_or_film.json")
        cls.PHOTOGRAPHER = cls.load_data_files("photographer.json")

        # used for random solection of colors, not selectable by user
        cls.COLORS = cls.load_data_files("color.json")

        # =========================================
        # SORTING LISTS
        # =========================================            
        sort_orders = cls.CONFIG_DATA["sort_orders"]

        if sort_orders['clothing_upper_sort_order'] == 'item':
            # sort by item as primary key
            cls.CLOTHING_UPPER.sort(key=lambda x: x['item'])
        elif sort_orders['clothing_upper_sort_order'] == 'category':
            # sort by category primary key, followed by item as secondary sort 
            cls.CLOTHING_UPPER.sort(key=lambda x: (x['category'], x['item']))

        if sort_orders['clothing_lower_sort_order'] == 'item':
            # sort by item as primary key
            cls.CLOTHING_LOWER.sort(key=lambda x: x['item'])
        elif sort_orders['clothing_lower_sort_order'] == 'category':
            # sort by category primary key, followed by item as secondary sort 
            cls.CLOTHING_LOWER.sort(key=lambda x: (x['category'], x['item']))
        
        if sort_orders['clothing_undergarment_sort_order'] == 'item':
            # sort by item as primary key
            cls.CLOTHING_UNDERGARMENT.sort(key=lambda x: x['item'])
        elif sort_orders['clothing_undergarment_sort_order'] == 'category':
            # sort by category primary key, followed by item as secondary sort 
            cls.CLOTHING_UNDERGARMENT.sort(key=lambda x: (x['category'], x['item']))

        if sort_orders['accessories_head_sort_order'] == 'item':
            # sort by item as primary key
            cls.ACCESSORIES_HEAD.sort(key=lambda x: x['item'])
        elif sort_orders['accessories_head_sort_order'] == 'category':
            # sort by category primary key, followed by item as secondary sort 
            cls.ACCESSORIES_HEAD.sort(key=lambda x: (x['category'], x['item']))

        if sort_orders['accessories_other_sort_order'] == 'item':
            # sort by item as primary key
            cls.ACCESSORIES_OTHER.sort(key=lambda x: x['item'])
        elif sort_orders['accessories_other_sort_order'] == 'category':
            # sort by category primary key, followed by item as secondary sort 
            cls.ACCESSORIES_OTHER.sort(key=lambda x: (x['category'], x['item']))


        # default sorts for scenes
        cls.SCENE_INDOOR.sort(key=lambda x: x['description'])
        cls.SCENE_OUTDOOR.sort(key=lambda x: x['description'])

    #  ==================================================================================
    @classmethod
    def load_config(cls):
        # Load the default config file
        base_dir = os.path.dirname(__file__)
        config_path = os.path.join(base_dir, "config.json")
        
        with open(config_path, "r") as config_file:
            config = json.load(config_file)

        return config
    
    #  ==================================================================================
    @classmethod
    def load_data_files(cls, file_name):
        # Define the path for default and custom files
        base_dir = os.path.dirname(__file__)
        default_path = os.path.join(base_dir, "data/default", file_name)
        custom_path = os.path.join(base_dir, "data/custom", file_name)

        # Load the default file
        with open(default_path, "r") as file:
            data = json.load(file)
        
        # Check if a custom variant of default file exists and merge it with the default
        if os.path.exists(custom_path):
            with open(custom_path, "r") as file:
                custom_data = json.load(file)
            # Assuming the data structure is a list
            data.extend(custom_data)

        return data
            
    #  ==================================================================================
    @classmethod
    def INPUT_TYPES(cls):
        cls.initialize_class_variables()  # Ensure variables are loaded

        main_settings = cls.CONFIG_DATA["main_settings"]

        return {
            "required": {
                "seed": ("INT", {"default": 123, "min": 0, "max": 1125899906842624}),
                "custom": ("STRING", {"default": f"{main_settings['custom']}"}),
                "style_and_framing": (
                    ["disabled", "random"] + cls.STYLE_AND_FRAMING,
                    {"default": f"{main_settings['style_and_framing']}"},            
                ),
                "subject_class": (
                    ["disabled", "random"] + cls.SUBJECT_CLASS,
                    {"default": f"{main_settings['subject_class']}"},              
                ),
                "role": (
                    ["disabled", "random"] + cls.ROLE,
                    {"default": f"{main_settings['role']}"},              
                ),
                # use a dictionary comprehension to extract only the "style_name" values for the list: 
                "hairstyle": (
                    ["disabled", "random"] + [hairstyle["style_name"] for hairstyle in cls.HAIRSTYLE],
                    {"default": f"{main_settings['hairstyle']}"},              
                ),
                # use a dictionary comprehension to extract only the "description" values for the list: 
                "body_shape": (
                    ["disabled", "random"] + [body_shape["description"] for body_shape in cls.BODY_SHAPE],
                    {"default": f"{main_settings['body_shape']}"},              
                ),                
                "use_random_clothing_color": (
                    "BOOLEAN", {"default": main_settings['use_random_clothing_color']}
                ),                 
                # use a dictionary comprehension to extract only the "item" values for the list: 
                "clothing_upper": (
                    ["disabled", "random"] + [clothing_upper["item"] for clothing_upper in cls.CLOTHING_UPPER],
                    {"default": f"{main_settings['clothing_upper']}"},  
                ),                
                # use a dictionary comprehension to extract only the "item" values for the list: 
                "clothing_lower": (
                    ["disabled", "random"] + [clothing_lower["item"] for clothing_lower in cls.CLOTHING_LOWER],
                    {"default": f"{main_settings['clothing_lower']}"}, 
                ),
                # use a dictionary comprehension to extract only the "item" values for the list: 
                "clothing_undergarment": (
                    ["disabled", "random"] + [clothing_underg["item"] for clothing_underg in cls.CLOTHING_UNDERGARMENT],
                    {"default": f"{main_settings['clothing_undergarment']}"}, 
                ),                
                # use a dictionary comprehension to extract only the "item" values for the list: 
                "footwear": (
                    ["disabled", "random"] + [footwear["item"] for footwear in cls.FOOTWEAR], 
                    {"default": f"{main_settings['footwear']}"}, 
                ),
                "accessories_head": (
                    ["disabled", "random"] + [accessories_head["item"] for accessories_head in cls.ACCESSORIES_HEAD],
                    {"default": f"{main_settings['accessories_head']}"}, 
                ),
                "accessories_other": (
                    ["disabled", "random"] + [accessories_other["item"] for accessories_other in cls.ACCESSORIES_OTHER] ,
                    {"default": f"{main_settings['accessories_other']}"}, 
                ),                
                "action": (
                    ["disabled", "random"] + cls.ACTION,
                    {"default": f"{main_settings['action']}"}, 
                ),    
                "gaze": (
                    ["disabled", "random"] + cls.GAZE,
                    {"default": f"{main_settings['gaze']}"}, 
                ),
                "hands": (
                    ["disabled", "random"] + cls.HANDS,
                    {"default": f"{main_settings['hands']}"}, 
                ),   
                "show_detailed_location": (
                    # {"default": f"{main_settings['show_detailed_location']}"}, 
                    "BOOLEAN", {"default": main_settings['show_detailed_location']}, 
                ),                    
                # use a dictionary comprehension to extract only the "description" values for the list: 
                "scene_indoor": (
                    ["disabled", "random"] + [location["description"] for location in cls.SCENE_INDOOR],
                    {"default": f"{main_settings['scene_indoor']}"}, 
                ),
                # use a dictionary comprehension to extract only the "description" values for the list: 
                "scene_outdoor": (
                    ["disabled", "random"] + [location["description"] for location in cls.SCENE_OUTDOOR],
                    {"default": f"{main_settings['scene_outdoor']}"}, 
                ),           
                "lighting": (
                    ["disabled", "random"] + cls.LIGHTING,
                    {"default": f"{main_settings['lighting']}"}, 
                ),                  
                "time_of_day": (
                    ["disabled", "random"] + cls.TIME_OF_DAY,
                    {"default": f"{main_settings['time_of_day']}"}, 
                ),  
                "weather": (
                    ["disabled", "random"] + cls.WEATHER,
                    {"default": f"{main_settings['weather']}"}, 
                ),   
                "camera_or_film": (
                    ["disabled", "random"] + cls.CAMERA_OR_FILM,
                    {"default": f"{main_settings['camera_or_film']}"}, 
                ),   
                "photographer": (
                    ["disabled", "random"] + cls.PHOTOGRAPHER,
                    {"default": f"{main_settings['photographer']}"}, 
                ),   
                "remove_commas_periods": (
                    "BOOLEAN", {"default": main_settings['remove_commas_periods']}, 
                ),                   
            },
        }

    RETURN_TYPES = (
        "STRING",
        # "INT",
        # "STRING",
        # "STRING",
        # "STRING"
    )
    RETURN_NAMES = (
        "Full Prompt",
        # "seed",
        # "t5xxl",
        # "clip_l",
        # "clip_g",
    )    

    FUNCTION = "generate_prompt"
    CATEGORY = "Y7/PromptGenerator"
    DESCRIPTION = """
## Description
"""
    
    def generate_prompt(self, **kwargs):
        # get seed. if not provided then default to 0
        seed = kwargs.get("seed", 0)
        # self.rng set to a new instance of random.Random, init with the seed value
        self.rng = random.Random(seed)
        print(f'seed = {seed}')
        # components hold all the separate prompt parts together
        components = []

        # ------------------------------------------------------------        
        # CUSTOM
        # ------------------------------------------------------------
        # get custom string, if nothing is passed in, default to ""
        custom = kwargs.get("custom", "")
        # if not empty, append custom string to components list
        if custom != "":
            components.append(f'{custom},')
                           
        # ------------------------------------------------------------
        # STYLE_AND_FRAMING 
        # ------------------------------------------------------------        
        style_and_framing = kwargs.get("style_and_framing", "disabled")
        if style_and_framing == self.DISABLED:
            style_and_framing = ""
        elif style_and_framing == self.RANDOM:
            style_and_framing = self.select_random_choice(self.STYLE_AND_FRAMING)      
        
        if style_and_framing: # if not empty
            # if framing string begins with a vowel
            if self.begins_with_vowel(style_and_framing) == True:
                # set either "a" or "an"
                components.append(f'an')  
            else:
                components.append(f'a')  

            # now add framing 
            components.append(f'{style_and_framing} photo of')  

        # ------------------------------------------------------------
        # SUBJECT / CLASS
        # ------------------------------------------------------------
        subject_or_class = kwargs.get("subject_class", "a man")
        if subject_or_class == self.DISABLED:
            subject_or_class = ""
        elif subject_or_class == self.RANDOM:
            subject_or_class = self.select_random_choice(self.SUBJECT_CLASS)  

        
        if subject_or_class:       
            components.append(f'{subject_or_class}')          
        # ------------------------------------------------------------
        # ROLE
        # ------------------------------------------------------------
        role = kwargs.get("role", "random")
        if role == self.DISABLED:
            role = ""
        elif role == self.RANDOM:
            role = self.select_random_choice(self.ROLE)    

        if role:
            article = 'an' if self.begins_with_vowel(role) else 'a'
            components.append(f'as {article} {role}') 

        # ------------------------------------------------------------
        # HAIR STYLE
        # ------------------------------------------------------------
        hairstyle = kwargs.get("hairstyle", "random")
        if hairstyle == self.DISABLED:
            hairstyle = ""
        elif hairstyle == self.RANDOM:
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
        # ------------------------------------------------------------
        body_shape = kwargs.get("body_shape", "random")
        if body_shape == self.DISABLED:
            body_shape = ""
        elif body_shape == self.RANDOM:
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
        # GET RANDOM COLOR FLAG
        # ------------------------------------------------------------
        use_random_clothing_color = bool(kwargs.get("use_random_clothing_color", False))
        
        # ------------------------------------------------------------
        # CLOTHING UPPER
        # ------------------------------------------------------------
        clothing_upper = kwargs.get("clothing_upper", "random") # get clothing selection, default to random if none found.
        if clothing_upper == self.DISABLED:
            clothing_upper = ""
        elif clothing_upper == self.RANDOM:
            # if set to random, we will lock it to the gender so it's less crazy              
            if self.subject_is_man_or_woman(subject_or_class) == SUBJECT_TYPE.MAN:
                CLOTHING_UPPER_FILTERED = self.filter_clothing_by_gender(SUBJECT_TYPE.MAN, self.CLOTHING_UPPER)
                clothing_upper = self.select_random_choice(CLOTHING_UPPER_FILTERED)    
            elif self.subject_is_man_or_woman(subject_or_class) == SUBJECT_TYPE.WOMAN:
                CLOTHING_UPPER_FILTERED = self.filter_clothing_by_gender(SUBJECT_TYPE.WOMAN, self.CLOTHING_UPPER) 
                clothing_upper = self.select_random_choice(CLOTHING_UPPER_FILTERED)      
            else: # select random from full list                 
                clothing_upper = self.select_random_choice(self.CLOTHING_UPPER)          
        else:
            # if user has made selection from the full list, then we only have 'item' property, 
            # not the whole obj that also contains the "default_color" attribute, so
            # we the whole selected location object based on the "item" attribute
            for cloth in self.CLOTHING_UPPER:
                if cloth["item"] == clothing_upper:
                    clothing_upper = cloth
                    break

        if clothing_upper: # if not an empty
            item = clothing_upper.get('item')  # defaults to 'none' if key does not exist
            color = ""
            if use_random_clothing_color:
                color = self.select_random_choice(self.COLORS)        
            else:
                # use default color associated with the item of clothing
                color = clothing_upper.get('default_color')  # defaults to 'none' if key does not existpass
            
            article = 'an' if self.begins_with_vowel(color) else 'a'
            clothing_string = f'wearing {article} {color} {item}'
            components.append(f'{clothing_string},')      
        # ------------------------------------------------------------
        # CLOTHING LOWER 
        clothing_lower = kwargs.get("clothing_lower", "random")
        if clothing_lower == self.DISABLED:
            clothing_lower = ""
        elif clothing_lower == self.RANDOM:
            # first look to see if the clothing_upper is a dress or one-piece - if so we will ignore clothing_lower
            if clothing_upper['category'] in ('dresses', 'one-piece', 'robes'):
                clothing_lower = ""

            else:
                # if set to random, we will lock it to the gender so it's less crazy              
                if self.subject_is_man_or_woman(subject_or_class) == SUBJECT_TYPE.MAN:
                    CLOTHING_FILTERED = self.filter_clothing_by_gender(SUBJECT_TYPE.MAN, self.CLOTHING_LOWER)
                    clothing_lower = self.select_random_choice(CLOTHING_FILTERED)    
                elif self.subject_is_man_or_woman(subject_or_class) == SUBJECT_TYPE.WOMAN:
                    CLOTHING_FILTERED = self.filter_clothing_by_gender(SUBJECT_TYPE.WOMAN, self.CLOTHING_LOWER) 
                    clothing_lower = self.select_random_choice(CLOTHING_FILTERED)      
                else: # select random from full list                                 
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
            color = ""
            if use_random_clothing_color:
                color = self.select_random_choice(self.COLORS)        
            else:
                # use default color associated with the item of clothing
                color = clothing_lower.get('default_color')  # defaults to 'none' if key does not existpass

            article = ""
            # if no upper clothing chosen, start with wearing and article 'an'/'a'
            if not clothing_upper:
                article = 'an' if self.begins_with_vowel(color) else 'a'
                clothing_string = f'wearing {article} {color} {item}' 
            else:
                clothing_string = f'{color} {item}' 

            components.append(f'{clothing_string},')   
        # ------------------------------------------------------------     
        # CLOTHING - UNDERGARMENTS & HOSIERY ETC
        clothing_undergarment_selected = kwargs.get("clothing_undergarment", random)
        if clothing_undergarment_selected == self.DISABLED:
            clothing_undergarment_selected = ""
        elif clothing_undergarment_selected == self.RANDOM:        
            # if set to random, we will filter out some items according to what lower clothing item has been selected
            # IF CLOTHING_LOWER = "" : ALLOW ALL ITEMS
            # IF CLOTHING_LOWER = PANTS, LEGGINGS, OTHER : ALLOW NONE
            # IF CLOTHING_LOWER = SKIRTS : ALLOW SOCKS OR HOSIERY ONLY            
            if clothing_lower == "":
                # first check if clothing_upper is the reason lower is empty....
                if clothing_upper['category'] in ('dresses', 'one-piece', 'robes'):
                    UNDERGARMENT_FILTERED = self.filter_undergarments_hosiery(self.CLOTHING_UNDERGARMENT)
                    clothing_undergarment_selected = self.select_random_choice(UNDERGARMENT_FILTERED)   
                else:                    
                    # allow all
                    clothing_undergarment_selected = self.select_random_choice(self.CLOTHING_UNDERGARMENT) 

            elif clothing_lower['category'] == "skirts":
                # filter list to socks and hosiery only
                UNDERGARMENT_FILTERED = self.filter_undergarments_hosiery(self.CLOTHING_UNDERGARMENT)
                clothing_undergarment_selected = self.select_random_choice(UNDERGARMENT_FILTERED)   
            else:
                clothing_undergarment_selected = ""
        else:
            # if user has made selection, then we only have 'item' property, not the whole obj that also contains the "default_color" attribute, so
            # we the whole selected location object based on the "item" attribute
            for underg in self.CLOTHING_UNDERGARMENT:
                if underg["item"] == clothing_undergarment_selected:
                    clothing_undergarment_selected = underg
                    break              

        if clothing_undergarment_selected: # if not empty

            item = clothing_undergarment_selected.get('item')  # defaults to 'none' if key does not exist
            color = ""
            if use_random_clothing_color:
                color = self.select_random_choice(self.COLORS)        
            else:
                # use default color associated with the item of clothing
                color = use_random_clothing_color.get('default_color')  # defaults to 'none' if key does not existpass

            article = ""
            # if no upper clothing chosen, start with wearing and article 'an'/'a'
            if not clothing_upper and not clothing_lower:
                article = 'an' if self.begins_with_vowel(color) else 'a'
                clothing_string = f'wearing {article} {color} {item}' 
            else:
                clothing_string = f'{color} {item}' 

            print(f'clothing_string = {clothing_string}')
            components.append(f'{clothing_string},')   
        # ------------------------------------------------------------              

        # ------------------------------------------------------------       
        # FOOTWEAR
        # ------------------------------------------------------------     
        footwear = kwargs.get("footwear", "random")
        if footwear == self.DISABLED:
            footwear = ""
        elif footwear == self.RANDOM:
            # if set to random, we will lock it to the gender so it's less crazy              
            if self.subject_is_man_or_woman(subject_or_class) == SUBJECT_TYPE.MAN:
                FOOTWEAR_FILTERED = self.filter_clothing_by_gender(SUBJECT_TYPE.MAN, self.FOOTWEAR)
                footwear = self.select_random_choice(FOOTWEAR_FILTERED)    
            elif self.subject_is_man_or_woman(subject_or_class) == SUBJECT_TYPE.WOMAN:
                FOOTWEAR_FILTERED = self.filter_clothing_by_gender(SUBJECT_TYPE.WOMAN, self.FOOTWEAR) 
                footwear = self.select_random_choice(FOOTWEAR_FILTERED)      
            else: # select random from full list                                              
                footwear = self.select_random_choice(self.FOOTWEAR) 
        else: 
            # if user has made selection, then we only have 'item' property, not the whole obj that also contains the "default_color" attribute, so
            # we the whole selected location object based on the "item" attribute
            for ftwear in self.CLOTHING_LOWER:
                if footwear["item"] == ftwear:
                    footwear = ftwear
                    break         
        
        if footwear:  # if not an empty   
            item = footwear.get('item')  # defaults to 'none' if key does not exist
            color = ""
            if use_random_clothing_color:
                color = self.select_random_choice(self.COLORS)        
            else:
                # use default color associated with the item of clothing
                color = footwear.get('default_color')  # defaults to 'none' if key does not existpass
            article = ""
            # if no upper clothing AND no lower clothing and no undergarments
            if not clothing_upper and not clothing_lower and not clothing_undergarment_selected:
                article = 'an' if self.begins_with_vowel(item) else 'a'
                footwear_string = f'wearing {article} {color} {item}' 
            else:
                footwear_string = f'{color} {item}' 
            components.append(f'{footwear_string},')            
        # ------------------------------------------------------------
        # ACCESSORIES - HEAD
        accessories_head = kwargs.get("accessories_head", "random")
        if accessories_head == self.DISABLED:
            accessories_head = ""
        elif accessories_head == self.RANDOM:
            accessories_head = self.select_random_choice(self.ACCESSORIES_HEAD)      
        else: 
            # if user has made selection, then we only have 'item' property, not the whole obj that also contains the "default_color" attribute, so
            # we the whole selected location object based on the "item" attribute
            for acc in self.ACCESSORIES_HEAD:
                if accessories_head["item"] == acc:
                    accessories_head = acc
                    break         

        if accessories_head: #if not empty
            item = accessories_head.get('item')
            color = ""
            if use_random_clothing_color:
                color = self.select_random_choice(self.COLORS)        
            else:
                # use default color associated with the item of clothing
                color = accessories_head.get('default_color')  # defaults to 'none' if key does not existpass
            article = ""
            # if no upper clothing AND no lower clothing and no undergarments and no footwear
            if not clothing_upper and not clothing_lower and not clothing_undergarment_selected and not footwear:
                article = 'an' if self.begins_with_vowel(item) else 'a'
                accessory_head_string = f'wearing {article} {color} {item}' 
            else:
                accessory_head_string = f'{color} {item}' 
            components.append(f'{accessory_head_string},')     

        # ------------------------------------------------------------
        # ACCESSORIES - OTHER
        accessories_other = kwargs.get("accessories_other", "random")
        if accessories_other == self.DISABLED:
            accessories_other = ""
        elif accessories_other == self.RANDOM:
            accessories_other = self.select_random_choice(self.ACCESSORIES_OTHER)      
        else: 
            # if user has made selection, then we only have 'item' property, not the whole obj that also contains the "default_color" attribute, so
            # we the whole selected location object based on the "item" attribute
            for acc in self.ACCESSORIES_OTHER:
                if accessories_other["item"] == acc:
                    accessories_other = acc
                    break         
                        
        if accessories_other: #if not empty
            item = accessories_other.get('item')
            color = ""
            if use_random_clothing_color:
                color = self.select_random_choice(self.COLORS)        
            else:
                # use default color associated with the item of clothing
                color = accessories_other.get('default_color')  # defaults to 'none' if key does not existpass
            article = ""
            # if no upper clothing AND no lower clothing and no undergarments and no footwear
            if not clothing_upper and not clothing_lower and not clothing_undergarment_selected and not footwear:
                article = 'an' if self.begins_with_vowel(item) else 'a'
                accessory_other_string = f'wearing {article} {color} {item}' 
            else:
                accessory_other_string = f'{color} {item}' 
            components.append(f'{accessory_other_string}.')    

        # ------------------------------------------------------------
        # MAIN ACTION
        action = kwargs.get("action", "random")
        if action == self.DISABLED:
            action = ""
        elif action == self.RANDOM:
            action = self.select_random_choice(self.ACTION)   

        # Only append if truthy (is not an empty string) 
        if action:
            if self.subject_is_man_or_woman(subject_or_class) == SUBJECT_TYPE.MAN:
                components.append("He is")
            elif self.subject_is_man_or_woman(subject_or_class) == SUBJECT_TYPE.WOMAN:
                components.append("She is")
            else:
                components.append("It is")
            
            components.append(f'{action},') 
        # ------------------------------------------------------------
        # GAZE
        gaze = kwargs.get("gaze", "random")
        if gaze == self.DISABLED:
            gaze = ""
        elif gaze == self.RANDOM:
            gaze = self.select_random_choice(self.GAZE)   

        # Only append if truthy (is not an empty string) 
        if gaze:
            if action == "": # if no action, add pronoun prefix
                if self.subject_is_man_or_woman(subject_or_class) == SUBJECT_TYPE.MAN:
                    components.append("He is")
                elif self.subject_is_man_or_woman(subject_or_class) == SUBJECT_TYPE.WOMAN:
                    components.append("She is")
                else:
                    components.append("It is")    

            components.append(f'{gaze},') 

        # ------------------------------------------------------------
        # HANDS
        hands = kwargs.get("hands", "random")
        if hands == self.DISABLED:
            hands = ""
        elif hands == self.RANDOM:
            hands = self.select_random_choice(self.HANDS)   

        # Only append if truthy (is not an empty string) 
        if hands:
            if action == "" and gaze == "" : # if no action and no gaze, add pronoun prefix
                if self.subject_is_man_or_woman(subject_or_class) == SUBJECT_TYPE.MAN:
                    components.append("His")
                elif self.subject_is_man_or_woman(subject_or_class) == SUBJECT_TYPE.WOMAN:
                    components.append("Her")
                else:
                    components.append("Its")
            else:                
                components.append(f'with') 

            components.append(f'{hands}.') 
        # ------------------------------------------------------------
        # SHOW DETAILED LOCATION DESCRIPTION (OR NOT)
        show_detailed_location = bool(kwargs.get("show_detailed_location", True))

        # ------------------------------------------------------------
        # LOCATION - INTERIOR
        # get initial selection froom drop down
        scene_indoor = kwargs.get("scene_indoor", "random")        
        if scene_indoor == self.DISABLED:
            scene_indoor = ""
        elif scene_indoor == self.RANDOM:
            scene_indoor = self.select_random_choice(self.SCENE_INDOOR)   
        else: # else is a selected value                    
            # Find the whole selected location object based on the description
            for loc in self.SCENE_INDOOR:
                if loc["description"] == scene_indoor:
                    scene_indoor = loc # get whol object
                    break

        # Only append if truthy (is not an empty) 
        if scene_indoor:
            if re.search(r'\bman\b', subject_or_class.lower()):
                components.append("He is")
            elif re.search(r'\bwoman\b', subject_or_class.lower()):
                components.append("She is")

            if show_detailed_location:
                location_string = f'{scene_indoor["preposition"]} {scene_indoor["description"]}, {scene_indoor["detail"]}'
            else:
                location_string = f'{scene_indoor["preposition"]} {scene_indoor["description"]}'
            components.append(f'{location_string},') 

        # ELSE IF INT LOCATION IS EMPTY, THEN WE LOOK AT EXTERNAL LOCATION
        else: # LOCATION - EXT
            print('interior is empty - looking at exterior....')
            # get initial selection froom drop down
            scene_outdoor = kwargs.get("scene_outdoor", "random")
            if scene_outdoor == self.DISABLED:
                scene_outdoor = ""
            elif scene_outdoor == self.RANDOM:
                scene_outdoor = self.select_random_choice(self.SCENE_OUTDOOR)   
            else:
                # Find the whole selected location object based on the description
                for loc in self.SCENE_OUTDOOR:
                    if loc["description"] == scene_outdoor:
                        scene_outdoor = loc
                        break

            # Only append if truthy (is not an empty string) 
            if scene_outdoor:
                if re.search(r'\bman\b', subject_or_class.lower()):
                    components.append("He is")
                elif re.search(r'\bwoman\b', subject_or_class.lower()):
                    components.append("She is")

                if show_detailed_location:
                    location_string = f'{scene_outdoor["preposition"]} {scene_outdoor["description"]}, {scene_outdoor["detail"]}'
                else:
                    location_string = f'{scene_outdoor["preposition"]} {scene_outdoor["description"]}'     

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
            if scene_indoor:
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
        # REMOVE COMMAS AND PERIODS
        remove_commas_periods = bool(kwargs.get("remove_commas_periods", False))

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
        
    # ==============================================================================================================  
    def subject_is_man_or_woman(self, subject_or_class):
        # Check if the subject or class description contains the word 'man'
        if re.search(r'\bman\b', subject_or_class.lower()):
            return SUBJECT_TYPE.MAN
        # Check if the subject or class description contains the word 'woman'
        elif re.search(r'\bwoman\b', subject_or_class.lower()):
            return SUBJECT_TYPE.WOMAN
        else:
            return SUBJECT_TYPE.OTHER   

    # ==============================================================================================================  
    def filter_clothing_by_gender(self, st: SUBJECT_TYPE, clothing_list):
        filtered_items = []  # Create an empty list to store the filtered items

        for item in clothing_list:  # Loop through each item in the class variable
        
            if st == SUBJECT_TYPE.MAN:
                # if male or unisex then is good
                if item['gender'] in ('male', 'unisex'):
                    filtered_items.append(item)  # Add the item to the filtered list
            elif st == SUBJECT_TYPE.WOMAN:
                # if female or unisex then is good
                if item['gender'] in ('female', 'unisex'):
                    filtered_items.append(item)  # Add the item to the filtered list

                    
        return filtered_items

    # ==============================================================================================================  
    def filter_undergarments_hosiery(self, undergarments_list):
        filtered_items = []  

        # allow stockings and socks only        
        for item in undergarments_list:  # Loop through each item in the class variable                
            if item['category'] in ("hosiery", "socks"):
                filtered_items.append(item)


        return filtered_items


NODE_CLASS_MAPPINGS = {
    "PhotoPrompter_(Y7)": PhotoPromptGenerator
}

# Human readable names for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "PhotoPrompter_(Y7)": "Photo Prompter (Y7)"
}        