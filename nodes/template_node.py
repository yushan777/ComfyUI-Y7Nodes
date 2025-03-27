import os
import torch
# import numpy as np
# from PIL import Image


# ===========================================================
class AlwaysEqualProxy(str):
    # AlwaysEqualProxy returns True for all equality checks and False for all inequality checks

    def __eq__(self, _):
        # Always True for == operations
        return True

    def __ne__(self, _):
        # Always False for != operations
        return False

# Wildcard that matches any type
any_type = AlwaysEqualProxy("*")

# ===========================================================
class Y7_TemplateNode:
    """
    A template node for ComfyUI with examples of all common input types
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Basic types
                # basic types when forceInput is true weirdly places the input dot below any output dots
                # to place it at top, instead of "STRING", just give it a random name "CHEESE"
                "string_input1": ("STRING", {"default": "default text", "forceInput": True}),
                "int_input": ("INT", {"default": 0, "min": -10, "max": 100, "step": 1}),
                "float_input": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "boolean_input": ("BOOLEAN", {"default": True}),
                
                # ComfyUI specific types
                # these types will always come first in terms of dots
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "latent": ("LATENT",),
                "conditioning": ("CONDITIONING",),
                
                # custom type
                "anything": (any_type, {}),
                "string_input2": ("CHEESE", {"default": "default text", "forceInput": True}),

                # Options/dropdown selection
                "dropdown": (["option1", "option2", "option3"], {"default": "option1"}),
                
                # File/directory selection
                "file_path": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                # Optional inputs
                "optional_string": ("STRING", {"default": ""}),
                "optional_int": ("INT", {"default": 0}),
                # "optional_image": ("IMAGE",),
                
                # Force a widget to require a connection
                "forced_input": ("STRING", {"forceInput": True}),
            },
            "hidden": {
                # Hidden inputs for workflow state persistence
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }
    
    RETURN_TYPES = ("STRING", "INT", "IMAGE", "LATENT")
    RETURN_NAMES = ("string_output", "int_output", "image_output", "latent_output")
    
    FUNCTION = "process"
    CATEGORY = "Y7Nodes"
    
    def process(self, 
                # Required inputs
                string_input, int_input, float_input, boolean_input,
                model, clip, vae, image, latent, conditioning,
                dropdown, file_path,
                # Optional inputs (with defaults to prevent errors when not connected)
                optional_string=None, optional_int=None, optional_image=None,
                forced_input="",
                # Hidden inputs
                unique_id=None, extra_pnginfo=None):
        """
        Main processing function for the node
        """
        # Your processing logic here
        print(f"Processing with string: {string_input}")
        print(f"Selected dropdown option: {dropdown}")
        
        # Example of checking optional inputs
        if optional_image is not None:
            print("Optional image was provided")
        
        # Create example return values (replace with actual processing)
        output_string = f"Processed: {string_input}"
        output_int = int_input * 2
        
        # Pass-through values as examples
        output_image = image
        output_latent = latent
        
        # Return values must match RETURN_TYPES order
        return (output_string, output_int, output_image, output_latent)
