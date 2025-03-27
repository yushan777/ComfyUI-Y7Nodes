# This is just a demo custom node
from .documentation import descriptions, as_html

class Y7Nodes_Brightness:
    
    def __init__(self):
        # Define default values
        pass
    
    # ====================================================
    # Defines what inputs your node accepts
    @classmethod  # This decorator makes it a class method instead of instance method
    def INPUT_TYPES(cls):  # cls is the class itself, like ExampleNode
        return {
            "required": {  # These inputs must be provided
                # Simple input type
                "image": ("IMAGE",),  # Tuple with trailing comma
                
                # Complex input type with configuration
                "strength": ("FLOAT", {  # Type is FLOAT
                    "default": 1.0,      # Initial value shown
                    "min": 0.0,         # Minimum allowed value
                    "max": 3.0,         # Maximum allowed value
                    "step": 0.01        # Slider/input step size
                }),
            },
           "optional": {
               "filename_prefix": ("STRING", {
                   "default": "",
                   "multiline": False,  # Single line text input
                   "description": "filename prefix that includes brightness strength"  # Tooltip in UI
               })
           }                  
        }
    
    # Return types (tuple). Specifies what type of data your node outputs 
    # (note the trailing comma, without it, it would not be a tuple, esp. for single item)
    RETURN_TYPES = ("IMAGE","STRING",)
    # Name of the method that will be called
    FUNCTION = "process_image"
    # Determines where your node appears in the UI
    CATEGORY = "Y7Nodes/Image/Processing"
    
    #  ==========================================
    # Main function of this node 
    # Optional parameters must have default values
    def process_image(self, image, strength, filename_prefix=""):  
        # processed image
        processed_image = image * strength
        
        # Use the string for a filename prefix for when saving. 
        if filename_prefix:
           fname_prefix = f"{filename_prefix}_{strength}"

        # return a tuple with a single item...
        return (processed_image,fname_prefix,)
