import os

# ===========================================================
class Y7Nodes_Text:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
            },
            "optional": {
                "text": ("STRING", {"default": "", "multiline": True}),
            },
            "hidden": {
                # # Hidden inputs for workflow state persistence
                # "unique_id": "UNIQUE_ID",
                # "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text_out",)    
    FUNCTION = "process"
    CATEGORY = "Y7Nodes/Utils"
    
    def process(self, **kwargs):
        
        text = kwargs.get("text")

        # Return values must match RETURN_TYPES order
        return (text,)
