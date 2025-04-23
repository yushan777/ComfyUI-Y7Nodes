import os

# ===========================================================
class Y7Nodes_Text:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
            },
            "optional": {
                "text": ("STRING", {"default": "", "multiline": True, "tooltip": "Text input that will be passed through to the output"}),
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
