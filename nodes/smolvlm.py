import torch

class Y7Nodes_SmolVLM:
    """
    A skeleton node for SmolVLM integration in ComfyUI.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (["SmolVLM-Instruct", "SmolVLM-500M-Instruct", "SmolVLM-256M-Instruct"], {"default": "SmolVLM-256M-Instruct"}),
                "image": ("IMAGE",),  # Added image input for VLM
                "query": ("STRING", {"default": "Caption this image", "multiline": True}),  # Changed to STRING with multiline for text box
                "max_new_tokens": ("INT", {"default": 256, "min": 1, "max": 1024, "step": 1}),
                "rep_penalty": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 5.0, "step": 0.01}),
                "do_sample": ("BOOLEAN", {"default": False}),
                "temperature": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string_out",)
    
    FUNCTION = "process"
    CATEGORY = "Y7Nodes/VLM"
    
    def process(self, model, image, query, max_new_tokens, rep_penalty, do_sample, temperature, top_p):  # Added image parameter
        """
        Main processing function for the SmolVLM node.
        """
        # Placeholder processing logic
        info = (
            f"SmolVLM Node Inputs:\n"
            f"Model: {model}\n"
            f"Query: {query}\n"
            f"Max New Tokens: {max_new_tokens}\n"
            f"Repetition Penalty: {rep_penalty}\n"
            f"Do Sample: {do_sample}\n"
            f"Temperature: {temperature}\n"
            f"Top P: {top_p}\n"
            f"Image shape: {image.shape if image is not None else 'None'}"
        )
        
        print(info)  # For debugging in the console
        
        return (info,)


# Required mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "Y7Nodes_SmolVLM": Y7Nodes_SmolVLM
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Y7Nodes_SmolVLM": "SmolVLM Image Captioning"
}