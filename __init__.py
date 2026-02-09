# from .nodes.brightness import Y7Nodes_Brightness
# from .nodes.template_node import Y7_TemplateNode
from .nodes.documentation import format_descriptions
from .nodes.text import Y7Nodes_Text
from .nodes.grid2batch import Y7Nodes_Grid2Batch
from .nodes.show_anything import Y7Nodes_ShowAnything
from .nodes.prompt_enhancer_flux import Y7Nodes_PromptEnhancerFlux
from .nodes.prompt_enhancer_flux2_klein import Y7Nodes_PromptEnhancerFlux2
from .nodes.t5_token_count import Y7Nodes_T5_TokenCounter
from .nodes.clip_token_count import Y7Nodes_CLIP_TokenCounter
from .nodes.catch_edit_text_dual import Y7Nodes_CatchEditTextNodeDual
from .nodes.image_row import Y7Nodes_ImageRow
from .nodes.image_size_presets import Y7Nodes_ImageSizePresets
from .nodes.smolvlm import Y7Nodes_SmolVLM
from .nodes.crop_to_nearest_multiple import Y7Nodes_CropToNearestMultiple
from .nodes.color_match_masked import Y7Nodes_ColorMatchMasked

NODE_CLASS_MAPPINGS = {
    "Y7Nodes_ImageRow": Y7Nodes_ImageRow,
    "Y7Nodes_ImageSizePresets": Y7Nodes_ImageSizePresets,
    "Y7Nodes_Text": Y7Nodes_Text,
    "Y7Nodes_ShowAnything": Y7Nodes_ShowAnything,
    "Y7Nodes_Grid2Batch": Y7Nodes_Grid2Batch,    
    "Y7Nodes_PromptEnhancerFlux": Y7Nodes_PromptEnhancerFlux,
    "Y7Nodes_PromptEnhancerFlux2": Y7Nodes_PromptEnhancerFlux2,
    "Y7Nodes_T5_TokenCounter": Y7Nodes_T5_TokenCounter,
    "Y7Nodes_CLIP_TokenCounter": Y7Nodes_CLIP_TokenCounter,
    "Y7Nodes_CatchEditTextNodeDual": Y7Nodes_CatchEditTextNodeDual,
    "Y7Nodes_SmolVLM": Y7Nodes_SmolVLM,
    "Y7Nodes_CropToNearestMultiple": Y7Nodes_CropToNearestMultiple,
    "Y7Nodes_ColorMatchMasked": Y7Nodes_ColorMatchMasked
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Y7Nodes_ImageRow": "Y7 Image Row",
    "Y7Nodes_ImageSizePresets" : "Y7 Image Size (Presets)",
    "Y7Nodes_Text": "Y7 Text",
    "Y7Nodes_ShowAnything": "Y7 Show Anything",
    "Y7Nodes_Grid2Batch": "Y7 Grid to Batch",
    "Y7Nodes_PromptEnhancerFlux": "Y7 Prompt Enhancer (Flux1)",
    "Y7Nodes_PromptEnhancerFlux2": "Y7 Prompt Enhancer (Flux2)",
    "Y7Nodes_T5_TokenCounter": "Y7 T5 Token Counter",
    "Y7Nodes_CLIP_TokenCounter": "Y7 CLIP Token Counter",    
    "Y7Nodes_CatchEditTextNodeDual": "Y7 Catch and Edit Text (Dual)",
    "Y7Nodes_SmolVLM": "Y7 SmolVLM",
    "Y7Nodes_CropToNearestMultiple": "Y7 Crop to Nearest Multiple",
    "Y7Nodes_ColorMatchMasked": "Y7 Color Match (Masked)"
}

# Apply HTML-formatted documentation to node classes
# This sets the DESCRIPTION and description properties for each node class
# using the predefined descriptions in documentation.py
format_descriptions(NODE_CLASS_MAPPINGS)

# Make sure these are accessible to ComfyUI
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# Tell ComfyUI where to find the web/JavaScript files
WEB_DIRECTORY = "./web"

#  =============================================================================
# these imports are for server route registration
from aiohttp import web
from server import PromptServer
from pathlib import Path
import os
import json

# Add server routes if PromptServer is available
if hasattr(PromptServer, "instance") and not PromptServer.instance.app.frozen:
    # detect directory name at runtime since it could be different from the 
    # Repo-name depending on whether installed via the ComfyUI Manager or a git clone
    # Get the actual directory name (last part of the absolute path)
    current_dir = Path(__file__).parent.absolute()
    dir_name = os.path.basename(current_dir)
    # print(f"{dir_name}")


    # Register routes using the actual directory name
    PromptServer.instance.app.add_routes([
        # Route for JavaScript files
        web.static(f"/{dir_name}", (current_dir / "web").as_posix())


    # # Register routes for web assets and data files
    # PromptServer.instance.app.add_routes([
    #     # Route for JavaScript files
    #     web.static("/ComfyUI-Y7Nodes", (Path(__file__).parent.absolute() / "web").as_posix())

    ])
