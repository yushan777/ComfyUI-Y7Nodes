# from .nodes.brightness import Y7Nodes_Brightness
# from .nodes.template_node import Y7_TemplateNode
from .nodes.documentation import format_descriptions
from .nodes.text import Y7Nodes_Text
from .nodes.show_anything import Y7Nodes_ShowAnything
from .nodes.prompt_enhancer_flux import Y7Nodes_PromptEnhancerFlux
from .nodes.prompt_enhancer_flux2_klein import Y7Nodes_PromptEnhancerFlux2
from .nodes.t5_token_count import Y7Nodes_T5_TokenCounter
from .nodes.clip_token_count import Y7Nodes_CLIP_TokenCounter
from .nodes.catch_edit_text_dual import Y7Nodes_CatchEditTextNodeDual
from .nodes.image_size_presets import Y7Nodes_ImageSizePresets
from .nodes.crop_to_nearest_multiple import Y7Nodes_CropToNearestMultiple
from .nodes.color_match_masked import Y7Nodes_ColorMatchMasked
from .nodes.aspect_ratio_picker import Y7Nodes_AspectRatioPicker
from .nodes.lm_studio import Y7Nodes_LMStudioText, Y7Nodes_LMStudioVision, Y7Nodes_SelectLMSModel
from .nodes.qwen3_vl import Y7Nodes_QwenVL
from .nodes.sampler_select_name import SamplerSelect_Name
from .nodes.paste_cropped_image_back import Y7Nodes_PasteCroppedImageBack
from .nodes.scale_image_to_total_pixels import Y7Nodes_ScaleImageToTotalPixels
from .nodes.scale_image_by import Y7Nodes_ScaleImageBy
from .nodes.joycaption import Y7Nodes_JoyCaption, Y7Nodes_JoyCaption_ExtraOptions
from .nodes.caption_tools import Y7Nodes_ImageBatchPath, Y7Nodes_CaptionSaver
from .nodes.load_image import Y7Nodes_LoadImage
from .nodes.image_stitcher import Y7Nodes_ImageStitcher

NODE_CLASS_MAPPINGS = {
    "Y7Nodes_ImageSizePresets": Y7Nodes_ImageSizePresets,
    "Y7Nodes_Text": Y7Nodes_Text,
    "Y7Nodes_ShowAnything": Y7Nodes_ShowAnything,
    "Y7Nodes_PromptEnhancerFlux": Y7Nodes_PromptEnhancerFlux,
    "Y7Nodes_PromptEnhancerFlux2": Y7Nodes_PromptEnhancerFlux2,
    "Y7Nodes_T5_TokenCounter": Y7Nodes_T5_TokenCounter,
    "Y7Nodes_CLIP_TokenCounter": Y7Nodes_CLIP_TokenCounter,
    "Y7Nodes_CatchEditTextNodeDual": Y7Nodes_CatchEditTextNodeDual,
    "Y7Nodes_CropToNearestMultiple": Y7Nodes_CropToNearestMultiple,
    "Y7Nodes_ColorMatchMasked": Y7Nodes_ColorMatchMasked,
    "Y7Nodes_AspectRatioPicker": Y7Nodes_AspectRatioPicker,
    "Y7Nodes_LMStudioText": Y7Nodes_LMStudioText,
    "Y7Nodes_LMStudioVision": Y7Nodes_LMStudioVision,
    "Y7Nodes_SelectLMSModel": Y7Nodes_SelectLMSModel,
    "Y7Nodes_QwenVL": Y7Nodes_QwenVL,
    "SamplerSelect_Name": SamplerSelect_Name,
    "Y7Nodes_PasteCroppedImageBack": Y7Nodes_PasteCroppedImageBack,
    "Y7Nodes_ScaleImageToTotalPixels": Y7Nodes_ScaleImageToTotalPixels,
    "Y7Nodes_ScaleImageBy": Y7Nodes_ScaleImageBy,
    "Y7Nodes_JoyCaption": Y7Nodes_JoyCaption,
    "Y7Nodes_JoyCaption_ExtraOptions": Y7Nodes_JoyCaption_ExtraOptions,
    "Y7Nodes_ImageBatchPath": Y7Nodes_ImageBatchPath,
    "Y7Nodes_CaptionSaver": Y7Nodes_CaptionSaver,
    "Y7Nodes_LoadImage": Y7Nodes_LoadImage,
    "Y7Nodes_ImageStitcher": Y7Nodes_ImageStitcher,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Y7Nodes_ImageSizePresets" : "Y7 Image Size (Presets)",
    "Y7Nodes_Text": "Y7 Text",
    "Y7Nodes_ShowAnything": "Y7 Show Anything",
    "Y7Nodes_PromptEnhancerFlux": "Y7 Prompt Enhancer (Flux1)",
    "Y7Nodes_PromptEnhancerFlux2": "Y7 Prompt Enhancer (Flux2)",
    "Y7Nodes_T5_TokenCounter": "Y7 T5 Token Counter",
    "Y7Nodes_CLIP_TokenCounter": "Y7 CLIP Token Counter",    
    "Y7Nodes_CatchEditTextNodeDual": "Y7 Catch and Edit Text (Dual)",
    "Y7Nodes_CropToNearestMultiple": "Y7 Crop to Nearest Multiple",
    "Y7Nodes_ColorMatchMasked": "Y7 Color Match (Masked)",
    "Y7Nodes_AspectRatioPicker": "Y7 Aspect Ratio Picker",
    "Y7Nodes_LMStudioText": "Y7 LM Studio (Text)",
    "Y7Nodes_LMStudioVision": "Y7 LM Studio (Vision)",
    "Y7Nodes_SelectLMSModel": "Y7 Select LMS Model",
    "Y7Nodes_QwenVL": "Y7 Qwen3-VL",
    "SamplerSelect_Name": "Sampler Select (Name)",
    "Y7Nodes_PasteCroppedImageBack": "Y7 Paste Cropped Image Back",
    "Y7Nodes_ScaleImageToTotalPixels": "Y7 Scale Image to Total Pixels",
    "Y7Nodes_ScaleImageBy": "Y7 Scale Image By",
    "Y7Nodes_JoyCaption": "Y7 JoyCaption",
    "Y7Nodes_JoyCaption_ExtraOptions": "Y7 JoyCaption Extra Options",
    "Y7Nodes_ImageBatchPath": "Y7 Image Batch Path",
    "Y7Nodes_CaptionSaver": "Y7 Caption Saver",
    "Y7Nodes_LoadImage": "Y7 Load Image",
    "Y7Nodes_ImageStitcher": "Y7 Image Stitcher",
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


    async def get_image_size_dims(request):
        preset = request.rel_url.query.get("preset", "default")
        dims = Y7Nodes_ImageSizePresets.get_dims_for_preset(preset)
        labels = [d["label"] for d in dims] + ["Custom"]
        return web.json_response({"labels": labels})

    # Register routes using the actual directory name
    PromptServer.instance.app.add_routes([
        # Route for JavaScript files
        web.static(f"/{dir_name}", (current_dir / "web").as_posix()),
        web.get("/y7nodes/image_size_dims", get_image_size_dims),


    # # Register routes for web assets and data files
    # PromptServer.instance.app.add_routes([
    #     # Route for JavaScript files
    #     web.static("/ComfyUI-Y7Nodes", (Path(__file__).parent.absolute() / "web").as_posix())

    ])
