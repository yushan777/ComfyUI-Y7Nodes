import re
import comfy.model_management as model_management
import lmstudio as lms
from torchvision.transforms.functional import to_pil_image
from tempfile import NamedTemporaryFile
import folder_paths
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent  # points to comfyui-y7nodes root
folder_paths.add_model_folder_path(
    "lms_config", (SCRIPT_DIR / "lms_config").as_posix())


# ---------------------------------------------------------------------------
# Shared helpers (not registered as nodes)
# ---------------------------------------------------------------------------

def _common_inputs():
    """Returns the input parameters shared by both Text and Vision nodes."""
    return {
        "model_identifier": ("STRING", {"default": ""}),
        "reasoning_tag": ("STRING", {"default": "think"}),
        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        "ip": ("STRING", {"default": "localhost"}),
        "port": ("INT", {"default": 1234}),
        "temperature": ("FLOAT", {"default": 0.7, "min": 0.01, "max": 1.0, "step": 0.01}),
        "max_tokens": ("INT", {"default": 600, "min": -1, "max": 0xffffffffffffffff}),
        "unload_llm": ("BOOLEAN", {"default": False}),
        "unload_comfy_models": ("BOOLEAN", {"default": False}),
    }


def _extract_reasoning(content, reasoning_tag):
    """Separate the main response from any <reasoning_tag>…</reasoning_tag> block."""
    result = re.sub(
        rf"<{reasoning_tag}>.*?</{reasoning_tag}>", "",
        content, flags=re.DOTALL).strip()
    reasoning = re.sub(
        rf".*<{reasoning_tag}>(.*?)</{reasoning_tag}>.*", r"\1",
        content, flags=re.DOTALL).strip()
    return result, reasoning


def _prepare_unload(unload_comfy_models):
    """Optionally free VRAM by unloading ComfyUI models before running the LLM."""
    if unload_comfy_models:
        model_management.unload_all_models()
        model_management.soft_empty_cache(True)


# ---------------------------------------------------------------------------
# Text node – pure text prompts (no vision)
# ---------------------------------------------------------------------------

class Y7Nodes_LMStudioText:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        common = _common_inputs()
        return {
            "required": {
                "prompt": ("STRING", {"forceInput": True}),
                **common,
                "draft_model": ("STRING", {"default": ""}),
                "system_message": ("STRING", {
                    "multiline": True,
                    "default": (
                        "You are an AI assistant specialized in generating detailed and creative image prompts for AI image "
                        "generation - specifically for Flux.2 Klein. Your task is to expand the given user prompt into a " 
                        "well-structured, vivid, and highly descriptive prompt while ensuring that all terms from the " 
                        "original prompt are included. Enhance the visual quality and artistic impact by adding relevant " 
                        "details including lighting conditions, but do not omit or alter any key elements provided by the " 
                        "user. Follow the given instructions or guidelines and respond only with the refined prompt."
                    )
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("Extended Prompt", "Reasoning")
    OUTPUT_NODE = False
    FUNCTION = "do_it"
    CATEGORY = "Y7Nodes/LMStudio"

    def do_it(self, prompt, model_identifier, draft_model, system_message,
              reasoning_tag, seed, ip, port, temperature, max_tokens,
              unload_llm, unload_comfy_models):

        _prepare_unload(unload_comfy_models)

        server_api_host = f"{ip}:{port}"

        with lms.Client(server_api_host) as client:
            model = client.llm.model(model_identifier)

            try:
                chat = lms.Chat(system_message)
                chat.add_user_message(prompt)
                content = str(model.respond(chat, config={
                    "temperature": temperature,
                    "maxTokens": max_tokens,
                    "draftModel": draft_model,
                }))
            except Exception:
                print(
                    "Prediction error: Trying alternative approach to "
                    "prevent prediction error based on wrong template type.")
                chat = lms.Chat()
                chat.add_user_message(
                    f"{system_message}: User input: {prompt}")
                content = str(model.respond(chat, config={
                    "temperature": temperature,
                    "maxTokens": max_tokens,
                    "draftModel": draft_model,
                }))

            result, reasoning = _extract_reasoning(content, reasoning_tag)

            if unload_llm:
                model.unload()

        return (result, reasoning)


# ---------------------------------------------------------------------------
# Vision node – image + instruction (VL models)
# ---------------------------------------------------------------------------

class Y7Nodes_LMStudioVision:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        common = _common_inputs()
        return {
            "required": {
                "image": ("IMAGE",),
                **common,
                "system_message": ("STRING", {
                    "multiline": True,
                    "default": (
                        "Describe this image in detail. Include the subject, "
                        "setting, lighting, colors, mood, composition, and "
                        "any notable artistic or stylistic qualities."
                    )
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("Response", "Reasoning")
    OUTPUT_NODE = False
    FUNCTION = "do_it"
    CATEGORY = "Y7Nodes/LMStudio"

    def do_it(self, image, model_identifier, system_message,
              reasoning_tag, seed, ip, port, temperature, max_tokens,
              unload_llm, unload_comfy_models):

        _prepare_unload(unload_comfy_models)

        server_api_host = f"{ip}:{port}"

        with lms.Client(server_api_host) as client:
            model = client.llm.model(model_identifier)

            info = model.get_info()
            if not info.vision:
                if unload_llm:
                    model.unload()
                raise Exception(
                    "The loaded model is not vision enabled. "
                    "Please try another model.")

            # Convert ComfyUI tensor → PIL → temp JPEG for the SDK
            image_new = image.squeeze(0).permute(2, 0, 1)
            image_pil = to_pil_image(image_new)

            with NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
                image_pil.save(temp, format="JPEG")
                temp.flush()
                image_handle = client.files.prepare_image(temp.name)

            chat = lms.Chat()
            chat.add_user_message(system_message, images=[image_handle])
            content = str(model.respond(chat, config={
                "temperature": temperature,
                "maxTokens": max_tokens,
            }))

            result, reasoning = _extract_reasoning(content, reasoning_tag)

            if unload_llm:
                model.unload()

        return (result, reasoning)


# ---------------------------------------------------------------------------
# Model selector (unchanged)
# ---------------------------------------------------------------------------

class Y7Nodes_SelectLMSModel:
    def __init__(self):
        pass

    @classmethod
    def get_models(cls, id="models.txt"):
        file_path = folder_paths.get_full_path("lms_config", id)
        if file_path is None:
            return ["(no models.txt found - add one to lms_config/)"]
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                models = [line.strip() for line in f if line.strip()]
                return models if models else ["(models.txt is empty)"]
        except Exception:
            return ["(error reading models.txt)"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_id": (
                    s.get_models(),
                    {"tooltip": "Add your favorite model names to the models.txt file in comfyui-y7nodes/lms_config/"}
                )
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_id",)
    OUTPUT_NODE = False
    FUNCTION = "do_it"
    CATEGORY = "Y7Nodes/LMStudio"

    def do_it(self, model_id):
        return (model_id,)
