import re
import comfy.model_management as model_management
import lmstudio as lms
from torchvision.transforms.functional import to_pil_image
from tempfile import NamedTemporaryFile
import folder_paths
from pathlib import Path

from comfy_api.latest import io

SCRIPT_DIR = Path(__file__).parent.parent  # points to comfyui-y7nodes root
folder_paths.add_model_folder_path(
    "lms_config", (SCRIPT_DIR / "lms_config").as_posix())


# ---------------------------------------------------------------------------
# Shared helpers (not registered as nodes)
# ---------------------------------------------------------------------------

def _common_inputs():
    """Returns the input parameters shared by both Text and Vision nodes."""
    return [
        io.String.Input("model_identifier", default=""),
        io.String.Input("reasoning_tag", default="think"),
        io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
        io.String.Input("ip", default="localhost"),
        io.Int.Input("port", default=1234),
        io.Float.Input("temperature", default=0.7, min=0.01, max=1.0, step=0.01),
        io.Int.Input("max_tokens", default=600, min=-1, max=0xffffffffffffffff),
        io.Boolean.Input("unload_llm", default=False),
        io.Boolean.Input("unload_comfy_models", default=False),
    ]


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

class Y7Nodes_LMStudioText(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Y7Nodes_LMStudioText",
            display_name="Y7 LM Studio (Text)",
            category="Y7Nodes/LMStudio",
            description="Expand a text prompt using a local LM Studio model.",
            inputs=[
                io.String.Input("prompt", force_input=True),
                *_common_inputs(),
                io.String.Input("draft_model", default=""),
                io.String.Input(
                    "system_message",
                    multiline=True,
                    default=(
                        "You are an AI assistant specialized in generating detailed and creative image prompts for AI image "
                        "generation - specifically for Flux.2 Klein. Your task is to expand the given user prompt into a " 
                        "well-structured, vivid, and highly descriptive prompt while ensuring that all terms from the " 
                        "original prompt are included. Enhance the visual quality and artistic impact by adding relevant " 
                        "details including lighting conditions, but do not omit or alter any key elements provided by the " 
                        "user. Follow the given instructions or guidelines and respond only with the refined prompt."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(display_name="Extended Prompt"),
                io.String.Output(display_name="Reasoning"),
            ],
        )

    @classmethod
    def execute(cls, prompt, model_identifier, draft_model, system_message,
                reasoning_tag, seed, ip, port, temperature, max_tokens,
                unload_llm, unload_comfy_models) -> io.NodeOutput:

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

        return io.NodeOutput(result, reasoning)


# ---------------------------------------------------------------------------
# Vision node – image + instruction (VL models)
# ---------------------------------------------------------------------------

class Y7Nodes_LMStudioVision(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Y7Nodes_LMStudioVision",
            display_name="Y7 LM Studio (Vision)",
            category="Y7Nodes/LMStudio",
            description="Describe an image using a local vision-enabled LM Studio model.",
            inputs=[
                io.Image.Input("image"),
                *_common_inputs(),
                io.String.Input(
                    "system_message",
                    multiline=True,
                    default=(
                        "Describe this image in detail. Include the subject, "
                        "setting, lighting, colors, mood, composition, and "
                        "any notable artistic or stylistic qualities."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(display_name="Response"),
                io.String.Output(display_name="Reasoning"),
            ],
        )

    @classmethod
    def execute(cls, image, model_identifier, system_message,
                reasoning_tag, seed, ip, port, temperature, max_tokens,
                unload_llm, unload_comfy_models) -> io.NodeOutput:

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

        return io.NodeOutput(result, reasoning)


# ---------------------------------------------------------------------------
# Model selector (unchanged)
# ---------------------------------------------------------------------------

class Y7Nodes_SelectLMSModel(io.ComfyNode):

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
    def define_schema(cls):
        return io.Schema(
            node_id="Y7Nodes_SelectLMSModel",
            display_name="Y7 Select LM Studio Model",
            category="Y7Nodes/LMStudio",
            description="Pick a model name from lms_config/models.txt.",
            inputs=[
                io.Combo.Input(
                    "model_id",
                    options=cls.get_models(),
                    tooltip="Add your favorite model names to the models.txt file in comfyui-y7nodes/lms_config/",
                ),
            ],
            outputs=[
                io.String.Output(display_name="model_id"),
            ],
        )

    @classmethod
    def execute(cls, model_id) -> io.NodeOutput:
        return io.NodeOutput(model_id)
