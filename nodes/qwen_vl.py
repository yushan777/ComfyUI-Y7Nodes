import gc
import os
import torch
import numpy as np
from PIL import Image

import folder_paths
import comfy.model_management
import comfy.model_patcher
from comfy.utils import ProgressBar
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

QWEN_VL_MODELS = [
    "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen3-VL-32B-Instruct",
]

# Preset instructions (image analysis tasks shown in the dropdown).
# Values are the actual instruction text sent to the model.
# Sourced from AILab_System_Prompts.json (AILab/ComfyUI-QwenVL).
PRESET_PROMPTS = {
    "🖼️ Tags": (
        "Your task is to generate a clean list of comma-separated tags for a text-to-image AI, "
        "based *only* on the visual information in the image. Limit the output to a maximum of 50 "
        "unique tags. Strictly describe visual elements like subject, clothing, environment, colors, "
        "lighting, and composition. Do not include abstract concepts, interpretations, marketing "
        "terms, or technical jargon. The goal is a concise list of visual descriptors. "
        "Avoid repeating tags."
    ),
    "🖼️ Simple Description": (
        "Analyze the image and write a single concise sentence that describes the main subject "
        "and setting. Keep it grounded in visible details only."
    ),
    "🖼️ Detailed Description": (
        "Write ONE detailed paragraph (6–10 sentences). Describe only what is visible: subject(s) "
        "and actions; people details if present (approx age group, gender expression if clear, hair, "
        "facial expression, pose, clothing, accessories); environment (location type, background "
        "elements, time cues); lighting (source, direction, softness/hardness, color temperature, "
        "shadows); camera viewpoint (eye-level/low/high, distance) and composition (framing, focal "
        "emphasis). No preface, no reasoning, no <think>."
    ),
    "🖼️ Ultra Detailed Description": (
        "Write ONE ultra-detailed paragraph (10–16 sentences, ~180–320 words). Stay grounded in "
        "visible details. Include: subject micro-details (materials, textures, patterns, wear, "
        "reflections); people details if present (hair, skin tones, makeup, jewelry, fabric types, "
        "fit); environment depth (foreground/midground/background, signage/props, surface materials); "
        "lighting analysis (key/fill/back light, direction, softness, highlights, shadow shape); "
        "camera perspective (angle, lens feel, depth of field) and composition (leading lines, "
        "negative space, symmetry/asymmetry, visual hierarchy). No preface, no reasoning, no <think>."
    ),
    "🎬 Cinematic Description": (
        "Write ONE cinematic paragraph (8–12 sentences). Describe the scene like a film still: "
        "subject(s) and action; environment and atmosphere; lighting design (practical lights vs "
        "ambient, direction, contrast); camera language (shot type, angle, lens feel, depth of field, "
        "motion implied); composition and mood. Keep it vivid but factual (no made-up story). "
        "No preface, no reasoning, no <think>."
    ),
    "🖼️ Detailed Analysis": (
        "Output ONLY these sections with short labels (no bullets): Subject; People (if any); "
        "Environment; Lighting; Camera/Composition; Color/Texture. In each section, write 2–4 "
        "sentences of concrete visible details. If something is not visible, write 'not visible'. "
        "No preface, no reasoning, no <think>."
    ),
    "📖 Short Story": (
        "Write a short, imaginative story inspired by this image."
    ),
    "🪄 Prompt Refine & Expand": (
        "Refine and enhance the following user prompt for creative text-to-image generation. "
        "Keep the meaning and keywords, make it more expressive and visually rich. Output ONLY "
        "the improved prompt text (no preface, no bullets, no JSON, no <think>, no commentary)."
    ),
}

_DEFAULT_PRESET = "🖼️ Detailed Description"


# ---------------------------------------------------------------------------
# Compatibility wrapper
# ---------------------------------------------------------------------------

class _QwenVLWrapper(torch.nn.Module):
    """Proxy around Qwen3VLForConditionalGeneration for ModelPatcher compatibility.

    HuggingFace's GenerationMixin defines `device` as a read-only property.
    ModelPatcher writes model.device = device_to as a tracking attribute,
    which raises AttributeError on any model that inherits that property.

    This wrapper is a plain nn.Module with no conflicting properties, so
    ModelPatcher can set tracking attributes freely.  The real model is
    registered as self._qwen_model (a child module), so .parameters(),
    .to(), .state_dict() etc. all work transparently on the full weight tree.
    """

    def __init__(self, model):
        super().__init__()
        self._qwen_model = model
        # Pre-initialise counters that ModelPatcher reads with += before writing
        self.lowvram_patch_counter = 0
        self.model_lowvram = False

    def forward(self, *args, **kwargs):
        return self._qwen_model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self._qwen_model.generate(*args, **kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _comfy_image_to_pil(image_tensor):
    """Convert a ComfyUI IMAGE tensor [B, H, W, C] float32 0-1 to PIL Image."""
    arr = (image_tensor[0].cpu().float().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class Y7Nodes_QwenVL:
    """
    Loads Qwen3-VL-8B-Instruct and runs vision-language inference.
    The model is cached on the instance between runs when keep_model_loaded=True.
    Image input is optional — omit it for text-only queries.
    """

    _BASE_DIR = os.path.join(folder_paths.models_dir, "LLM")

    def __init__(self):
        self._patcher = None
        self._processor = None
        self._loaded_dtype = None
        self._loaded_model = None

    @classmethod
    def _local_model_dir(cls, hf_model_id):
        """Return the local directory for a HuggingFace model ID (org/name → name)."""
        return os.path.join(cls._BASE_DIR, hf_model_id.split("/")[-1])

    @classmethod
    def INPUT_TYPES(cls):
        preset_names = list(PRESET_PROMPTS.keys())
        return {
            "required": {
                "model_name": (QWEN_VL_MODELS, {"default": "Qwen/Qwen3-VL-8B-Instruct"}),
                "dtype": (["bfloat16", "float16"], {"default": "bfloat16"}),
                "preset_prompt": (
                    preset_names,
                    {
                        "default": _DEFAULT_PRESET,
                        "tooltip": "Built-in instruction describing how the model should analyse the image.",
                    },
                ),
                "custom_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "If filled, replaces the preset instruction entirely.",
                    },
                ),
                "max_new_tokens": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
                "keep_model_loaded": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Keep the model in VRAM/RAM after the run so the next prompt skips reloading.",
                    },
                ),
                "download_model": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Automatically download the selected model from HuggingFace if it is not found locally.",
                    },
                ),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "run"
    CATEGORY = "Y7Nodes/VLM"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self, model_name, dtype, download_model=False):
        model_path = self._local_model_dir(model_name)
        if not os.path.isdir(model_path):
            if not download_model:
                short_name = model_name.split("/")[-1]
                raise FileNotFoundError(
                    f"Model '{model_name}' is not available locally.\n"
                    f"Expected path: {model_path}\n"
                    f"Enable 'download_model' to download it automatically, or manually place "
                    f"the weights in: models/LLM/{short_name}"
                )
            print(f"[QwenVL] Downloading {model_name} to {model_path} ...")
            from huggingface_hub import snapshot_download
            os.makedirs(model_path, exist_ok=True)
            snapshot_download(repo_id=model_name, local_dir=model_path)
            print(f"[QwenVL] Download complete.")

        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16

        # Processor is CPU-only (tokeniser + image preprocessor, no GPU weights)
        processor = AutoProcessor.from_pretrained(model_path)

        # Load weights to CPU.  Do NOT use device_map="auto" — that moves tensors
        # directly to GPU and bypasses ComfyUI's memory manager, preventing it from
        # coordinating VRAM with other loaded models (diffusion models, VAE, etc.).
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        model.eval()

        wrapped = _QwenVLWrapper(model)

        device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()

        patcher = comfy.model_patcher.ModelPatcher(
            wrapped,
            load_device=device,
            offload_device=offload_device,
            size=comfy.model_management.module_size(model),
        )

        self._patcher = patcher
        self._processor = processor
        self._loaded_dtype = dtype
        self._loaded_model = model_name

    def _unload(self):
        if self._patcher is not None:
            try:
                # detach() moves the model back to offload_device (CPU) and
                # zeroes model_loaded_weight_memory so ComfyUI's accounting is correct.
                # The stale LoadedModel entry is cleaned up on the next
                # cleanup_models_gc() call at the top of load_models_gpu().
                self._patcher.detach(unpatch_all=False)
            except Exception as e:
                print(f"[QwenVL] Patcher detach warning: {e}")
            self._patcher = None
        self._processor = None
        self._loaded_dtype = None
        self._loaded_model = None
        gc.collect()

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def run(
        self,
        model_name,
        preset_prompt,
        custom_prompt,
        max_new_tokens,
        temperature,
        top_p,
        repetition_penalty,
        seed,
        dtype,
        keep_model_loaded,
        download_model,
        image=None,
    ):
        pbar = ProgressBar(4)

        # Reload if not cached or if model/dtype changed
        if self._patcher is None or self._loaded_model != model_name or self._loaded_dtype != dtype:
            self._unload()
            self._load(model_name, dtype, download_model=download_model)

        pbar.update(1)  # model ready

        patcher = self._patcher
        processor = self._processor

        # Resolve instruction: custom query overrides preset
        instruction = PRESET_PROMPTS.get(preset_prompt, "Describe this image.")
        if custom_prompt and custom_prompt.strip():
            instruction = custom_prompt.strip()

        # Ask ComfyUI's memory manager to move this model to the GPU,
        # coordinating with all other currently loaded models.
        overhead = int(patcher.model_size() * 0.20)
        comfy.model_management.load_models_gpu([patcher], memory_required=overhead)
        device = patcher.load_device

        pbar.update(1)  # loaded to GPU

        torch.manual_seed(seed)

        # Build messages --------------------------------------------------------
        user_content = []
        if image is not None:
            user_content.append({"type": "image", "image": _comfy_image_to_pil(image)})
        user_content.append({"type": "text", "text": instruction})
        messages = [{"role": "user", "content": user_content}]

        # Tokenise and prepare pixel values ------------------------------------
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images = [item["image"] for item in user_content if item["type"] == "image"]

        inputs = processor(
            text=[text],
            images=images or None,
            return_tensors="pt",
            padding=True,
        )
        inputs.pop("token_type_ids", None)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        pbar.update(1)  # inputs ready, generating

        # Generate --------------------------------------------------------------
        do_sample = temperature > 0.0
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        try:
            with torch.no_grad():
                generated_ids = patcher.model.generate(**inputs, **gen_kwargs)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            input_len = inputs["input_ids"].shape[-1]
            response = processor.decode(
                generated_ids[0][input_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()

            pbar.update(1)  # done

            return (response,)

        finally:
            if not keep_model_loaded:
                self._unload()
