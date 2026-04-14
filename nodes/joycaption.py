# Stripped-down variant of 1038lab/ComfyUI-JoyCaption (AILab)
# https://github.com/1038lab/ComfyUI-JoyCaption
# Only includes the Advanced and Extra Options nodes.
# GGUF support and the simple node have been removed.
# Only HuggingFace JoyCaption models are supported.
# Caption tools (ImageBatchPath, CaptionSaver) are in caption_tools.py.

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
import folder_paths
from pathlib import Path
from PIL import Image
from torchvision.transforms import ToPILImage
import json
import gc
import os

with open(Path(__file__).parent / "jc_data.json", "r", encoding="utf-8") as f:
    _config = json.load(f)
    CAPTION_TYPE_MAP = _config["caption_type_map"]
    EXTRA_OPTIONS = _config["extra_options"]
    MEMORY_EFFICIENT_CONFIGS = _config["memory_efficient_configs"]
    MODEL_SETTINGS = _config["model_settings"]
    CAPTION_LENGTH_CHOICES = _config["caption_length_choices"]
    HF_MODELS = _config["hf_models"]


def _build_prompt(caption_type: str, caption_length, extra_options: list, name_input: str) -> str:
    if caption_length == "any":
        map_idx = 0
    else:
        map_idx = 1

    prompt = CAPTION_TYPE_MAP[caption_type][map_idx]

    if extra_options:
        prompt += " " + " ".join(extra_options)

    return prompt.format(
        name=name_input or "{NAME}",
        length=caption_length,
    )


# Global cache for "Global Cache" memory management mode
_MODEL_CACHE = {}


class _JCModel:
    """Loads and runs a JoyCaption LLaVA model."""

    def __init__(self, model_key: str, memory_mode: str):
        cache_key = f"{model_key}_{memory_mode}"

        if cache_key in _MODEL_CACHE:
            try:
                self.processor = _MODEL_CACHE[cache_key]["processor"]
                self.model = _MODEL_CACHE[cache_key]["model"]
                self.device = _MODEL_CACHE[cache_key]["device"]
                if not next(self.model.parameters()).is_cuda:
                    raise RuntimeError("Cached model not on GPU")
                print(f"[JoyCaption] Using cached model: {cache_key}")
                return
            except Exception as e:
                print(f"[JoyCaption] Cache validation failed: {e}, reloading...")
                del _MODEL_CACHE[cache_key]
                torch.cuda.empty_cache()

        model_id = HF_MODELS[model_key]["name"]
        checkpoint_path = Path(folder_paths.models_dir) / "LLM" / Path(model_id).stem

        if not checkpoint_path.exists():
            from huggingface_hub import snapshot_download
            print(f"[JoyCaption] Downloading {model_id} ...")
            snapshot_download(repo_id=model_id, local_dir=str(checkpoint_path),
                              force_download=False, local_files_only=False)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
            if hasattr(torch.backends, "cuda"):
                if hasattr(torch.backends.cuda, "matmul"):
                    torch.backends.cuda.matmul.allow_tf32 = True
                if hasattr(torch.backends.cuda, "allow_tf32"):
                    torch.backends.cuda.allow_tf32 = True
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

        self.processor = AutoProcessor.from_pretrained(
            str(checkpoint_path),
            use_fast=True,
            image_processor_type="CLIPImageProcessor",
            image_size=336,
        )

        if hasattr(self.processor, "image_processor") and hasattr(self.processor.image_processor, "size"):
            sz = self.processor.image_processor.size
            if isinstance(sz, dict):
                self.target_size = (sz.get("height", 336), sz.get("width", 336))
            elif isinstance(sz, (list, tuple)):
                self.target_size = tuple(sz) if len(sz) == 2 else (sz[0], sz[0])
            else:
                self.target_size = (sz, sz)
        else:
            self.target_size = (336, 336)

        model_kwargs = {"device_map": "cuda" if self.device == "cuda" else "cpu"}

        try:
            if "FP8-Dynamic" in model_id:
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    str(checkpoint_path), torch_dtype="auto", **model_kwargs)
            elif memory_mode == "Full Precision (bf16)":
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    str(checkpoint_path), torch_dtype=torch.bfloat16, **model_kwargs)
            elif memory_mode == "Balanced (8-bit)":
                qnt = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.float16,
                    bnb_8bit_use_double_quant=True,
                    llm_int8_skip_modules=["vision_tower", "multi_modal_projector"],
                    llm_int8_enable_fp32_cpu_offload=True,
                )
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    str(checkpoint_path), torch_dtype=torch.float16,
                    quantization_config=qnt, **model_kwargs)
            else:  # Maximum Savings (4-bit)
                qnt = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    llm_int8_skip_modules=["vision_tower", "multi_modal_projector"],
                    llm_int8_enable_fp32_cpu_offload=True,
                )
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    str(checkpoint_path), torch_dtype="auto",
                    quantization_config=qnt, **model_kwargs)

            self.model.eval()

            if self.device == "cuda" and not next(self.model.parameters()).is_cuda:
                raise RuntimeError("Model failed to load on GPU")

            if memory_mode == "Global Cache":
                _MODEL_CACHE[cache_key] = {
                    "processor": self.processor,
                    "model": self.model,
                    "device": self.device,
                }

        except Exception as e:
            del self.model, self.processor
            torch.cuda.empty_cache()
            gc.collect()
            raise RuntimeError(f"[JoyCaption] Error loading model: {e}") from e

    @torch.inference_mode()
    def generate(self, image: Image.Image, system: str, prompt: str,
                 max_new_tokens: int, temperature: float, top_p: float, top_k: int,
                 seed: int = 0) -> str:
        convo = [
            {"role": "system", "content": system.strip()},
            {"role": "user", "content": prompt.strip()},
        ]
        convo_string = self.processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)

        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.resize(self.target_size, Image.Resampling.LANCZOS)

        inputs = self.processor(text=[convo_string], images=[image], return_tensors="pt").to(self.device)

        if hasattr(inputs, "pixel_values") and inputs["pixel_values"] is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)

        torch.manual_seed(seed)
        with torch.cuda.amp.autocast(enabled=True):
            generate_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                suppress_tokens=None,
                use_cache=True,
                temperature=temperature,
                top_k=None if top_k == 0 else top_k,
                top_p=top_p,
            )[0]

        generate_ids = generate_ids[inputs["input_ids"].shape[1]:]
        return self.processor.tokenizer.decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ).strip()


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

class Y7Nodes_JoyCaption_ExtraOptions:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {"required": {}}
        for key, value in EXTRA_OPTIONS.items():
            inputs["required"][key] = ("BOOLEAN", {"default": value["default"]})
        inputs["required"]["character_name"] = (
            "STRING", {"default": "", "multiline": True, "placeholder": "Character Name"})
        return inputs

    RETURN_TYPES = ("JOYCAPTION_EXTRA_OPTIONS",)
    RETURN_NAMES = ("extra_options",)
    FUNCTION = "get_extra_options"
    CATEGORY = "Y7Nodes/JoyCaption"

    def get_extra_options(self, character_name, **kwargs):
        selected = [v["description"] for k, v in EXTRA_OPTIONS.items() if kwargs.get(k, False)]
        return ([selected, character_name],)


class Y7Nodes_JoyCaption:
    @classmethod
    def INPUT_TYPES(cls):
        model_list = list(HF_MODELS.keys())
        s = MODEL_SETTINGS
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (model_list, {"default": model_list[1]}),
                "quantization": (list(MEMORY_EFFICIENT_CONFIGS.keys()), {"default": "Full Precision (bf16)"}),
                "prompt_style": (list(CAPTION_TYPE_MAP.keys()), {"default": "Descriptive"}),
                "caption_length": (CAPTION_LENGTH_CHOICES, {"default": "any"}),
                "max_new_tokens": ("INT", {"default": s["default_max_tokens"], "min": 1, "max": 2048}),
                "temperature": ("FLOAT", {"default": s["default_temperature"], "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": s["default_top_p"], "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": s["default_top_k"], "min": 0, "max": 100}),
                "seed": ("INT", {"default": 1234567890, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": "fixed"}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True}),
                "memory_management": (["Keep in Memory", "Clear After Run", "Global Cache"], {"default": "Keep in Memory"}),
            },
            "optional": {
                "extra_options": ("JOYCAPTION_EXTRA_OPTIONS",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("PROMPT", "STRING")
    FUNCTION = "generate"
    CATEGORY = "Y7Nodes/JoyCaption"

    def __init__(self):
        self.predictor = None
        self.current_model = None
        self.current_quantization = None

    def generate(self, image, model, quantization, prompt_style, caption_length,
                 max_new_tokens, temperature, top_p, top_k, seed, custom_prompt,
                 memory_management, extra_options=None):
        try:
            needs_reload = (
                self.predictor is None
                or self.current_model != model
                or self.current_quantization != quantization
            )

            if memory_management == "Global Cache" or needs_reload:
                if needs_reload and self.predictor is not None:
                    del self.predictor
                    self.predictor = None
                    torch.cuda.empty_cache()
                    gc.collect()
                try:
                    self.predictor = _JCModel(model, quantization)
                    self.current_model = model
                    self.current_quantization = quantization
                except Exception as e:
                    return (str(e), "")

            if custom_prompt and custom_prompt.strip():
                prompt = custom_prompt.strip()
            else:
                prompt = _build_prompt(
                    prompt_style,
                    caption_length,
                    extra_options[0] if extra_options else [],
                    extra_options[1] if extra_options else "{NAME}",
                )

            pil_image = ToPILImage()(image[0].permute(2, 0, 1))
            response = self.predictor.generate(
                image=pil_image,
                system=MODEL_SETTINGS["default_system_prompt"],
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=seed,
            )

            if memory_management == "Clear After Run":
                del self.predictor
                self.predictor = None
                torch.cuda.empty_cache()
                gc.collect()

            return (prompt, response)

        except Exception as e:
            if memory_management == "Clear After Run":
                del self.predictor
                self.predictor = None
                torch.cuda.empty_cache()
                gc.collect()
            raise e
