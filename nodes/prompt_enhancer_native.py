import re
import hashlib

from ..utils.logger import logger
from ..utils.colored_print import color, style


# Default instruction. Kept deliberately blunt about output format: the empty
# thought-channel priming in Gemma4's template is only a nudge, so the wording
# here does most of the work in keeping reasoning out of the answer.
DEFAULT_INSTRUCTION = """Rewrite the user's text into a single detailed image generation prompt.

- Keep every element the user asked for. Invent concrete visual detail where they were vague: lighting, materials, textures, setting, composition.
- Describe only what is visible. No smell, taste, sound or emotion.
- Output the prompt only. No preamble, no explanation, no markdown, no quotes."""


# Matches the tags Gemma4's decode() emits (it translates its <|channel>thought
# markers into <think>/</think>) and Qwen3's native thinking tags.
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK_RE = re.compile(r"<think>(.*)$", re.DOTALL | re.IGNORECASE)


# =====================================================================================
class Y7Nodes_PromptEnhancerNative:
    """
    Prompt enhancer driven by a native ComfyUI text encoder (CLIP input).

    Unlike the other Y7 enhancers this loads nothing itself - the CLIP object
    arrives already loaded and ComfyUI handles all VRAM management.
    Requires a text encoder that supports generation: Gemma 3 / Gemma 4,
    Qwen3 / Qwen3.5 / Qwen3-VL.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "A generation-capable text encoder loaded with CLIPLoader (Gemma 3/4, Qwen3, Qwen3-VL). T5/CLIP-L will not work."}),
                "text": ("STRING", {"default": "", "multiline": True, "tooltip": "The basic prompt to enhance."}),
                "instruction": ("STRING", {"default": DEFAULT_INSTRUCTION, "multiline": True, "tooltip": "System-style instruction placed before the text."}),
                "max_length": ("INT", {"default": 2048, "min": 64, "max": 32768, "step": 64,
                                       "tooltip": "Maximum NEW tokens to generate (not the context window). Reasoning is spent from this same budget. Costs ~84KB of VRAM per token in KV cache, reserved up front."}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01,
                                          "tooltip": "1.0 is Google's recommended value for Gemma. 0 switches to greedy decoding, which ignores top_k/top_p entirely."}),
                "top_k": ("INT", {"default": 64, "min": 0, "max": 1000,
                                  "tooltip": "Keep only the k most likely tokens. 64 is Google's recommended value for Gemma. 0 disables the filter."}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01,
                                    "tooltip": "Nucleus sampling: keep the smallest set of tokens whose probabilities sum to p. 0.95 is Google's recommended value for Gemma. 1.0 disables the filter."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "thinking": ("BOOLEAN", {"default": False, "tooltip": "Let the model reason first. Its reasoning goes to the thinking output, never the prompt output."}),
            },
            "hidden": {},
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("thinking_output", "enhanced_prompt",)
    FUNCTION = "enhance"
    CATEGORY = "Y7Nodes/Prompt"
    OUTPUT_NODE = False

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # clip is unhashable, and ComfyUI already tracks the model object itself,
        # so only the generation-affecting widgets go into the hash.
        parts = [
            kwargs.get("text", ""),
            kwargs.get("instruction", ""),
            kwargs.get("max_length", 2048),
            kwargs.get("temperature", 1.0),
            kwargs.get("top_k", 64),
            kwargs.get("top_p", 0.95),
            kwargs.get("seed", 0),
            kwargs.get("thinking", False),
        ]
        return hashlib.md5("_".join(str(p) for p in parts).encode()).hexdigest()

    # ==================================================================================
    # Private helpers
    # ==================================================================================

    @staticmethod
    def _generation_error(clip):
        """
        Return an error string if this encoder cannot generate text, else None.

        Every encoder inherits a generate() from SD1ClipModel, so checking the
        wrapper tells us nothing - the real test is whether the underlying
        transformer implements one. Llama-family transformers do; T5, CLIP and
        UMT5 do not.
        """
        unsupported = (
            "This text encoder does not support text generation.\n"
            "Supported: Gemma 3, Gemma 4, Qwen3, Qwen3.5, Qwen3-VL.\n"
            "Not supported: T5 (all sizes), CLIP-L/G, UMT5, Gemma 2, LLaMA-3.1."
        )

        cond = getattr(clip, "cond_stage_model", None)
        if cond is None:
            return "No text encoder found on the CLIP input."

        # SD1ClipModel-style wrappers hold the real encoder under an attribute
        # named by self.clip. Composite encoders (LTX AV) override generate()
        # themselves and are left for the runtime check below.
        inner = getattr(cond, getattr(cond, "clip", ""), cond)
        transformer = getattr(inner, "transformer", None)
        if transformer is not None and not hasattr(transformer, "generate"):
            return unsupported
        return None

    @staticmethod
    def _split_thinking(generated_text):
        """Separate reasoning from the answer. Returns (thinking, prompt)."""
        thinking_parts = []
        remainder = generated_text

        # The common Gemma4 case. Its template primes an empty, pre-closed
        # thought channel to suppress reasoning, but the model routinely just
        # carries on writing inside it and closes it again itself. The opener
        # therefore sits in the prompt, not the output, so what comes back is
        # reasoning + an orphan </think> + the answer.
        lowered = remainder.lower()
        first_close = lowered.find("</think>")
        first_open = lowered.find("<think>")
        if first_close != -1 and (first_open == -1 or first_close < first_open):
            thinking_parts.append(remainder[:first_close])
            remainder = remainder[first_close + len("</think>"):]

        # Normal matched blocks.
        thinking_parts.extend(_THINK_RE.findall(remainder))
        remainder = _THINK_RE.sub("", remainder)

        # A think block truncated by max_length never gets its closing tag.
        # Treat everything after the opener as reasoning so it can't leak into
        # the prompt output.
        unclosed = _UNCLOSED_THINK_RE.search(remainder)
        if unclosed:
            thinking_parts.append(unclosed.group(1))
            remainder = remainder[:unclosed.start()]

        thinking = "\n\n".join(t.strip() for t in thinking_parts if t.strip())
        return thinking.strip(), remainder.strip()

    # ==================================================================================

    def enhance(self, clip, text, instruction, max_length, temperature, top_k, top_p, seed, thinking=False):
        if clip is None:
            raise ValueError("No CLIP provided. Load a text encoder with CLIPLoader.")

        error = self._generation_error(clip)
        if error:
            raise ValueError(error)

        prompt = f"{instruction.strip()}\n\n{text.strip()}" if instruction.strip() else text.strip()
        if not prompt:
            return ("", "")

        logger.info(f"Y7 Prompt Enhancer (Native): generating up to {max_length} tokens, thinking={thinking}")

        # min_length=1 mirrors core's TextGenerate: without it short prompts get
        # padded and the model sees trailing pad tokens.
        tokens = clip.tokenize(
            prompt,
            min_length=1,
            skip_template=False,
            thinking=thinking,
        )

        try:
            generated_ids = clip.generate(
                tokens,
                do_sample=temperature > 0.0,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=0.05,
                repetition_penalty=1.05,
                presence_penalty=0.0,
                seed=seed,
            )
        except AttributeError as e:
            # Composite encoders the static check above cannot resolve still
            # fail here, deep inside the encoder, with an opaque message.
            if "generate" in str(e):
                raise ValueError(
                    "This text encoder does not support text generation.\n"
                    "Supported: Gemma 3, Gemma 4, Qwen3, Qwen3.5, Qwen3-VL.\n"
                    f"(underlying error: {e})"
                )
            raise

        generated_text = clip.decode(generated_ids)
        thinking_output, enhanced_prompt = self._split_thinking(generated_text)

        if thinking_output:
            logger.info(f"Y7 Prompt Enhancer (Native): stripped {len(thinking_output)} chars of reasoning")
        if not enhanced_prompt:
            print("Y7 Prompt Enhancer (Native): model produced only reasoning - raise max_length or set temperature to 0", color.BRIGHT_YELLOW)

        return (thinking_output, enhanced_prompt)
