import logging
import os
import platform
import gc
import hashlib
import re
from typing import List, Optional, Tuple, Union, Dict, Any
from ..utils.colored_print import color, style
import comfy.model_management
import comfy.model_patcher
import folder_paths
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, TextIteratorStreamer
from huggingface_hub import snapshot_download
import numpy as np
from threading import Thread

# function call sequence
# Y7Nodes_PromptEnhancerFlux
# .enhance()
# │
# ├── [Optional VRAM Cleanup]
# │   └── unload models, clear cache, clear internal cache
# │
# ├── down_load_llm_model()
# │   ├── Check cache (self._loaded_models)
# │   ├── [If cache miss]:
# │   │   ├── model_path_download_if_needed() -> snapshot_download()
# │   │   ├── AutoModelForCausalLM.from_pretrained() -> .to(device)
# │   │   ├── AutoTokenizer.from_pretrained()
# │   │   └── Store in cache
# │   └── Return model, tokenizer
# │
# ├── [Memory Management]
# │   └── get_model_size(), comfy.model_management.free_memory()
# │
# ├── generate_both_prompts()
# │   ├── Format input (apply_chat_template/format_chat_messages)
# │   ├── Tokenize input
# │   ├── model.generate() [via Thread + TextIteratorStreamer]
# │   ├── Decode streamed response
# │   ├── Extract T5/CLIP prompts
# │   ├── T5_PostProcess() -> extract_square_brackets()
# │   └── Return cleaned (clip_prompt, t5_prompt)
# │
# └── Return (clip_l_prompt, t5xxl_prompt)

# function to detect Apple Silicon
def is_apple_silicon():    
    return platform.system() == "Darwin" and platform.machine().startswith(("arm", "M"))

# Function to check if CUDA is available
def is_cuda_available():
    return torch.cuda.is_available()


# LLM model information - unchanged
LLM_MODELS = [    
    ("OpenHermes-2.5-Mistral-7B", "teknium/OpenHermes-2.5-Mistral-7B"),
    ("Hermes-Trismegistus-Mistral-7B", "teknium/Hermes-Trismegistus-Mistral-7B"),
    ("Dolphin3.0-Llama3.1-8B", "cognitivecomputations/Dolphin3.0-Llama3.1-8B")
]

LLM_DISPLAY_NAMES = [model[0] for model in LLM_MODELS]

def get_repo_info(display_name):
    for model_info in LLM_MODELS:
        if model_info[0] == display_name:
            return model_info[1]
    return None

# Required files definitions - unchanged
openhermes_2_5_mistral_7b_req_files = [
    "added_tokens.json",
    "config.json",
    "generation_config.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "transformers_inference.py"
]

hermes_trismegistus_mistral_7b_req_files = [
    "added_tokens.json",
    "config.json",
    "generation_config.json",
    "pytorch_model-00001-of-00002.bin",
    "pytorch_model-00002-of-00002.bin",
    "pytorch_model.bin.index.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "tokenizer_config.json"
]

dolphin3_0_llama3_1_8b_req_files = [
    "config.json",
    "generation_config.json",
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "trainer_state.json"
]

_MAX_NEW_TOKENS = 1280
MODELS_PATH_KEY = "LLM"
DEFAULT_PROMPT = ""

# ==================================================================================
# REVISED PROMPT INSTRUCTIONS
# ==================================================================================

# Base introduction and purpose
PROMPT_BASE = """
You are a Generative AI assistant specialized in generating comprehensive text-to-image prompts for 
the Flux image generation model. I'm going to ask you to create a prompt in two parts.
"""

# T5 Prompt instructions (Shorter)
# PROMPT_T5_INSTRUCTIONS_SHORTER = """
# First, I need you to create a T5 Prompt:

# This should be a highly detailed natural language description (up to 512 T5 tokens - aiming for closer to 450-500 words) with:

# - Subject always comes first: Start with the main subject(s): determined by the input prompt. this can be a person, object or scene or something else.
# - Subject Details: When applicable, describe physical appearance, pose, action, expression, attire
# - Scene Description: Overall setting, environment, background, visual style
# - Time & Place: Time of day, season, location
# - Lighting and color palette: Sources, intensity, direction, color temperature, shadows
# - Composition: Layout of elements and focal points
# - Mood & Atmosphere: Emotional tone using evocative language (but not too much of this!)
# - Do not start with, "in this image..." or similar assume that we know it is an image, go straight to the point!
# Bad:
# "In this image, a woman stands in..."
# "The image shows a woman who is..."

# Good:
# "A woman stands in..."

# - Use only positive descriptions — focus on what should appear in the image.
# - Avoid repetition of the subject, instead use she, he, her, his etc
# """

# T5 Prompt instructions (Longer)
PROMPT_T5_INSTRUCTIONS_LONGER = """
First, I need you to create a T5 Prompt:

This must be a **very rich and highly detailed** natural language description (up to 512 T5 tokens - aiming for closer to 450-500 words). **Elaborate significantly** on each aspect:

- Subject always comes first: Start with the main subject(s): determined by the input prompt. this can be a person, object or scene or something else.
- Subject Details: **Thoroughly describe** physical appearance, pose, action, expression, attire, including textures and materials.
- Scene Description: **Paint a vivid picture** of the overall setting, environment, background details, and visual style.
- Time & Place: Specify time of day, season, location with **atmospheric details**.
- Lighting and colors: Detail the light sources, intensity, direction, color temperature, shadows, and **how light interacts with surfaces**.
- Composition: Explain the layout of elements, focal points, and camera angle/perspective.
- Mood & Atmosphere: Convey the emotional tone using **evocative and descriptive language**.
- Do not start with, "in this image..." or similar assume that we know it is an image, go straight to the point!
Bad:
"In this image, a woman stands in..."
"The image shows a woman who is..."

Good:
"A woman stands in..."

- Use only positive descriptions — focus on what should appear in the image.
- Avoid repetition of the subject, instead use she, he, her, his etc
"""


# T5 example
PROMPT_T5_EXAMPLE = """
Example T5 Prompt:
A woman with shoulder-length black hair and luminous brown eyes stands alone in a dimly lit interior hallway. She wears a sleek, 
emerald green satin dress that catches the light with a subtle shimmer. Her posture is still, almost statuesque, as she gazes slightly 
off-camera with a pensive expression. One hand rests gently on a weathered wooden railing, while the other clutches a small vintage clutch. 
The hallway is narrow and elegant, lined with tall windows draped in sheer curtains that allow soft shafts of moonlight to filter in. 
Dust particles drift through the air, caught in the light. Ornate wall sconces cast warm amber glows along the wallpapered walls, 
creating deep shadows that frame the woman in dramatic contrast.
"""

# CLIP prompt instructions
PROMPT_CLIP_INSTRUCTIONS = """
Next, I need you to create a CLIP Prompt:

This should be a keyword-focused list (up to 70 CLIP TOKENS, aiming for roughly 45-55 words):
- Prioritized, comma-separated list of essential keywords. Aim for the target word count.
- Subject(s) MUST come first
- Include: subject(s), art style, setting, major visual features, mood, lighting, color scheme
- Include specific artistic terms if relevant (e.g., "soft focus," "cinematic lighting")
- Ensure alignment with the T5 prompt, focusing on the most impactful keywords but including secondary relevant details if space allows within the target word count.
- Avoid words that describe sounds or smells if they add nothing to the visuals.
- Do not start with the words 'CLIP Prompt:' - go straight to the keywords
- Aim for the target word count of 40-55 words. Add or remove keywords as needed to meet this range.
"""

# CLIP example
PROMPT_CLIP_EXAMPLE = """
Example CLIP Prompt:
woman, shoulder-length black hair, brown eyes, emerald satin dress, vintage clutch, hallway, dim, lighting, pensive, moonlight, 
sheer curtains, amber wall sconces, dust in air, shadows, soft focus, contemplative mood
"""

# Special subject override instruction
PROMPT_SPECIAL_OVERRIDE = """
CRITICAL SUBJECT INSTRUCTION: 
When a user's prompt contains a phrase inside square brackets like "[agg woman]", you MUST treat this EXACT text as the primary subject.

1. NEVER modify, interpret, or expand the bracketed phrase
2. NEVER add adjectives or descriptions to the bracketed phrase itself
3. DO NOT USE the bracketed phrase as a descriptor or an adjective
4. The bracketed phrase MUST appear EXACTLY as written at the beginning of your description
5. NEVER use any part of the bracketed phrase as a descriptor or adjective ANYWHERE in the text
6. NEVER refer to the style, aesthetic, or features of the bracketed phrase

CORRECT EXAMPLE:
User input: "[agg woman] in a cafe"
Correct T5 start: "agg woman sits in a dimly lit cafe..."
INCORRECT: "A woman with an agg-inspired look sits in a cafe..."
INCORRECT: "agg woman sits in a cafe. Her features reflect the agg style..."

User input: "[ohwx man] smoking"
Correct T5 start: "ohwx man smoking a cigarette..."
INCORRECT: "A man with ohwx-style features smoking..."
INCORRECT: "ohwx man smoking a cigarette. His appearance has the ohwx aesthetic..."

User input: "[ohwx woman] sitting"
Correct T5 start: "ohwx woman sitting..."
INCORRECT: "A woman with an ohwx aesthetic sitting..."
INCORRECT: "A woman with a distinct ohwx style..." 
INCORRECT: "ohwx woman sitting... with features inspired by the ohwx style..."

If you see a bracketed subject, preserve it EXACTLY as provided as the subject - think of it as the subject's name. DO NOT reference the bracketed term anywhere else in the description.
"""

# ==================================================================================
# SUBJECT EXTRACTION HELPER FUNCTIONS
# ==================================================================================

def extract_square_brackets(text):
    """Extract content inside square brackets."""
    if not text:
        return None
    
    match = re.search(r'\[(.*?)\]', text)
    if match:
        return match.group(1)
    return None

# ==================================================================================
# COMBO & SEPARATE GENERATION FUNCTIONS
# ==================================================================================

# COMBO generation func.
def generate_both_prompts(
                        prompt_enhancer_model, 
                        prompt_enhancer_tokenizer, 
                        user_prompt: str, 
                        seed: int = None, 
                        temperature: float = 0.7,
                        top_p: float = 0.9,
                        top_k: int = 40,
                        max_new_tokens: int = _MAX_NEW_TOKENS) -> tuple:
    
    """
    Generate both T5 and CLIP prompts in a single model call.
    Returns a tuple of (clip_prompt, t5_prompt)
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Combined instruction that asks for both prompts in a single generation
    combined_instruction = f"""

{PROMPT_BASE}

SPECIAL PROMPT OVERRIDE:
{PROMPT_SPECIAL_OVERRIDE}

I need you to create two different prompts for the same image:

1. T5 PROMPT:
{PROMPT_T5_INSTRUCTIONS_LONGER}

SPECIAL PROMPT OVERRID REMINDER:
{PROMPT_SPECIAL_OVERRIDE}

2. CLIP PROMPT:
{PROMPT_CLIP_INSTRUCTIONS}

**CRITICAL: You MUST provide BOTH a T5 and a CLIP prompt, enclosed in their respective tags.** Failure to provide both sections will result in an error.

Respond **EXACTLY** in the following format, including the tags:
[T5]
Your detailed T5 prompt here...
[/T5]

[CLIP]
Your concise CLIP prompt here...
[/CLIP]

**REMINDER: Both [T5]...[/T5] and [CLIP]...[/CLIP] sections are mandatory.**
"""
    
    # Create messages with combined instructions
    messages = [
        {"role": "system", "content": combined_instruction},
        {"role": "user", "content": user_prompt.strip() if user_prompt.strip() else DEFAULT_PROMPT},
    ]
    
    # Format messages
    try:
        if hasattr(prompt_enhancer_tokenizer, 'chat_template') and prompt_enhancer_tokenizer.chat_template is not None:
            # print("Using tokenizer's built-in chat template for combined prompt", color.BRIGHT_GREEN)
            formatted_text = prompt_enhancer_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # print("Using custom formatting for combined prompt", color.YELLOW)
            formatted_text = format_chat_messages(messages, add_generation_prompt=True)
    except Exception as e:
        print(f"Error formatting messages for combined prompt: {str(e)}", color.RED)
        formatted_text = format_chat_messages(messages, add_generation_prompt=True)
    
    # Free memory before tokenization
    gc.collect()
    
    # Get device information from model
    device = prompt_enhancer_model.device
    device_type = device.type if hasattr(device, 'type') else str(device)
    # print(f"Model is on device: {device_type}", color.BRIGHT_GREEN)
    
    # Create inputs 
    model_inputs = prompt_enhancer_tokenizer([formatted_text], return_tensors="pt")
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

    # === Print token count and context window size ===
    input_token_count = model_inputs["input_ids"].shape[-1]
    context_window_size = prompt_enhancer_model.config.max_position_embeddings # Use model config for accurate context window
    print(f"\n--- Prompt Enhancer Token Info ---", color.BRIGHT_MAGENTA)
    print(f"Input Prompt Token Count: {input_token_count}", color.BRIGHT_MAGENTA)
    print(f"Model Context Window: {context_window_size}", color.BRIGHT_MAGENTA)
    print(f"----------------------------------\n", color.BRIGHT_MAGENTA)


    # Create a streamer, skip the input prompt (instructions)
    streamer = TextIteratorStreamer(prompt_enhancer_tokenizer, skip_prompt=True, skip_special_tokens=True, stream_every=2)

    # Set up generation kwargs
    generation_kwargs = {
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs.get("attention_mask", None),
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "streamer": streamer,
    }

    # Start generation in a separate thread
    print("🔄 Starting prompt generation...", color.BRIGHT_BLUE)
    thread = None # Initialize thread to None

    # Apply platform-specific optimizations
    try:
        if is_apple_silicon() and device_type == "mps":
            with torch.inference_mode(), torch.autocast("mps"):
                thread = Thread(target=prompt_enhancer_model.generate, kwargs=generation_kwargs)
                thread.start()
        elif device_type == "cuda":
            with torch.inference_mode(), torch.amp.autocast(device_type="cuda"):
                thread = Thread(target=prompt_enhancer_model.generate, kwargs=generation_kwargs)
                thread.start()
        else:
            with torch.inference_mode():
                thread = Thread(target=prompt_enhancer_model.generate, kwargs=generation_kwargs)
                thread.start()
    except Exception as e:
        print(f"Error with optimized generation: {str(e)}, falling back to standard mode", color.YELLOW)
        with torch.inference_mode():
            thread = Thread(target=prompt_enhancer_model.generate, kwargs=generation_kwargs)
            thread.start()

    # Process the streamed tokens
    decoded_response = ""
    print("📝 Generating: ", end="", flush=True)
    token_count = 0
    printed_something = False # Flag to track if we've started printing actual content

    if thread: # Check if thread was successfully created and started
        for token in streamer:
            decoded_response += token
            if token: # Only print non-empty tokens
                print(token, end="", flush=True)
                printed_something = True
                token_count += 1 # Only count printed tokens
                if token_count > 0 and token_count % 50 == 0:  # Add a newline every 50 *printed* tokens for readability
                    print() # Use print() for newline, flush is handled

        thread.join() # Wait for the generation thread to finish
        if printed_something: # Add a final newline only if something was printed
             print() # Ensure the completion message is on a new line
        print("✅ Generation complete.  Post-processing...", color.BRIGHT_GREEN)
    else:
        print("\n❌ Generation thread failed to start.", color.BRIGHT_RED)
        return "Error: Generation failed", "Error: Generation failed"

    # Clean up memory explicitly
    del model_inputs, generation_kwargs, streamer, thread
    gc.collect()

    # print(f"Combined raw response:\n{decoded_response}\n", color.ORANGE)
    
    # ==============================================================================
    # T5 PROMPT EXTRACTION (Robust)
    # ==============================================================================
    # Regex to find [T5] and capture content until [/T5], [CLIP], or end of string
    t5_match = re.search(r'\[T5\](.*?)(?:\[/T5\]|\[CLIP\]|$)', decoded_response, re.DOTALL)
    if t5_match:
        t5_prompt = t5_match.group(1).strip()
        # Remove potential leading "T5 PROMPT:" label if present
        if t5_prompt.upper().startswith("T5 PROMPT:"):
             t5_prompt = t5_prompt[len("T5 PROMPT:"):].strip()
    else:
        print("Could not find [T5] tag. T5 prompt extraction failed.", color.YELLOW)
        t5_prompt = "Error: Could not extract T5 prompt" # Set error message

    # Post process (remove articles if bracketed subject is used)
    if not t5_prompt.startswith("Error:"): # Only post-process if extraction was successful
        t5_prompt = T5_PostProcess(user_prompt, t5_prompt)

    # ==============================================================================
    # CLIP PROMPT EXTRACTION (Robust)
    # ==============================================================================
    # Regex to find [CLIP] and capture content until [/CLIP] or end of string
    clip_match = re.search(r'\[CLIP\](.*?)(?:\[/CLIP\]|$)', decoded_response, re.DOTALL)
    if clip_match:
        clip_prompt = clip_match.group(1).strip()
        # Remove potential leading "CLIP PROMPT:" label if present
        if clip_prompt.upper().startswith("CLIP PROMPT:"):
            clip_prompt = clip_prompt[len("CLIP PROMPT:"):].strip()
    else:
        print("Could not find [CLIP] tag. CLIP prompt extraction failed.", color.YELLOW)
        clip_prompt = "Error: Could not extract CLIP prompt" # Set error message

    # Clean up the prompts (remove any leftover tags or labels only if not error)
    if not t5_prompt.startswith("Error:"):
        t5_prompt = t5_prompt.replace("[T5]", "").replace("[/T5]", "").strip() # Remove tags just in case
    if not clip_prompt.startswith("Error:"):
        clip_prompt = clip_prompt.replace("[CLIP]", "").replace("[/CLIP]", "").strip() # Remove tags just in case
    
    # Final cleanup - remove brackets from final output
    final_t5 = t5_prompt.replace("[", "").replace("]", "").replace("\n\n", " ")
    final_clip = clip_prompt.replace("[", "").replace("]", "")
    

    # print(f"Extracted T5 prompt:\n{final_t5}\n", color.YELLOW)
    # print(f"Extracted CLIP prompt:\n{final_clip}", color.BRIGHT_YELLOW)
    
    return final_clip, final_t5

# ==================================================================================
def T5_PostProcess(user_prompt, t5_prompt):
    # Post-Process the T5 prompt response:
    # Ensures bracketed subjects from the user prompt appear at the beginning of the T5 prompt.
    # Args:
    #     user_prompt (str): The original user prompt
    #     t5_prompt (str): The generated T5 prompt        

    # Check if user_prompt contains a bracketed subject
    bracketed_subject = extract_square_brackets(user_prompt)
    
    if bracketed_subject:
        # Remove square brackets from the extracted subject
        non_bracketed_subject = bracketed_subject.strip()
        
        # Check if the clean subject appears in t5_prompt
        if non_bracketed_subject.lower() in t5_prompt.lower():            
            # Bracketed subject found in T5 prompt, checking for indefinite/definite articles

            # Match articles "A", "An", or "The" only if it's directly before the bracketed subject anywhere in the string
            pattern = rf'\b(?:A|An|The)\s+(?={re.escape(non_bracketed_subject)})'
            t5_prompt = re.sub(pattern, '', t5_prompt, flags=re.IGNORECASE)
        
            # remove  from the start ("A", "An", "The")
            # t5_prompt = re.sub(r'^(?:A|An|The)\s+', '', t5_prompt, flags=re.IGNORECASE)

            return t5_prompt
        else:
            # Check if t5_prompt starts with common generic subjects
            # common_starts = ["a woman", "a man", "a girl", "a boy"]
            common_starts = ["a woman", "a man", "a girl", "a boy", "the woman", "the man", "the girl", "the boy"]
            t5_prompt_lower = t5_prompt.lower()
            
            for start in common_starts:
                if t5_prompt_lower.startswith(start):
                    # Find the actual case in the original string
                    start_length = len(start)
                    original_start = t5_prompt[:start_length]
                    
                    # Replace the generic beginning with the bracketed subject
                    print(f"Replacing '{original_start}' with '{non_bracketed_subject}' at beginning of T5 prompt", color.YELLOW)
                    return t5_prompt.replace(original_start, non_bracketed_subject, 1)  # Replace only the first occurrence
            
            # If no common generic beginning found, just prepend the clean subject
            # print(f"Adding bracketed subject '{non_bracketed_subject}' to beginning of T5 prompt", color.YELLOW)
            return f"{non_bracketed_subject} {t5_prompt}"
    
    # If no bracketed subject or it's already in the prompt, return the original
    return t5_prompt

# ==================================================================================
# Helper function to format chat messages - unchanged
def format_chat_messages(messages, add_generation_prompt=True):
    """Format chat messages into a single string."""
    formatted_text = ""
    
    for message in messages:
        role = message.get("role", "").lower()
        content = message.get("content", "")
        
        if role == "system":
            formatted_text += f"System: {content}\n\n"
        elif role == "user":
            formatted_text += f"User: {content}\n\n"
        elif role == "assistant":
            formatted_text += f"Assistant: {content}\n\n"
        else:
            formatted_text += f"{content}\n\n"
    
    if add_generation_prompt:
        formatted_text += "Assistant: "
    
    return formatted_text

# Helper function to calculate model size - unchanged
def get_model_size(model):
    """Calculate the memory size of a model based on parameters and buffers."""
    total_size = sum(p.numel() * p.element_size() for p in model.parameters())
    total_size += sum(b.numel() * b.element_size() for b in model.buffers())
    return total_size

# ==================================================================================
# MAIN NODE CLASS
# ==================================================================================
class Y7Nodes_PromptEnhancerFlux:
    # Class variable to cache loaded models and tokenizers, so they aren't needed to be loaded
    # from disk each time.  these are cleared if unload_models_before_run is true
    _loaded_models: Dict[str, Tuple[Any, Any]] = {}
    _last_used_model_name: Optional[str] = None # Stores the name of the last used model - used to know when to flush cache

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": DEFAULT_PROMPT
                    }
                ),
                "llm_name": (
                    LLM_DISPLAY_NAMES,
                    {
                        "default": LLM_DISPLAY_NAMES[0],
                        "tooltip": "Select LLM model."
                    }
                ),                
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.1, "max": 2.0, "step": 0.1, 
                     "tooltip": "Controls randomness: Higher values produce more diverse outputs"}
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 0.9, "min": 0.8, "max": 1.0, "step": 0.01, 
                     "tooltip": "Nucleus sampling: limits tokens to the most probable ones"}
                ),
                "top_k": (
                    "INT",
                    {"default": 40, "min": 22, "max": 100, 
                     "tooltip": "Limits token selection to the k most likely next tokens"}
                ),
                "seed": (
                    "INT", 
                    {"default": 0, "min": 0, "max": 0xffffffffffffffff}
                ),
                "unload_models_before_run": (
                    "BOOLEAN", 
                    {"default": True, 
                     "tooltip": "Frees up Memory (unload all models) before each run."}
                ),                
            }, 
            "hidden":{}
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("clip_l_prompt", "t5xxl_prompt",)
    FUNCTION = "enhance"
    CATEGORY = "Y7Nodes/Prompt"
    OUTPUT_NODE = False
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        prompt = kwargs.get("prompt", "")
        llm_display_name = kwargs.get("llm_name", "")                
        seed = kwargs.get("seed", 0)
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 0.9)
        top_k = kwargs.get("top_k", 40)
        
        input_string = f"{prompt}_{llm_display_name}_{seed}_{temperature}_{top_p}_{top_k}"
        hash_object = hashlib.md5(input_string.encode())
        hash_hex = hash_object.hexdigest()
        
        # print(f"IS_CHANGED hash = {hash_hex}", color.ORANGE)
        return hash_hex
        
    # ==================================================================================
    # main function
    # ==================================================================================
    def enhance(self, **kwargs):
        prompt = kwargs.get("prompt")
        llm_display_name = kwargs.get("llm_name")
        seed = kwargs.get("seed")
        temperature = kwargs.get("temperature")
        top_p = kwargs.get("top_p")
        top_k = kwargs.get("top_k")
        unload_models_before_run = kwargs.get("unload_models_before_run")

        # === Check if model changed from the one last used and clear cache if needed ===
        if self._last_used_model_name is not None and self._last_used_model_name != llm_display_name:
            print(f"ℹ️ Model changed from '{self._last_used_model_name}' to '{llm_display_name}'. Clearing internal cache to prevent OOM.", color.YELLOW)
            if hasattr(self, '_loaded_models'):
                self._loaded_models.clear()
            gc.collect() # Force garbage collection after clearing cache
        

        # Default prompt if empty
        if not prompt.strip():
            t5xxl_prompt = 'Please provide a prompt, no matter how basic. If you wish to use a token or trigger words enclose them in square brackets.\nExamples:\n\n"A man sitting in a cafe".\n"[ohwx woman] standing in the middle of a busy street"'
            clip_l_prompt = ""
            return (clip_l_prompt, t5xxl_prompt,)
        
        try:
            # ================= VRAM Cleanup =================
            # if unload_models_before_run is on, then flush the vram and cache
            # this helps if this node is used in a larger workflow that contains other model loaders.
            if unload_models_before_run == True:
                # If FREE_VRAM_BEFORE_RUNNING is True
                print("unloading models and clearing cache...", color.YELLOW)

                try:
                    # Use Comfy's function to unload models potentially loaded by other nodes
                    comfy.model_management.unload_all_models() 
                    gc.collect() # Garbage collect first
                    if is_cuda_available():
                        torch.cuda.empty_cache()
                        comfy.model_management.soft_empty_cache() # Added for good measure
                        print("   - CUDA cache cleared.", color.YELLOW)
                    elif is_apple_silicon() and hasattr(torch.mps, 'empty_cache'):
                        torch.mps.empty_cache()
                        print("   - MPS cache cleared.", color.YELLOW)
                    # Clear the node's internal cache as well
                    if hasattr(self, '_loaded_models'):
                        print("   - Clearing internal node model cache.", color.YELLOW)
                        self._loaded_models.clear()
                    print("✅ VRAM cleanup attempt finished.", color.YELLOW)
                except Exception as cleanup_error:
                    print(f"⚠️ Warning during VRAM cleanup: {str(cleanup_error)}", color.YELLOW)
            # =============== End VRAM Cleanup ===============

            # Force garbage collection before starting (kept original gc.collect)
            gc.collect()

            # For Apple Silicon, use MPS device if available
            if is_apple_silicon() and torch.backends.mps.is_available():
                print("Using MPS (Metal Performance Shaders) for Apple Silicon", color.BRIGHT_GREEN)
                load_device = "mps"
            elif is_cuda_available():
                print("Using CUDA device", color.BRIGHT_GREEN)
                load_device = comfy.model_management.get_torch_device()                
            else:
                print("No GPU detected, using CPU (this will be slow)", color.YELLOW)
                load_device = "cpu"
                            
            # Load/download the model - it will be placed on the correct device by down_load_llm_model
            llm_model, llm_tokenizer = self.down_load_llm_model(llm_display_name, load_device)
            self._last_used_model_name = llm_display_name # Update the last used model name
            
            # Calculate model size for memory management
            model_size = get_model_size(llm_model) 
            if is_apple_silicon():
                # Use a smaller buffer on Apple Silicon to avoid over-allocation
                model_size += 536870912  # Add 512MB as buffer instead of 1GB
            else:
                model_size += 1073741824  # Add 1GB as buffer for other platforms
                comfy.model_management.free_memory(model_size, load_device)
                        
            print(f"Seed={seed}", color.BRIGHT_BLUE)

            # Verify model is on the correct device
            if hasattr(llm_model, 'device'):
                print(f"Model is on device: {llm_model.device}", color.BRIGHT_GREEN)
            else:
                print(f"Model device info not available", color.YELLOW)

            # ======================================================================    
            # Generate both prompts in a single model call
            print("Generating both T5 and CLIP prompts...", color.BRIGHT_BLUE)
            
            clip_l_prompt, t5xxl_prompt = generate_both_prompts(
                                                            llm_model, 
                                                            llm_tokenizer, 
                                                            prompt, 
                                                            seed, 
                                                            temperature=temperature, 
                                                            top_p=top_p, 
                                                            top_k=top_k
                                                        )        

            # ======================================================================


            # print("Cleaning up model...\n", color.BRIGHT_BLUE)
            
            # First, try to move model to CPU regardless of platform
            # try:
            #     llm_model.to("cpu")
            # except Exception as e:
            #     print(f"Warning: Could not move model to CPU: {str(e)}", color.YELLOW)
            
            # # Clear any references to the model and tokenizer
            # del llm_model
            # del llm_tokenizer
            
            # # # Platform-specific cleanup
            # if is_apple_silicon():
            #     # For Apple Silicon
            #     gc.collect()
            #     if hasattr(torch.mps, 'empty_cache'):
            #         torch.mps.empty_cache()
            
            # elif is_cuda_available():
            #     # For NVIDIA GPUs
            #     gc.collect()
            #     torch.cuda.empty_cache()
            #     comfy.model_management.soft_empty_cache()
            #     # More aggressive CUDA cleanup
            #     comfy.model_management.cleanup_models()
            
            # else:
            #     # CPU fallback
            #     gc.collect()
            
            # # Final garbage collection pass
            gc.collect()
            
            return (clip_l_prompt, t5xxl_prompt,)

        # ========================================
        except RuntimeError as runtime_e:
            # Catch specific allocation errors first
            error_message = str(runtime_e)
            if "Allocation on device" in error_message:
                print(f"❌ Critical Error: Device Allocation Failed: {error_message}", color.BRIGHT_RED)
                # Re-raise the exception to stop ComfyUI workflow execution
                raise runtime_e
            else:
                # Re-raise other RuntimeErrors to be caught by the general Exception handler below
                # or propagate up if not caught later.
                raise runtime_e
        except Exception as e:
            # General exception handler
            print(f"❌ Error: {str(e)}", color.BRIGHT_RED)
            # This return allows the workflow to continue, which might be undesirable
            # depending on the error. Consider re-raising 'e' here too if errors
            # should generally halt the workflow.
            return (
                f"Error: {str(e)}",
                f"Error: Please check the model output format"
            )
    
    # ==================================================================================
    # Model download and loading methods - unchanged functionality but optimized
    def model_path_download_if_needed(self, model_display_name):
        repo_path = get_repo_info(model_display_name)
        llm_model_directory = os.path.join(folder_paths.models_dir, MODELS_PATH_KEY)
        os.makedirs(llm_model_directory, exist_ok=True)

        model_name = repo_path.rsplit("/", 1)[-1]
        full_model_path = os.path.join(llm_model_directory, model_name)

        if not os.path.exists(full_model_path):
            print(f"⬇️ Downloading model {repo_path} from HF to models/LLM. This may take a while.", color.YELLOW)
            try:
                # Select the correct file list based on model
                if "OpenHermes-2.5-Mistral-7B" in repo_path:
                    print(f"ℹ️ Downloading {repo_path} (≈14.5GB)", color.BRIGHT_BLUE)
                    allow_patterns = openhermes_2_5_mistral_7b_req_files
                elif "Hermes-Trismegistus-Mistral-7B" in repo_path:
                    print(f"ℹ️ Downloading {repo_path} (≈14.5GB)", color.BRIGHT_BLUE)
                    allow_patterns = hermes_trismegistus_mistral_7b_req_files
                # <<< ADDED BLOCK START >>>
                elif "Dolphin3.0-Llama3.1-8B" in repo_path:
                    # Note: Add estimated size if known, otherwise omit or estimate
                    print(f"ℹ️ Downloading {repo_path} (≈16.1GB)", color.BRIGHT_BLUE)
                    allow_patterns = dolphin3_0_llama3_1_8b_req_files
                # <<< ADDED BLOCK END >>>

                snapshot_download(
                    repo_id=repo_path,
                    local_dir=full_model_path,
                    allow_patterns=allow_patterns                    
                )
                print(f"✅ Model {repo_path} downloaded successfully.", color.BRIGHT_GREEN)
            except Exception as e:
                print(f"❌ Error downloading model {repo_path}: {str(e)}", color.BRIGHT_RED)
                raise
        else:
            # Check for missing files
            missing_files = []
            required_files = []

            if model_display_name == "OpenHermes-2.5-Mistral-7B":
                required_files = openhermes_2_5_mistral_7b_req_files
            elif model_display_name == "Hermes-Trismegistus-Mistral-7B":
                required_files = hermes_trismegistus_mistral_7b_req_files
            elif model_display_name == "Dolphin3.0-Llama3.1-8B":
                 required_files = dolphin3_0_llama3_1_8b_req_files


            for file in required_files:
                if not os.path.exists(os.path.join(full_model_path, file)):
                    missing_files.append(file)
            
            if missing_files:
                print(f"ℹ️ Found {repo_path} directory but missing files: {', '.join(missing_files)}", color.YELLOW)
                print(f"⬇️ Downloading missing files for {repo_path}", color.YELLOW)
                try:
                    snapshot_download(
                        repo_id=repo_path,
                        local_dir=full_model_path,
                        allow_patterns=missing_files
                    )
                    print(f"✅ Missing files for {repo_path} downloaded successfully!", color.BRIGHT_GREEN)
                except Exception as e:
                    print(f"❌ Error downloading missing files for {repo_path}: {str(e)}", color.BRIGHT_RED)
                    raise
            else:
                print(f"✅ All required files for {repo_path} found.", color.BRIGHT_GREEN)

        return full_model_path

    # ==============================================================================================
    # down_load_llm_model (or load from Cache)
    # ==============================================================================================
    def down_load_llm_model(self, model_display_name, load_device):
        repo_path = get_repo_info(model_display_name)
        cache_key = f"{model_display_name}_{load_device}" # Cache key includes device

        # Check cache first
        if cache_key in self._loaded_models:
            print(f"✅ Using cached model {model_display_name} on device {load_device}", color.BRIGHT_GREEN)
            llm_model, llm_tokenizer = self._loaded_models[cache_key]
            # Ensure model is on the requested device (might have changed if Comfy moved things)
            if llm_model.device != torch.device(load_device):
                 print(f"⚠️ Cached model was on {llm_model.device}, moving to {load_device}", color.YELLOW)
                 try:
                     llm_model = llm_model.to(load_device)
                     self._loaded_models[cache_key] = (llm_model, llm_tokenizer) # Update cache with correct device
                 except Exception as e:
                     print(f"❌ Error moving cached model to {load_device}: {e}. Reloading.", color.RED)
                     del self._loaded_models[cache_key] # Remove invalid cache entry
                     # Proceed to load model below
            else:
                # Model is already on the correct device in cache
                return llm_model, llm_tokenizer

        # If not in cache or moved device failed, load it
        print(f"🔄 Loading model {model_display_name} to device {load_device}", color.BRIGHT_BLUE)
        model_path = self.model_path_download_if_needed(model_display_name)

        try:
            # Force garbage collection before loading model
            gc.collect()
            if is_cuda_available():
                torch.cuda.empty_cache()
            elif is_apple_silicon() and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()

            # Determine torch_dtype based on device
            if is_apple_silicon():
                torch_dtype = torch.float16
            else: # CUDA or CPU
                torch_dtype = torch.bfloat16 # Use bfloat16 for CUDA/CPU

            # Load model with low_cpu_mem_usage=True to potentially reduce host RAM usage during loading
            llm_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True, # Helps especially when loading large models
            )

            # load model to device
            llm_model = llm_model.to(load_device)
            print(f"   Model loaded on {llm_model.device}", color.GREEN)

            # Load tokenizer
            llm_tokenizer = AutoTokenizer.from_pretrained(model_path)

            # Store in cache
            print(f"   Caching model {model_display_name} for device {load_device}", color.BLUE)
            self._loaded_models[cache_key] = (llm_model, llm_tokenizer)

            return llm_model, llm_tokenizer

        except (FileNotFoundError, ValueError) as e:
            print(f"❌ Error: Model files are incomplete or corrupted: {str(e)}", color.RED)
            print(f"🔄 Please manually delete the directory : {model_path}", color.YELLOW)
            print(f"🔄 Then re-launch the workflow to attempt downloading again.", color.YELLOW)
            print(f"🔄 Alternatively you can manually download the model from: \nhttps://huggingface.co/{repo_path}", color.YELLOW)
            print(f"   and places the files to: {model_path}", color.YELLOW)
            raise RuntimeError(f"Model at {model_path} is incomplete or corrupted. Please delete this directory and try again.")
        except Exception as e:
            print(f"❌ Unexpected error loading model {model_display_name}: {str(e)}", color.RED)
            # Attempt to clean up potentially partially loaded model from cache if it exists
            if cache_key in self._loaded_models:
                del self._loaded_models[cache_key]
            gc.collect()
            if is_cuda_available():
                torch.cuda.empty_cache()
            elif is_apple_silicon() and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            raise # Re-raise the exception after cleanup attempt
