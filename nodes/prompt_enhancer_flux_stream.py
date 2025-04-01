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
from threading import Thread

# Y7Nodes_PromptEnhancerFlux

# .enhance()
# │
# ├── down_load_llm_model()
# │   ├── model_path_download_if_needed() [if model not cached]
# │   │   └── snapshot_download() [if model files missing]
# │   └── AutoModelForCausalLM.from_pretrained() + AutoTokenizer.from_pretrained()
# │
# ├── generate_t5_prompt()
# │   ├── get_t5_prompt_instruction()
# │   ├── format_chat_messages() or tokenizer.apply_chat_template()
# │   ├── model.generate() [with torch.inference_mode()]
# │   ├── tokenizer.decode()
# │   └── process_subject_override() [if square brackets present]
# │       └── extract_square_brackets()
# │
# ├── generate_clip_prompt()
# │   ├── get_clip_prompt_instruction()
# │   ├── format_chat_messages() or tokenizer.apply_chat_template()
# │   ├── model.generate() [with torch.inference_mode()]
# │   ├── tokenizer.decode()
# │   └── extract_square_brackets() [to check for subject override]
# │
# └── [Memory Management]
#     └── Offload model or keep loaded based on keep_model_loaded parameter

# function to detect Apple Silicon
def is_apple_silicon():    
    return platform.system() == "Darwin" and platform.machine().startswith(("arm", "M"))

# Function to check if CUDA is available
def is_cuda_available():
    return torch.cuda.is_available()

# ======================================
# ModelCache class - unchanged
class ModelCache:
    loaded_models = {}
    loaded_tokenizers = {}


# LLM model information - unchanged
LLM_MODELS = [
    ("OpenHermes-2.5-Mistral-7B", "teknium/OpenHermes-2.5-Mistral-7B"),
    ("Hermes-Trismegistus-Mistral-7B", "teknium/Hermes-Trismegistus-Mistral-7B"),
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
3. NEVER change "[agg woman]" into "agg-inspired woman" or any variation
4. The bracketed phrase MUST appear EXACTLY as written at the beginning of your description

CORRECT EXAMPLE:
User input: "[agg woman] in a cafe"
Correct T5 start: "agg woman sits in a dimly lit cafe..."
INCORRECT: "A woman with an agg-inspired look sits in a cafe..."

User input: "[ohwx man] smoking"
Correct T5 start: "ohwx man smoking a cigarette..."
INCORRECT: "A man with ohwx-style features smoking..."

User input: "[ohwx woman] sitting"
Correct T5 start: "ohwx mwoman sitting..."
INCORRECT: "A woman with the ohwx aesthetic"

If you see a bracketed subject, preserve it EXACTLY as provided as the subject - think of it as the subject's name.
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

2. CLIP PROMPT:
{PROMPT_CLIP_INSTRUCTIONS}

Respond in the following format:
[T5]
Your detailed T5 prompt here...
[/T5]

[CLIP]
Your concise CLIP prompt here...
[/CLIP]

The START AND END TAGS ARE IMPORTANT!
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
    
    # Create inputs - more memory efficient
    model_inputs = prompt_enhancer_tokenizer([formatted_text], return_tensors="pt")
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

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
    # T5 PROMPT
    # ==============================================================================
    # Extract T5 prompt
    t5_match = re.search(r'\[T5\](.*?)\[/T5\]', decoded_response, re.DOTALL)
    if t5_match:
        t5_prompt = t5_match.group(1).strip()
    else:
        # Fallback extraction if tags are missing
        print("Could not find T5 tags, attempting fallback extraction", color.YELLOW)
        parts = decoded_response.split("[CLIP]", 1)
        t5_prompt = parts[0].strip()
        if t5_prompt.startswith("T5 PROMPT:"):
            t5_prompt = t5_prompt[len("T5 PROMPT:"):].strip()
    
    # post process (remove articles if bracketed subject is used)
    t5_prompt = T5_PostProcess(user_prompt, t5_prompt)

    # ==============================================================================
    # CLIP PROMPT
    # ==============================================================================
    # Extract CLIP prompt
    clip_match = re.search(r'\[CLIP\](.*?)\[/CLIP\]', decoded_response, re.DOTALL)
    if clip_match:
        clip_prompt = clip_match.group(1).strip()
    else:
        # Fallback extraction if tags are missing
        print("Could not find CLIP tags, attempting fallback extraction", color.YELLOW)
        if "[CLIP]" in decoded_response:
            clip_prompt = decoded_response.split("[CLIP]", 1)[1].strip()
        elif "CLIP PROMPT:" in decoded_response:
            clip_prompt = decoded_response.split("CLIP PROMPT:", 1)[1].strip()
        else:
            # Last resort - try to find a comma-separated list after T5
            parts = decoded_response.split("[/T5]", 1)
            if len(parts) > 1:
                clip_prompt = parts[1].strip()
            else:
                clip_prompt = "Error: Could not extract CLIP prompt"
    
    # Clean up the prompts (remove any leftover tags or labels)
    t5_prompt = t5_prompt.replace("T5 PROMPT:", "").strip()
    clip_prompt = clip_prompt.replace("CLIP PROMPT:", "").replace("[/CLIP]", "").strip()
    
    # Final cleanup - remove brackets from final output
    final_t5 = t5_prompt.replace("[", "").replace("]", "").replace("\n\n", " ")
    final_clip = clip_prompt.replace("[", "").replace("]", "")
    

    # print(f"Extracted T5 prompt:\n{final_t5}\n", color.YELLOW)
    # print(f"Extracted CLIP prompt:\n{final_clip}", color.BRIGHT_YELLOW)
    
    return final_clip, final_t5

# ==================================================================================
def T5_PostProcess(user_prompt, t5_prompt):
    """
    Post-Process the T5 prompt based on the user's prompt.
    Ensures bracketed subjects from the user prompt appear at the beginning of the T5 prompt.
    And replaces subject where bracketed_subject does not appear in the T5 text (it happens)
    
    Args:
        user_prompt (str): The original user prompt
        t5_prompt (str): The generated T5 prompt
        
    Returns:
        str: The processed T5 prompt
    """
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
            common_starts = ["a woman", "a man", "a girl", "a boy"]
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
                "keep_model_loaded": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "If enabled, keeps the model loaded in VRAM for faster subsequent runs"}
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
        keep_model_loaded = kwargs.get("keep_model_loaded")

        # Default prompt if empty
        if not prompt.strip():
            t5xxl_prompt = 'Please provide a prompt, no matter how basic. If you wish to use a token or trigger words enclose them in square brackets.\nExamples:\n\n"A man sitting in a cafe".\n"[ohwx woman] standing in the middle of a busy street"'
            clip_l_prompt = ""
            return (clip_l_prompt, t5xxl_prompt,)
        
        try:
            # Force garbage collection before starting
            gc.collect()

            # For Apple Silicon, use MPS device if available
            if is_apple_silicon() and torch.backends.mps.is_available():
                print("Using MPS (Metal Performance Shaders) for Apple Silicon", color.BRIGHT_GREEN)
                load_device = "mps"
            elif is_cuda_available():
                print("Using CUDA device", color.BRIGHT_GREEN)
                load_device = comfy.model_management.get_torch_device()
                offload_device = comfy.model_management.unet_offload_device()                
            else:
                print("No GPU detected, using CPU (this will be slow)", color.YELLOW)
                load_device = "cpu"
                offload_device = "cpu"
            
            # Load/download the model - it will be placed on the correct device by down_load_llm_model
            llm_model, llm_tokenizer = self.down_load_llm_model(llm_display_name, load_device)
            
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

            # Memory management based on keep_model_loaded parameter
            if not keep_model_loaded:
                print("Cleaning up model...\n", color.BRIGHT_BLUE)
                
                # Platform-specific cleanup
                if is_apple_silicon():
                    # For Apple Silicon, just delete references and let GC handle it
                    if llm_display_name in ModelCache.loaded_models:
                        del ModelCache.loaded_models[llm_display_name]
                        del ModelCache.loaded_tokenizers[llm_display_name]
                    
                    # Force garbage collection
                    gc.collect()
                    if hasattr(torch.mps, 'empty_cache'):
                        torch.mps.empty_cache()
                
                elif is_cuda_available():
                    # For NVIDIA GPUs, we can safely move to CPU then clean up
                    try:
                        llm_model.to("cpu")
                    except Exception as e:
                        print(f"Warning: Could not move model to CPU: {str(e)}", color.YELLOW)
                    
                    if llm_display_name in ModelCache.loaded_models:
                        del ModelCache.loaded_models[llm_display_name]
                        del ModelCache.loaded_tokenizers[llm_display_name]
                    
                    # Force garbage collection and CUDA cache cleanup
                    gc.collect()
                    torch.cuda.empty_cache()
                    comfy.model_management.soft_empty_cache()
                
                else:
                    # CPU fallback
                    if llm_display_name in ModelCache.loaded_models:
                        del ModelCache.loaded_models[llm_display_name]
                        del ModelCache.loaded_tokenizers[llm_display_name]
                    gc.collect()
            else:
                print("Keeping model loaded.", color.BRIGHT_BLUE)
                ModelCache.loaded_models[llm_display_name] = llm_model
                ModelCache.loaded_tokenizers[llm_display_name] = llm_tokenizer
            
            return (clip_l_prompt, t5xxl_prompt,)
        
        # ========================================
        except Exception as e:
            print(f"❌ Error: {str(e)}", color.BRIGHT_RED)
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
    def down_load_llm_model(self, model_display_name, load_device):
        repo_path = get_repo_info(model_display_name)
        
        # Check cache first
        if model_display_name in ModelCache.loaded_models and model_display_name in ModelCache.loaded_tokenizers:
            print(f"Using cached model {model_display_name} from previous run", color.BRIGHT_GREEN)
            return ModelCache.loaded_models[model_display_name], ModelCache.loaded_tokenizers[model_display_name]
            
        # Download if needed
        model_path = self.model_path_download_if_needed(model_display_name)
                
        try:
            print(f"Loading model {model_display_name}", color.BRIGHT_BLUE)
            
            # Force garbage collection before loading model
            gc.collect()
            torch.cuda.empty_cache() if hasattr(torch.cuda, 'empty_cache') else None
            
            # Apply optimizations for Apple Silicon
            if is_apple_silicon():
                 # "Detected Apple Silicon...
                 # Use float16 instead of bfloat16 for Apple Silicon
                torch_dtype = torch.float16  
                
                # Load model with Apple-specific optimizations but DON'T use device_map="auto"
                llm_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                )
                
                # Explicitly move to the correct device
                if torch.backends.mps.is_available():
                    print("Using MPS backend for model", color.BRIGHT_GREEN)
                    llm_model = llm_model.to("mps")
                else:
                    print("MPS not available, using CPU", color.YELLOW)
                    llm_model = llm_model.to("cpu")

            elif is_cuda_available():
                # Detected CUDA device...                
                # Use native CUDA device from comfy
                cuda_device = comfy.model_management.get_torch_device()
                
                # Load model with CUDA optimizations
                llm_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                )
                
                # Explicitly move to CUDA device
                llm_model = llm_model.to(cuda_device)
            else:
                # Fallback for other devices
                # "No GPU detected, using CPU (this will be very slow)

                # Load model
                llm_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                )
                

            # llm_tokenizer
            llm_tokenizer = AutoTokenizer.from_pretrained(
                model_path,
            )

            
            # Cache model and tokenizer
            ModelCache.loaded_models[model_display_name] = llm_model
            ModelCache.loaded_tokenizers[model_display_name] = llm_tokenizer
            
            return llm_model, llm_tokenizer
            
        except (FileNotFoundError, ValueError) as e:
            print(f"❌ Error: Model files are incomplete or corrupted: {str(e)}", color.RED)
            print(f"🔄 Please manually delete the directory : {model_path}", color.YELLOW)
            print(f"🔄 Then re-launch the workflow to attempt downloading again.", color.YELLOW)
            print(f"🔄 Alternatively you can manually download the model from: \nhttps://huggingface.co/{repo_path}", color.YELLOW)                
            print(f"   and places the files to: {model_path}", color.YELLOW)
            
            raise RuntimeError(f"Model at {model_path} is incomplete or corrupted. Please delete this directory and try again.")
