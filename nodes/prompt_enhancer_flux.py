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
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from huggingface_hub import snapshot_download
import numpy as np

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

_MAX_NEW_TOKENS = 1024
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

# T5 Prompt instructions
PROMPT_T5_INSTRUCTIONS = """
First, I need you to create a T5 Prompt:

This should be a detailed natural language description (up to 512 T5 tokens - maybe max 400 words) with:

- Do not start with, "in this image..." or similar assume that we know it is an image, go straight to the point!
Bad:
"In this image, a woman stands in..."
"The image shows a woman who is..."

Good:
"A woman stands in..."

- Subject always comes first: Start with the main subject(s): determined by the input prompt. this can be a person, object or scene or something else. 
- Subject Details: When applicable, describe physical appearance, pose, action, expression, attire
- Scene Description: Overall setting, environment, background, visual style
- Time & Place: Time of day, season, location
- Lighting and color palette: Sources, intensity, direction, color temperature, shadows
- Composition: Layout of elements and focal points
- Mood & Atmosphere: Emotional tone using evocative language (but not too much of this!)
- Avoid purple prose - overly elaborate, flowery, or excessively descriptive - this is bad
- Minimize words that describe sounds or smells if they add nothing to the visuals. 
- Avoid descriptions of motion and action (unless important to the subject's pose)

Use only positive descriptions — focus on what should appear in the image.
Avoid repetition of the subject, instead use she, he, her, his etc
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

This must be a concise, short, less verbose keyword list (up to 70 CLIP TOKENS which roughly equates to 40-50 words):
- Prioritized, comma-separated list of essential keywords, but word limit priority is crucial.
- Subject(s) MUST come first
- Include: subject(s), art style, setting, major visual features, mood, lighting, color scheme
- Include specific artistic terms if relevant (e.g., "soft focus," "cinematic lighting")
- Ensure alignment with the T5 prompt, but remove any extraneous details - it does not need to be exhaustive.
- Avoid words that describe sounds or smells if they add nothing to the visuals. 
- Do not start with the words 'CLIP Prompt:' - go straight to the keywords
- Remember to keep the word count under 40. - remove keywords if they go over the limit
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

If you see a bracketed subject, preserve it EXACTLY as provided.
"""

# Combine sections for T5 instruction
def get_t5_prompt_instruction():
    return (
        PROMPT_BASE + "\n\n" + 
        PROMPT_SPECIAL_OVERRIDE + "\n\n" + 
        PROMPT_T5_INSTRUCTIONS + "\n\n" + 
        PROMPT_T5_EXAMPLE
    )

# Combine sections for CLIP instruction
def get_clip_prompt_instruction():
    return (
        PROMPT_CLIP_INSTRUCTIONS + "\n\n" + 
        PROMPT_CLIP_EXAMPLE + "\n\n" +
        "Create a CLIP prompt that perfectly complements the T5 prompt I generated earlier."
    )

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

def process_subject_override_dumb_version(short_prompt, generated_text):
    """
    Process subject override if square brackets are present in the original short prompt.
    Apply post-processing to ensure override is applied correctly.
    short_prompt could be:
                        "[ohwx man] sitting in a cafe" (bracketed_subject exists)
                        "a man sitting in a cafe" (no bracketed_subject)
    generated_text: either the T5 or CLIP generated response 
    """

    print(f"prompt=\n{short_prompt}\n\ngenerated_text=\n{generated_text}", color.ORANGE)

    # get the token or trigger word from the sqare brackets
    bracketed_subject = extract_square_brackets(short_prompt)
    
    # If no bracketed subject found, then just return original T5 or CLIP response text as is
    if not bracketed_subject:
        return generated_text
    
    # otherwise bracketed_subject subject exists
    # If bracketed subject exists, check if it is already in the generated text
    if bracketed_subject.lower() in generated_text.lower():
        # if so then return the generated text as is.  no changes. 
        print(f"Subject '{bracketed_subject}' already present in generated text", color.BRIGHT_GREEN)
        return generated_text
    
    # If not present, prepend the subject to the generated text
    print(f"Applying subject override: '{bracketed_subject}'", color.YELLOW)
    return f"{bracketed_subject} {generated_text}"

# ==================================================================================
# COMBO & SEPARATE GENERATION FUNCTIONS
# ==================================================================================

# COMBO generation func.
def generate_both_prompts(
                        prompt_enhancer_model, 
                        prompt_enhancer_tokenizer, 
                        prompt: str, 
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
{PROMPT_T5_INSTRUCTIONS}

2. CLIP PROMPT:
{PROMPT_CLIP_INSTRUCTIONS}

Respond in the following format:
[T5]
Your detailed T5 prompt here...
[/T5]

[CLIP]
Your concise CLIP prompt here...
[/CLIP]
"""
    
    # Create messages with combined instructions
    messages = [
        {"role": "system", "content": combined_instruction},
        {"role": "user", "content": prompt.strip() if prompt.strip() else DEFAULT_PROMPT},
    ]
    
    # Format messages
    try:
        if hasattr(prompt_enhancer_tokenizer, 'chat_template') and prompt_enhancer_tokenizer.chat_template is not None:
            print("Using tokenizer's built-in chat template for combined prompt", color.BRIGHT_GREEN)
            formatted_text = prompt_enhancer_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            print("Using custom formatting for combined prompt", color.YELLOW)
            formatted_text = format_chat_messages(messages, add_generation_prompt=True)
    except Exception as e:
        print(f"Error formatting messages for combined prompt: {str(e)}", color.RED)
        formatted_text = format_chat_messages(messages, add_generation_prompt=True)
    
    # Free memory before tokenization
    gc.collect()
    
    # Get device information from model
    device = prompt_enhancer_model.device
    device_type = device.type if hasattr(device, 'type') else str(device)
    print(f"Model is on device: {device_type}", color.BRIGHT_GREEN)
    
    # Create inputs and generate - more memory efficient
    model_inputs = prompt_enhancer_tokenizer([formatted_text], return_tensors="pt")
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    
    # Apply platform-specific optimizations
    try:
        if is_apple_silicon() and device_type == "mps":
            print("Using Apple Silicon MPS optimizations", color.BRIGHT_GREEN)
            with torch.inference_mode(), torch.autocast("mps"):
                outputs = prompt_enhancer_model.generate(
                    **model_inputs, 
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k
                )
        elif device_type == "cuda":
            print("Using CUDA optimizations", color.BRIGHT_GREEN)
            with torch.inference_mode(), torch.amp.autocast(device_type="cuda"):
                outputs = prompt_enhancer_model.generate(
                    **model_inputs, 
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k
                )
        else:
            print(f"Using standard inference on {device_type}", color.YELLOW)
            with torch.inference_mode():
                outputs = prompt_enhancer_model.generate(
                    **model_inputs, 
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k
                )
    except Exception as e:
        print(f"Error with optimized generation: {str(e)}, falling back to standard mode", color.YELLOW)
        with torch.inference_mode():
            outputs = prompt_enhancer_model.generate(
                **model_inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            )
    
    # Efficiently get the generated text portion only
    generated_ids = outputs[0][len(model_inputs["input_ids"][0]):]
    decoded_response = prompt_enhancer_tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Clean up memory explicitly
    del model_inputs, outputs, generated_ids
    gc.collect()
    
    print(f"Combined raw response:\n{decoded_response}\n", color.ORANGE)
    
    # Extract T5 and CLIP prompts using regex
    import re
    
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
    final_t5 = t5_prompt.replace("[", "").replace("]", "").replace("\n", "")
    final_clip = clip_prompt.replace("[", "").replace("]", "")
    

    print(f"Extracted T5 prompt:\n{final_t5}\n", color.YELLOW)
    print(f"Extracted CLIP prompt:\n{final_clip}", color.BRIGHT_YELLOW)
    
    return final_clip, final_t5

def generate_t5_prompt(
        prompt_enhancer_model, 
        prompt_enhancer_tokenizer, 
        prompt: str, 
        seed: int = None, 
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        max_new_tokens: int = _MAX_NEW_TOKENS
    ) -> str:


    # First we generate the wordy T5 prompt
    
    if seed is not None:
        torch.manual_seed(seed)
    
    # Create messages with T5-specific instructions
    messages = [
        {"role": "system", "content": get_t5_prompt_instruction()},
        {"role": "user", "content": prompt.strip() if prompt.strip() else DEFAULT_PROMPT},
    ]
    
    # Format messages
    try:
        if hasattr(prompt_enhancer_tokenizer, 'chat_template') and prompt_enhancer_tokenizer.chat_template is not None:
            print("Using tokenizer's built-in chat template for T5 prompt", color.BRIGHT_GREEN)
            formatted_text = prompt_enhancer_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            print("Using custom formatting for T5 prompt", color.YELLOW)
            formatted_text = format_chat_messages(messages, add_generation_prompt=True)
    except Exception as e:
        print(f"Error formatting messages for T5 prompt: {str(e)}", color.RED)
        formatted_text = format_chat_messages(messages, add_generation_prompt=True)
    


    # Free memory before tokenization
    gc.collect()
    
    # Get device information from model
    device = prompt_enhancer_model.device
    device_type = device.type if hasattr(device, 'type') else str(device)
    print(f"Model is on device: {device_type}", color.BRIGHT_GREEN)
    
    # Create inputs and generate - more memory efficient
    model_inputs = prompt_enhancer_tokenizer([formatted_text], return_tensors="pt")
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    
    # Apply platform-specific optimizations
    try:
        if is_apple_silicon() and device_type == "mps":
            print("Using Apple Silicon MPS optimizations", color.BRIGHT_GREEN)
            with torch.inference_mode(), torch.autocast("mps"):
                outputs = prompt_enhancer_model.generate(
                    **model_inputs, 
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k
                )
        elif device_type == "cuda":
            print("Using CUDA optimizations", color.BRIGHT_GREEN)
            with torch.inference_mode(), torch.amp.autocast(device_type="cuda"):
                outputs = prompt_enhancer_model.generate(
                    **model_inputs, 
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k
                )
        else:
            print(f"Using standard inference on {device_type}", color.YELLOW)
            with torch.inference_mode():
                outputs = prompt_enhancer_model.generate(
                    **model_inputs, 
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k
                )
    except Exception as e:
        print(f"Error with optimized generation: {str(e)}, falling back to standard mode", color.YELLOW)
        with torch.inference_mode():
            outputs = prompt_enhancer_model.generate(
                **model_inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            )
    
    # Efficiently get the generated text portion only
    generated_ids = outputs[0][len(model_inputs["input_ids"][0]):]
    decoded_response = prompt_enhancer_tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Clean up memory explicitly
    del model_inputs, outputs, generated_ids
    gc.collect()
    
    print(f"T5 raw response:\n{decoded_response}\n", color.ORANGE)
    
    # Clean up and post-process the T5 prompt
    t5_prompt = decoded_response.strip().strip('"').strip("'")
    
    # print(f"{prompt}\n{t5_prompt}", color.MAGENTA)

    # Apply subject override from original prompt if needed
    # processed_t5 = process_subject_override_smart_version(prompt, t5_prompt, prompt_enhancer_model, prompt_enhancer_tokenizer)
    
    processed_t5 = t5_prompt

    # Final cleanup - remove brackets from final output
    final_t5 = processed_t5.replace("[", "").replace("]", "").replace("\n", "")
    
    print(f"final_t5 (processed)=\n{final_t5}", color.ORANGE)

    return final_t5

def generate_clip_prompt(
        prompt_enhancer_model, 
        prompt_enhancer_tokenizer, 
        original_prompt: str,
        t5_prompt: str,
        seed: int = None, 
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        max_new_tokens: int = _MAX_NEW_TOKENS
    ) -> str:

    # Generate CLIP prompt based on original prompt + T5 prompt response
    
    if seed is not None:
        torch.manual_seed(seed)
    
    # Create messages with CLIP-specific instructions and reference to T5
    messages = [
        {"role": "system", "content": get_clip_prompt_instruction()},
        {"role": "user", "content": f"Original prompt: {original_prompt}\n\nT5 prompt: {t5_prompt}\n\nNow create a matching CLIP prompt."},
    ]
    
    # Format messages
    try:
        if hasattr(prompt_enhancer_tokenizer, 'chat_template') and prompt_enhancer_tokenizer.chat_template is not None:
            print("Using tokenizer's built-in chat template for CLIP prompt", color.BRIGHT_GREEN)
            formatted_text = prompt_enhancer_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            print("Using custom formatting for CLIP prompt", color.YELLOW)
            formatted_text = format_chat_messages(messages, add_generation_prompt=True)
    except Exception as e:
        print(f"Error formatting messages for CLIP prompt: {str(e)}", color.RED)
        formatted_text = format_chat_messages(messages, add_generation_prompt=True)
    
    # Free memory before tokenization
    gc.collect()
    
    # Get device information from model
    device = prompt_enhancer_model.device
    device_type = device.type if hasattr(device, 'type') else str(device)
    
    # Create inputs and generate - more memory efficient
    model_inputs = prompt_enhancer_tokenizer([formatted_text], return_tensors="pt")
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    
    # Apply platform-specific optimizations
    try:
        if is_apple_silicon() and device_type == "mps":
            with torch.inference_mode(), torch.autocast("mps"):
                outputs = prompt_enhancer_model.generate(
                    **model_inputs, 
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k
                )
        elif device_type == "cuda":
            with torch.inference_mode(), torch.amp.autocast(device_type="cuda"):
                outputs = prompt_enhancer_model.generate(
                    **model_inputs, 
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k
                )
        else:
            with torch.inference_mode():
                outputs = prompt_enhancer_model.generate(
                    **model_inputs, 
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k
                )
    except Exception as e:
        print(f"Error with optimized generation: {str(e)}, falling back to standard mode", color.YELLOW)
        with torch.inference_mode():
            outputs = prompt_enhancer_model.generate(
                **model_inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            )
    
    generated_ids = outputs[0][len(model_inputs["input_ids"][0]):]
    decoded_response = prompt_enhancer_tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Clean up memory explicitly
    del model_inputs, outputs, generated_ids
    gc.collect()
    
    print(f"CLIP raw response:\n{decoded_response}\n", color.ORANGE)
    
    # Clean up the CLIP prompt (simpler than T5 as it's just keywords)
    clip_prompt = decoded_response.strip().strip('"').strip("'")
    
    # # Check if bracketed subject needs to be included
    # bracketed_subject = extract_square_brackets(original_prompt)
    # print(f"bracketed_subject =\n{bracketed_subject}\n\nclip_prompt =\n{clip_prompt}", color.ORANGE)

    # if bracketed_subject and bracketed_subject.lower() not in clip_prompt.lower():
    #     print(f"Adding subject '{bracketed_subject}' to CLIP prompt", color.YELLOW)
    #     clip_prompt = f"{bracketed_subject}, {clip_prompt}"
    
    # Final cleanup - remove brackets from final output
    final_clip = clip_prompt.replace("[", "").replace("]", "")
    
    return final_clip
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
        
        print(f"IS_CHANGED hash = {hash_hex}", color.ORANGE)
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
                print("Using CUDA device for NVIDIA GPU", color.BRIGHT_GREEN)
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
            print("Generating both T5 and CLIP prompts in a single call...", color.BRIGHT_BLUE)
            
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
            # # FIRST: Generate T5 prompt
            # print("Generating T5 prompt...", color.BRIGHT_BLUE)

            # t5xxl_prompt = generate_t5_prompt(
            #     llm_model, 
            #     llm_tokenizer, 
            #     prompt, 
            #     seed, 
            #     temperature=temperature, 
            #     top_p=top_p, 
            #     top_k=top_k
            # )

            # # Force cleanup between generations
            # gc.collect()
            # if is_cuda_available():
            #     torch.cuda.empty_cache()

            # # THEN: Generate CLIP prompt with reference to T5 prompt
            # print("Generating CLIP prompt...", color.BRIGHT_BLUE)
            # clip_l_prompt = generate_clip_prompt(
            #     llm_model, 
            #     llm_tokenizer,
            #     prompt,  # Original user prompt
            #     t5xxl_prompt,  # Generated T5 prompt
            #     seed, 
            #     temperature=temperature, 
            #     top_p=top_p, 
            #     top_k=top_k
            # )
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
                print("Keeping model loaded for future use.", color.BRIGHT_BLUE)
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
                print(f"Detected Apple Silicon, applying optimizations...", color.BRIGHT_GREEN)
                torch_dtype = torch.float16  # Use float16 instead of bfloat16 for Apple Silicon
                
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
                print(f"Detected CUDA device, using NVIDIA GPU optimizations...", color.BRIGHT_GREEN)
                
                # Use native CUDA device from comfy
                cuda_device = comfy.model_management.get_torch_device()
                print(f"Using CUDA device: {cuda_device}", color.BRIGHT_GREEN)
                
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
                print(f"No GPU detected, using CPU (this will be very slow)", color.YELLOW)                
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