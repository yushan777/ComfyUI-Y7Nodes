import logging
import os
import random
import shutil
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

# ModelCache class - unchanged
class ModelCache:
    loaded_models = {}
    loaded_tokenizers = {}

# LLM model information - unchanged
LLM_MODELS = [
    ("Llama-3.2-3B-Instruct", "unsloth/Llama-3.2-3B-Instruct"),
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
llama_3_2_3b_instruct_req_files = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json"
]

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
You are an AI assistant specialized in generating comprehensive text-to-image prompts for 
the Flux image generation model. I'm going to ask you to create a prompt in two parts.
"""

# Special subject override instruction
PROMPT_SPECIAL_OVERRIDE = """
IMPORTANT: If the user's prompt contains a phrase inside square brackets (e.g., "[agg woman]"), 
then treat the content inside the brackets as the exact subject. Do not reinterpret, paraphrase, 
or alter this phrase. Use it exactly as written, as the primary subject.

Examples:
- "[agg woman] walking through a neon-lit alley" = The subject is exactly "agg woman"
- "[ohwx man] sitting in a cafe" = The subject is exactly "ohwx man"
- "[sks knight] standing in ruins" = The subject is exactly "sks knight"
"""

# T5 Prompt instructions
PROMPT_T5_INSTRUCTIONS = """
First, I need you to create a T5 Prompt:

This should be a detailed natural language description (up to 512 T5 tokens - maybe max 350 words) with:

- Subject comes first: Start with the main subject(s)
- Subject Details: Describe physical appearance, pose, action, expression, attire
- Scene Description: Overall setting, environment, background, visual style
- Time & Place: Time of day, season, architecture, objects
- Lighting: Sources, intensity, direction, color temperature, shadows
- Color Palette: Dominant and supporting colors
- Composition: Layout of elements and focal points
- Mood & Atmosphere: Emotional tone using evocative language

Use only positive descriptions — focus on what should appear in the image.
Avoid repetition and use diverse, sensory-rich vocabulary.
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

This should be a concise keyword list (up to 60 CLIP TOKENS) with:

- Prioritized, comma-separated list of essential keywords
- Include: subject(s), art style, setting, major visual features, mood, lighting, color scheme
- Include specific artistic terms if relevant (e.g., "soft focus," "cinematic lighting")
- Ensure full alignment with the T5 prompt
- Use only positive keywords — no negative terms or exclusions
"""

# CLIP example
PROMPT_CLIP_EXAMPLE = """
Example CLIP Prompt:
woman, shoulder-length black hair, luminous brown eyes, emerald satin dress, vintage clutch, dim hallway, cinematic lighting, pensive, statuesque, wooden railing, moonlight, sheer curtains, amber wall sconces, dust in air, dramatic shadows, chandelier, soft focus, contemplative mood
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

def process_subject_override(prompt, generated_text):
    """
    Process subject override if square brackets are present in the original prompt.
    Apply post-processing to ensure override is applied correctly.
    """
    bracketed_subject = extract_square_brackets(prompt)
    
    # If no bracketed subject found, return original text
    if not bracketed_subject:
        return generated_text
    
    # If brackets are found, check if the content is already in the generated text
    if bracketed_subject.lower() in generated_text.lower():
        print(f"Subject '{bracketed_subject}' already present in generated text", color.BRIGHT_GREEN)
        return generated_text
    
    # If not present, prepend the subject to the generated text
    print(f"Applying subject override: '{bracketed_subject}'", color.YELLOW)
    return f"{bracketed_subject} {generated_text}"

# ==================================================================================
# NEW SEPARATE GENERATION FUNCTIONS
# ==================================================================================

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
    """Generate T5 prompt separately."""
    
    if seed is not None:
        random.seed(seed)
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
    
    # Create inputs and generate
    model_inputs = prompt_enhancer_tokenizer([formatted_text], return_tensors="pt")
    model_inputs = model_inputs.to(prompt_enhancer_model.device)
    
    with torch.inference_mode():
        outputs = prompt_enhancer_model.generate(
            **model_inputs, 
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )
        
        generated_ids = outputs[0][len(model_inputs.input_ids[0]):]
        decoded_response = prompt_enhancer_tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    print(f"T5 raw response:\n{decoded_response}\n", color.ORANGE)
    
    # Clean up and post-process the T5 prompt
    t5_prompt = decoded_response.strip().strip('"').strip("'")
    
    # Apply subject override from original prompt if needed
    processed_t5 = process_subject_override(prompt, t5_prompt)
    
    # Final cleanup - remove brackets from final output
    final_t5 = processed_t5.replace("[", "").replace("]", "")
    
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
    """Generate CLIP prompt based on original prompt and T5 prompt."""
    
    if seed is not None:
        random.seed(seed)
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
    
    # Create inputs and generate
    model_inputs = prompt_enhancer_tokenizer([formatted_text], return_tensors="pt")
    model_inputs = model_inputs.to(prompt_enhancer_model.device)
    
    with torch.inference_mode():
        outputs = prompt_enhancer_model.generate(
            **model_inputs, 
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )
        
        generated_ids = outputs[0][len(model_inputs.input_ids[0]):]
        decoded_response = prompt_enhancer_tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    print(f"CLIP raw response:\n{decoded_response}\n", color.ORANGE)
    
    # Clean up the CLIP prompt (simpler than T5 as it's just keywords)
    clip_prompt = decoded_response.strip().strip('"').strip("'")
    
    # Check if bracketed subject needs to be included
    bracketed_subject = extract_square_brackets(original_prompt)
    if bracketed_subject and bracketed_subject.lower() not in clip_prompt.lower():
        print(f"Adding subject '{bracketed_subject}' to CLIP prompt", color.YELLOW)
        clip_prompt = f"{bracketed_subject}, {clip_prompt}"
    
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
            load_device = comfy.model_management.get_torch_device()
            offload_device = comfy.model_management.unet_offload_device()
            
            # Load/download the model 
            llm_model, llm_tokenizer = self.down_load_llm_model(llm_display_name, load_device)
            
            # Calculate model size for memory management and handle device placement
            model_size = get_model_size(llm_model) + 1073741824  # Add 1GB extra as buffer
            
            # Free memory before loading model to GPU
            comfy.model_management.free_memory(model_size, comfy.model_management.get_torch_device(),)
            
            # Ensure model is on the correct device
            llm_model.to(load_device)
            print(f"Model device: {next(llm_model.parameters()).device}", color.BRIGHT_GREEN)
            
            # FIRST: Generate T5 prompt separately
            print("Generating T5 prompt...", color.BRIGHT_BLUE)
            t5xxl_prompt = generate_t5_prompt(
                llm_model, 
                llm_tokenizer, 
                prompt, 
                seed, 
                temperature=temperature, 
                top_p=top_p, 
                top_k=top_k
            )
            
            # THEN: Generate CLIP prompt with reference to T5 prompt
            print("Generating CLIP prompt...", color.BRIGHT_BLUE)
            clip_l_prompt = generate_clip_prompt(
                llm_model, 
                llm_tokenizer,
                prompt,  # Original user prompt
                t5xxl_prompt,  # Generated T5 prompt
                seed, 
                temperature=temperature, 
                top_p=top_p, 
                top_k=top_k
            )
            
            # Memory management based on keep_model_loaded parameter
            if not keep_model_loaded:
                print("Offloading model from VRAM...\n", color.BRIGHT_BLUE)
                llm_model.to(offload_device)
                
                if llm_display_name in ModelCache.loaded_models:
                    del ModelCache.loaded_models[llm_display_name]
                    del ModelCache.loaded_tokenizers[llm_display_name]
                
                comfy.model_management.soft_empty_cache()
            else:
                print("Keeping model loaded in VRAM for future use.", color.BRIGHT_BLUE)
                llm_model.to(load_device)
                ModelCache.loaded_models[llm_display_name] = llm_model
                ModelCache.loaded_tokenizers[llm_display_name] = llm_tokenizer
            
            return (clip_l_prompt, t5xxl_prompt,)
            
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
                if "Llama-3.2-3B-Instruct" in repo_path:
                    print(f"ℹ️ Downloading {repo_path} (≈6.5GB)", color.BRIGHT_BLUE)
                    allow_patterns = llama_3_2_3b_instruct_req_files
                elif "OpenHermes-2.5-Mistral-7B" in repo_path:
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

            if model_display_name == "Llama-3.2-3B-Instruct":                                       
                required_files = llama_3_2_3b_instruct_req_files
            elif model_display_name == "OpenHermes-2.5-Mistral-7B":
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
            
            # Load model
            llm_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
            )
            
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