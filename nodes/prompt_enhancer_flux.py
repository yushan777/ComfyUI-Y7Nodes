import logging
import os
import random
import shutil
import hashlib
from typing import List, Optional, Tuple, Union, Dict, Any
from ..utils.colored_print import color, style
import comfy.model_management
import comfy.model_patcher
import folder_paths
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from huggingface_hub import snapshot_download
import numpy as np

 
# Function Call Sequence
# Y7Nodes_PromptEnhancerFlux.enhance()
#   ├── model_path_download_if_needed() - Checks if model exists, downloads if needed
#   │
#   ├── down_load_llm_model() - Loads model from disk
#   │   └── model_path_download_if_needed() - Called again from within
#   │
#   ├── Calculate model size for memory management
#   │
#   ├── comfy.model_management.free_memory() - Ensures GPU has space
#   │
#   ├── Model moved to the correct device (GPU)
#   │
#   ├── generate_flux_t5_clip_prompts() - Generates enhanced prompts
#   │   ├── Prompt preprocessing and tokenization
#   │   └── _generate_and_decode_flux_prompts()
#   │       ├── model.generate() - Generates text with the LLM
#   │       ├── batch_decode() - Decodes output tokens to text
#   │       └── Parsing and cleanup of the response
#   │
#   └── return (t5_prompt, clip_prompt)

# ==================================================================================
# MODEL PERSISTENCE
# ==================================================================================
# ModelCache class provides a persistent storage mechanism for loaded models and tokenizers.
# Unlike other (similar) custom nodes which use separate nodes for model loading and 
# inference, we use a class-level cache to maintain refs to loaded models between 
# calls. This prevents Python's garbage collector from freeing models when 
# keep_model_loaded=True.
# ==================================================================================
class ModelCache:
    # These dicts persist for the entire lifetime of the Python process
    # Models stored here will not be garbage collected until explicitly removed
    loaded_models = {}
    loaded_tokenizers = {}

# logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# LLM_REPO_NAME = "unsloth/Llama-3.2-3B-Instruct"

# LLM MODELS : 
# Tuples of : model_name, model repo path, quantized version (FOR GGUF models only)
LLM_MODELS = [
    ("Llama-3.2-3B-Instruct", "unsloth/Llama-3.2-3B-Instruct"),
    ("OpenHermes-2.5-Mistral-7B", "teknium/OpenHermes-2.5-Mistral-7B"),
    ("Hermes-Trismegistus-Mistral-7B", "teknium/Hermes-Trismegistus-Mistral-7B"),
    # Add more models here as needed
]

# Extract display names for the dropdown
LLM_DISPLAY_NAMES = [model[0] for model in LLM_MODELS]

# Helper function to get repository information from display name
def get_repo_info(display_name):
    # Returns the repository information from a given model's display name.    
    for model_info in LLM_MODELS:
        if model_info[0] == display_name:
            return model_info[1]
    return None


# required files for unsloth/llama_3_2_3b_instruct
# single file available so we can ignore the sharded version of the model
llama_3_2_3b_instruct_req_files = [
                                    "config.json",
                                    "generation_config.json",
                                    "model.safetensors",
                                    "model.safetensors.index.json",
                                    "special_tokens_map.json",
                                    "tokenizer.json",
                                    "tokenizer_config.json"]

# required files for teknium/OpenHermes-2.5-Mistral-7B
# only sharded version available
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
                                "transformers_inference.py"]

# required files for teknium/Hermes-Trismegistus-Mistral-7B
# only sharded bin version available
hermes_trismegistus_mistral_7b_req_files = [
                    "added_tokens.json",
                    "config.json",
                    "generation_config.json",
                    "pytorch_model-00001-of-00002.bin",
                    "pytorch_model-00002-of-00002.bin",
                    "pytorch_model.bin.index.json",
                    "special_tokens_map.json",
                    "tokenizer.model",
                    "tokenizer_config.json"]

# _MAX_NEW_TOKENS=1024
_MAX_NEW_TOKENS=2048


MODELS_PATH_KEY = "LLM"
DEFAULT_PROMPT = ""

# Base introduction and purpose
PROMPT_BASE = """
You are an AI assistant specialized in generating comprehensive text-to-image prompts for 
the Flux image generation model. Each output must include two complementary prompt types that work together:

---
"""

# Special override rules
PROMPT_SPECIAL_OVERRIDES = """
**Special Subject Override Rule**:

If the user's prompt contains a phrase inside square brackets (e.g., "[agg woman]"), then the content inside the brackets as 
the **explicit and literal subject** of the image. Do not reinterpret, paraphrase, or alter this phrase. 
Use it exactly as written, as the primary subject in both the T5 and CLIP prompts.
- preserve the subject phrase exactly inside the square brackets, including the case of the phrase!

Examples:
- "[agg woman] walking through a neon-lit alley" = "agg woman walking through a neon-lit alley"
- "[agg, a young woman] standing on a busy sidewalk" = "agg, a young woman standing on a busy sidewalk"
- "[ohwx man] sitting in a cafe" = "ohwx man sitting in a quiet cafe"
- "[ohwx, an old man] standing next to an old car" = "ohwx, an old man standing next to an old car"
- "[sks knight] standing in ruins" = "sks knight standing in ruins"
- "[sks dog] playing in a park" = "sks dog playing in a park"
"""

# T5 Prompt instructions section
PROMPT_T5_INSTRUCTIONS = """
1. **T5 Prompt** (Detailed natural language description, simulate approximately up to 512 T5 tokens (roughly up to 400 words)):

Structure your description in this order:

- **Subject comes first**: Clearly state the main subject(s) at the beginning - obey the special subject override rule explained previously (even it it grammar rules).
- **Subject Details**: Describe the subject(s) in vivid detail, including physical appearance, pose, action, expression, attire, and interactions.
- **Scene Description**: Describe the overall setting including environment, background, location type (e.g., interior, exterior), and visual style.
- **Time & Place**: Indicate time of day, season, architecture, and relevant objects or decor.
- **Lighting**: Describe lighting sources, intensity, direction, color temperature, shadows, and effects.
- **Color Palette**: Specify dominant and supporting colors, including visual harmony or contrasts.
- **Composition**: Detail the layout — foreground, middle ground, background, and focal points.
- **Mood & Atmosphere**: Convey emotional tone using evocative, poetic, or cinematic language.
- **Use only positive descriptions** — focus solely on what should appear in the image.
- **Avoid repetition and filler words. Use diverse, sensory-rich vocabulary.**
"""

# T5 example section
PROMPT_T5_EXAMPLE = """
**Example T5 Prompt**:
A woman with shoulder-length black hair and luminous brown eyes stands alone in a dimly lit interior hallway. She wears a sleek, 
emerald green satin dress that catches the light with a subtle shimmer. Her posture is still, almost statuesque, as she gazes 
slightly off-camera with a pensive expression. One hand rests gently on a weathered wooden railing, while the other clutches a 
small vintage clutch. The hallway is narrow and elegant, lined with tall windows draped in sheer curtains that allow soft shafts of moonlight to filter 
in. Dust particles drift through the air, caught in the light. Ornate wall sconces cast warm amber glows along the wallpapered walls, 
creating deep shadows that frame the woman in dramatic contrast. Muted jewel tones dominate the color palette: deep greens, soft golds, and shadowy blues. 
In the background, a blurred chandelier hangs overhead, adding a subtle sparkle. The composition places the woman slightly off-center, with the lines of the corridor drawing 
focus toward her. The overall mood is contemplative and cinematic, evoking the quiet tension of a moment suspended in time.

**Example T5 Prompt (with subject override rule)**:
agg woman with shoulder-length black hair and luminous brown eyes stands alone in a dimly lit interior hallway. She wears a sleek, 
emerald green satin dress that catches the light with a subtle shimmer. Her posture is still, almost statuesque, as she gazes 
slightly off-camera with a pensive expression. One hand rests gently on a weathered wooden railing, while the other clutches a 
small vintage clutch. The hallway is narrow and elegant, lined with tall windows draped in sheer curtains that allow soft shafts of moonlight to filter 
in. Dust particles drift through the air, caught in the light. Ornate wall sconces cast warm amber glows along the wallpapered walls, 
creating deep shadows that frame the woman in dramatic contrast. Muted jewel tones dominate the color palette: deep greens, soft golds, and shadowy blues. 
In the background, a blurred chandelier hangs overhead, adding a subtle sparkle. The composition places the woman slightly off-center, with the lines of the corridor drawing 
focus toward her. The overall mood is contemplative and cinematic, evoking the quiet tension of a moment suspended in time.
---
"""

# CLIP prompt instructions section
PROMPT_CLIP_INSTRUCTIONS = """
2. **CLIP Prompt** (Concise keyword list; simulate approximately up to 75 CLIP tokens (roughly up to 30-40 words)):

- Provide a prioritized, comma-separated list of essential keywords.
- Include: subject(s), art style (if any), setting, major visual features, mood, lighting, and color scheme.
- Include specific artistic or stylistic terms if relevant (e.g., "soft focus," "cinematic lighting," "Baroque detail").
- Ensure full alignment with the T5 prompt.
- Use only **positive keywords** — no negative terms or exclusions.
- Avoid overgeneralizations or generic terms like "high quality" or "detailed" unless critical to the style.
"""

# CLIP example section
PROMPT_CLIP_EXAMPLE = """
**Example CLIP Prompt**:  
woman, shoulder-length black hair, luminous brown eyes, emerald satin dress, vintage clutch, dim hallway, cinematic lighting, pensive, 
statuesque, wooden railing, moonlight, sheer curtains, amber wall sconces, dust in air, dramatic shadows, chandelier, soft focus, 
contemplative mood, baroque detail

**Example CLIP Prompt (with subject override rule)**:  
agg woman, shoulder-length black hair, luminous brown eyes, emerald satin dress, vintage clutch, dim hallway, cinematic lighting, pensive, 
statuesque, wooden railing, moonlight, sheer curtains, amber wall sconces, dust in air, dramatic shadows, chandelier, soft focus, 
contemplative mood, baroque detail
---
"""



# Combine all sections when needed in the code
def get_prompt_instruction():
    return (
        PROMPT_BASE + 
        PROMPT_T5_INSTRUCTIONS + 
        PROMPT_T5_EXAMPLE + 
        PROMPT_CLIP_INSTRUCTIONS + 
        PROMPT_CLIP_EXAMPLE + 
        PROMPT_SPECIAL_OVERRIDES
    )

# **Output Format**:

# T5 Prompt: [Your detailed natural language description]  
# CLIP Prompt: [Your concise keyword list]
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
                    LLM_DISPLAY_NAMES,  # Use display names for the dropdown
                    {
                        "default": LLM_DISPLAY_NAMES[0],  # Default to first model
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
            "hidden":{

            }
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("clip_l_prompt", "t5xxl_prompt",)
    FUNCTION = "enhance"
    CATEGORY = "Y7Nodes/Prompt"
    OUTPUT_NODE = False
    

    # =================================================================================
    # the IS_CHANGED method used to determine whether the node's outputs need to be 
    # recalculated when inputs change. Return a unique value (usually a string or hash) 
    # that represents the current state of the node. When this value changes compared 
    # to the previous run, ComfyUI knows it needs to re-execute the node.
    @classmethod
    def IS_CHANGED(cls, **kwargs):
                
        # Extract parameters that affect the output
        prompt = kwargs.get("prompt", "")
        llm_display_name = kwargs.get("llm_name", "")                
        seed = kwargs.get("seed", 0)
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 0.9)
        top_k = kwargs.get("top_k", 40)
        
        # Create a string with all parameters
        input_string = f"{prompt}_{llm_display_name}_{seed}_{temperature}_{top_p}_{top_k}"
        
        # Generate a hash of the input string (cos prompts can be long)
        hash_object = hashlib.md5(input_string.encode())
        hash_hex = hash_object.hexdigest()
        
        print(f"IS_CHANGED hash = {hash_hex}", color.ORANGE)
        return hash_hex
        
   # ==================================================================================
    # def enhance(self, prompt, llm_name, seed=0, temperature=0.7, top_p=0.9, top_k=40, keep_model_loaded=False, **kwargs):
    def enhance(self, **kwargs):

        prompt = kwargs.get("prompt")
        llm_display_name = kwargs.get("llm_name")  # This is now the display name
        
        # Get the repository path
        # repo_path = get_repo_info(llm_display_name)
        
        seed = kwargs.get("seed")
        temperature = kwargs.get("temperature")
        top_p = kwargs.get("top_p")
        top_k = kwargs.get("top_k")
        keep_model_loaded = kwargs.get("keep_model_loaded")

        # Default prompt if empty
        if not prompt.strip():
            prompt = DEFAULT_PROMPT
            t5xxl_prompt = 'Please provide a prompt, no matter how basic.  If you wish to use a token or trigger words enclose them in square brackets.\nExamples:\n\n"A man sitting in a cafe".\n"[ohwx woman] standing in the middle of a busy street"'
            clip_l_prompt = ""
            return (t5xxl_prompt, clip_l_prompt,)
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
            
            # Generate prompts
            t5_clip_prompts = generate_flux_t5_clip_prompts(
                llm_model, 
                llm_tokenizer, 
                prompt, 
                seed, 
                temperature=temperature, 
                top_p=top_p, 
                top_k=top_k
            )

            if not t5_clip_prompts or not t5_clip_prompts[0] or not t5_clip_prompts[0][0]:
                print("WARNING: Empty or invalid result from generate_flux_t5_clip_prompts", color.RED)
                # Return a fallback response
                return (
                    f"Error: Model failed to generate proper output.",
                    f"Error: Please try again with different parameters or a different model."
                )

            # Get the first pair of prompts (t5, clip)
            t5xxl_prompt, clip_l_prompt = t5_clip_prompts[0]
            
            # ==================================================================================
            # MODEL PERSISTENCE - MEMORY MANAGEMENT
            # ==================================================================================
            # This section determines whether to keep or unload the model based on the 
            # keep_model_loaded parameter. Two critical functions are performed here:
            #
            # 1. Device Management: Moving the model between GPU and CPU/offload device
            # 2. Reference Management: Maintaining or removing references in the ModelCache
            #
            # Even if a model stays on the GPU, without a persistent reference (like in the
            # ModelCache), Python's garbage collector would free it once this function returns.
            # ==================================================================================
            if not keep_model_loaded:
                print("Offloading model from VRAM...\n", color.BRIGHT_BLUE)
                
                # DEVICE MANAGEMENT: Move the model to the offload device (usually CPU)
                llm_model.to(offload_device)
                
                # REFERENCE MANAGEMENT: Remove from cache to allow garbage collection
                # This is crucial - simply moving to CPU isn't enough to free VRAM
                # We must remove all references so the garbage collector can free the memory
                if llm_display_name in ModelCache.loaded_models:
                    del ModelCache.loaded_models[llm_display_name]
                    del ModelCache.loaded_tokenizers[llm_display_name]
                
                # Trigger memory cleanup
                comfy.model_management.soft_empty_cache()
            else:
                print("Keeping model loaded in VRAM for future use.", color.BRIGHT_BLUE)
                
                # DEVICE MANAGEMENT: Ensure the model stays on the GPU
                llm_model.to(load_device)
                
                # REFERENCE MANAGEMENT: Ensure the model stays in the cache
                # This maintains a persistent reference to prevent garbage collection
                # other custom nodes keeps models in separate nodes,
                # our approach uses this dictionary to maintain references
                ModelCache.loaded_models[llm_display_name] = llm_model
                ModelCache.loaded_tokenizers[llm_display_name] = llm_tokenizer
            
            return (clip_l_prompt, t5xxl_prompt,)
            
        except Exception as e:
            # Return a generic error message
            print(f"❌ Error: {str(e)}", color.BRIGHT_RED)
            return (
                f"Error: {str(e)}",
                f"Error: Please check the model output format"
            )
                
    # ==================================================================================
    def model_path_download_if_needed(self, model_display_name):

        # Get repo path
        repo_path = get_repo_info(model_display_name)
        
        # path to the LLM model i.e /Path-To/ComfyUI/models/LLM
        llm_model_directory = os.path.join(folder_paths.models_dir, MODELS_PATH_KEY)
        
        # make the dir if not exist, don't raise error otherwise
        os.makedirs(llm_model_directory, exist_ok=True)

        # split and get last element of repo_path
        model_name = repo_path.rsplit("/", 1)[-1]
        full_model_path = os.path.join(llm_model_directory, model_name)

        # Check if directory and files exists, if not....
        if not os.path.exists(full_model_path):
            # attempt to download the model
            print(f"⬇️ Downloading model {repo_path} from HF to models/LLM.\nThis may take a while. (Download might appear stuck but it really is downloading)", color.YELLOW)
                        
            try:                

                # Regular model download logic
                if "Llama-3.2-3B-Instruct" in repo_path:
                    print(f"ℹ️ Downloading {repo_path} (≈6.5GB)", color.BRIGHT_BLUE)
                    allow_patterns = llama_3_2_3b_instruct_req_files

                elif "OpenHermes-2.5-Mistral-7B" in repo_path:
                    print(f"ℹ️ Downloading {repo_path} (≈14.5GB)", color.BRIGHT_BLUE)
                    allow_patterns = openhermes_2_5_mistral_7b_req_files

                elif "Hermes-Trismegistus-Mistral-7B" in repo_path:
                    print(f"ℹ️ Downloading {repo_path} (≈14.5GB)", color.BRIGHT_BLUE)
                    allow_patterns = openhermes_2_5_mistral_7b_req_files

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
            # directory does exist, check if all necessary files exist (per model)

            # track missing files 
            missing_files = []
            required_files = []

            if model_display_name == "Llama-3.2-3B-Instruct":                                       
                required_files = llama_3_2_3b_instruct_req_files
                                
            elif model_display_name == "OpenHermes-2.5-Mistral-7B":
                required_files = openhermes_2_5_mistral_7b_req_files
                
            elif model_display_name == "Hermes-Trismegistus-Mistral-7B":
                required_files = hermes_trismegistus_mistral_7b_req_files

            # find any missing files
            for file in required_files:
                if not os.path.exists(os.path.join(full_model_path, file)):
                    missing_files.append(file)
            
            # if found.. attempt to download. 
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

    # ==================================================================================
    def down_load_llm_model(self, model_display_name, load_device):
        # ==================================================================================
        # MODEL PERSISTENCE OPTIMIZATION
        # ==================================================================================
        # Check if the model is already in the cache before loading it again
        # This is critical for the keep_model_loaded functionality, as it:
        # 1. Avoids redundant loading of the same model
        # 2. Provides fast access to previously loaded models
        # 3. Maintains model state between function calls
        # ==================================================================================
        # Get repo path 
        repo_path = get_repo_info(model_display_name)
        
        if model_display_name in ModelCache.loaded_models and model_display_name in ModelCache.loaded_tokenizers:
            print(f"Using cached model {model_display_name} from previous run", color.BRIGHT_GREEN)
            # immediately return the model and tokenizer
            return ModelCache.loaded_models[model_display_name], ModelCache.loaded_tokenizers[model_display_name]
            
        # otherwise.. attempt to download the model
        model_path = self.model_path_download_if_needed(model_display_name)
                
        try:
            # Try to load the model
            
            print(f"Loading model {model_display_name}", color.BRIGHT_BLUE)
        
            # Regular Transformers model loading
            llm_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
            )
            
            llm_tokenizer = AutoTokenizer.from_pretrained(
                model_path,
            )
            
            # ==================================================================================
            # MODEL PERSISTENCE - CACHING MECHANISM
            # ==================================================================================
            # Store the model and tokenizer in the class-level cache to maintain references
            # This prevents Python's garbage collector from freeing the model memory
            # when the local variables go out of scope after the function returns.
            # These cached models will persist until explicitly removed.
            # ==================================================================================
            ModelCache.loaded_models[model_display_name] = llm_model
            ModelCache.loaded_tokenizers[model_display_name] = llm_tokenizer
            
            return llm_model, llm_tokenizer
            
        except (FileNotFoundError, ValueError) as e:
            # Handle the case of a partially downloaded or corrupted model
            print(f"❌ Error: Model files are incomplete or corrupted: {str(e)}", color.RED)
            print(f"🔄 Please manually delete the directory : {model_path}", color.YELLOW)
            print(f"🔄 Then re-launch the workflow to attempt downloading again.", color.YELLOW)
            
            
            print(f"🔄 Alternatively you can manually download the model from: \nhttps://huggingface.co/{repo_path}", color.YELLOW)                
            print(f"   and places the files to: {model_path}", color.YELLOW)
            
            # Re-raise the exception to stop execution
            raise RuntimeError(f"Model at {model_path} is incomplete or corrupted. Please delete this directory and try again.")
                        

 

# Helper function to calculate model size
def get_model_size(model):
    """Calculate the memory size of a model based on parameters and buffers.
    
    Args:
        model: PyTorch model
        
    Returns:
        Total memory size in bytes
    """

    # For PyTorch models, calculate based on parameters and buffers
    total_size = sum(p.numel() * p.element_size() for p in model.parameters())
    total_size += sum(b.numel() * b.element_size() for b in model.buffers())
    return total_size

# ==================================================================================
# Custom function to format chat messages without relying on apply_chat_template
def format_chat_messages(messages, add_generation_prompt=True):
    """
    Format chat messages into a single string without using apply_chat_template.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        add_generation_prompt: Whether to add a generation prompt at the end
        
    Returns:
        Formatted string with all messages
    """
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

def generate_flux_t5_clip_prompts(
                    prompt_enhancer_model, 
                    prompt_enhancer_tokenizer, 
                    prompt: Union[str, List[str]], 
                    seed: int = None, 
                    temperature: float = 0.7,
                    top_p: float = 0.9,
                    top_k: int = 40,
                    max_new_tokens: int = _MAX_NEW_TOKENS
                ) -> List[Tuple[str, str]]:
    
    """
    Generate T5 and CLIP prompts for Flux image generation model.
    
    Args:
        prompt_enhancer_model: The language model to use for prompt enhancement
        prompt_enhancer_tokenizer: The tokenizer for the language model
        prompt: Input prompt(s) to enhance
        seed: Random seed for reproducibility
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Controls randomness of outputs
        top_p: Nucleus sampling parameter
        top_k: Limits token selection to k most likely tokens
        
    Returns:
        List of tuples containing (t5_prompt, clip_prompt) pairs
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
    
    # Process single prompt or list of prompts
    if isinstance(prompt, str):
        prompts = [prompt.strip() if prompt.strip() else DEFAULT_PROMPT]
    else:
        prompts = [p.strip() if p.strip() else DEFAULT_PROMPT for p in prompt]

    messages = [
        [
            {"role": "system", "content": get_prompt_instruction()},
            {"role": "user", "content": f"{p}"},
        ]
        for p in prompts
    ]

    # Check if the tokenizer has a chat template
    has_chat_template = hasattr(prompt_enhancer_tokenizer, 'chat_template') and prompt_enhancer_tokenizer.chat_template is not None
    
    try:
        if has_chat_template:
            print("Using tokenizer's built-in chat template", color.BRIGHT_GREEN)
            texts = [
                prompt_enhancer_tokenizer.apply_chat_template(
                    m, tokenize=False, add_generation_prompt=True
                )
                for m in messages
            ]
        else:
            print("Tokenizer has no chat template, using custom formatting", color.YELLOW)
            texts = [format_chat_messages(m, add_generation_prompt=True) for m in messages]
    except Exception as e:
        print(f"Error applying chat template: {str(e)}", color.RED)
        print("Falling back to custom message formatting", color.YELLOW)
        texts = [format_chat_messages(m, add_generation_prompt=True) for m in messages]
    
    # Handle device placement for PyTorch models
    model_inputs = prompt_enhancer_tokenizer(texts, return_tensors="pt")
    model_inputs = model_inputs.to(prompt_enhancer_model.device)

    return _generate_and_decode_flux_prompts(
        prompt_enhancer_model, 
        prompt_enhancer_tokenizer, 
        model_inputs, 
        seed,
        temperature,
        top_p,
        top_k,
        max_new_tokens
    )

# ==================================================================================
def _generate_and_decode_flux_prompts(
    prompt_enhancer_model, 
    prompt_enhancer_tokenizer, 
    model_inputs, 
    seed,
    temperature,
    top_p,
    top_k,
    max_new_tokens=_MAX_NEW_TOKENS
) -> List[Tuple[str, str]]:
    
    """
    Generate and decode the T5 and CLIP prompts from the model output.
    
    Args:
        prompt_enhancer_model: The language model
        prompt_enhancer_tokenizer: The tokenizer
        model_inputs: Inputs to the model
        max_new_tokens: Maximum number of new tokens to generate
        seed: Random seed for reproducibility
        temperature: Controls randomness of outputs
        top_p: Nucleus sampling parameter
        top_k: Limits token selection to k most likely tokens
        
    Returns: List of tuples containing (t5_prompt, clip_prompt) pairs
    """

    # For transformers models, use the standard API
    with torch.inference_mode():
        outputs = prompt_enhancer_model.generate(
                                        **model_inputs, 
                                        max_new_tokens=max_new_tokens,
                                        do_sample=True,
                                        temperature=temperature,
                                        top_p=top_p,
                                        top_k=top_k)
        
        # seed is already set in generate_flux_t5_clip_prompts() function
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, outputs)
        ]
        decoded_responses = prompt_enhancer_tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

    result_pairs = []

    for response in decoded_responses:
        t5_prompt = ""
        clip_prompt = ""

        print(f"RAW response = \n{response}\n\n", color.ORANGE)
        
        # Enhanced parsing logic with better error handling
        try:
            if "T5 Prompt" in response and "CLIP Prompt" in response:
                # Extract everything after "T5 Prompt"
                t5_raw = response.split("T5 Prompt", 1)[1]
                # Now split at "CLIP Prompt" to isolate just the T5 section
                t5_section = t5_raw.split("CLIP Prompt", 1)[0]

                # Cleanup: remove colons, quotes, asterisks, and trim whitespace
                t5_prompt = (
                    t5_section.replace(":", "") # remove colons
                    .replace("*", "") # remove asterisks
                    .replace("[", "") # remove [
                    .replace("]", "") # remove ]
                    .replace("\n", "") # remove \n
                    .strip() # removes leading and trailing whitespace
                    .strip('"') # removes leading or trailing double quotes             
                    
                )

                # Now extract everything after "CLIP Prompt"
                clip_section = response.split("CLIP Prompt", 1)[1]
                clip_prompt = (
                    clip_section.replace(":", "")
                    .replace("*", "")
                    .replace("[", "") # remove [
                    .replace("]", "") # remove ]   
                    .replace("\n", "") # remove \n             
                    .strip()
                    .strip('"')
                )
            else:
                # Llama-3.2-3B-Instruct is quite censored and will return strings that start with:
                if response.startswith("I can't"):
                    print("Refusal detected.")
                    t5_prompt = response
                    clip_prompt = response  

                    

        except Exception as e:
            print(f"Error during prompt parsing: {str(e)}", color.RED)
            t5_prompt = f"Error during prompt parsing: {str(e)}"
            clip_prompt = "Unable to extract CLIP prompt from model output"
        
        # Check if prompts are empty after parsing
            t5_prompt = "Error: Unable to extract T5 prompt from model response"
        
        if not clip_prompt.strip():
            clip_prompt = "Error: Unable to extract CLIP prompt from model response"
        
        result_pairs.append((t5_prompt, clip_prompt))
    
    return result_pairs
