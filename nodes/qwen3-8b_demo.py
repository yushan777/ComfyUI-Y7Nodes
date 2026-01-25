import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys

# Use relative path from script location
script_dir = os.path.dirname(os.path.abspath(__file__))
model_name = os.path.join(script_dir, "model", "Qwen3-8B")

# Validate that model path exists
if not os.path.exists(model_name):
    print(f"\033[91mERROR: Model path not found: {model_name}\033[0m", file=sys.stderr)
    print(f"\033[91mPlease ensure the model files are downloaded to the correct location.\033[0m", file=sys.stderr)
    sys.exit(1)

torch.cuda.empty_cache()

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

# torch_dtype options:
# - "auto (default)": Uses model's native dtype (recommended - respects original config)
# - torch.bfloat16: 16-bit, good for modern GPUs, balanced speed/stability
# - torch.float16: 16-bit, faster but less stable
# - torch.float32: 32-bit, slowest, 2x memory, no quality benefit over native dtype
# Mismatching dtype (e.g. float32 on bf16 model) auto-converts but wastes memory/speed

# device_map options:
# - "auto": Automatically distributes model across available GPU(s)/CPU (recommended)
# - "cuda" or "cuda:0": Places entire model on specific GPU
# - "cpu": Places entire model on CPU
# - "balanced": Evenly distributes across multiple GPUs
# - Custom dict: Manual control over layer placement per device

# low_cpu_mem_usage options:
# - True: Loads model layer-by-layer to reduce peak RAM usage (recommended for large models)
# - False: Loads entire model at once (faster loading but uses 2x RAM temporarily)

# trust_remote_code options:
# - True: Allows executing custom Python code from model files (required for some models like Qwen)
# - False: Only uses standard HuggingFace code (safer but incompatible with custom model architectures)
# WARNING: Only set to True for models from trusted sources


PROMPT = "embellish this basic prompt: 'a young girl is sitting on a park bench on a cloudy day'"

messages = [
    {"role": "user", "content": PROMPT}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=4096,
    use_cache=True,
    temperature=0.7,           # Controls randomness (0.1=focused, 2.0=creative)
    top_p=0.9,                 # Nucleus sampling threshold
    top_k=50,                  # Limits vocabulary to top K tokens
    do_sample=True,            # Enable sampling for more natural output
    repetition_penalty=1.1,    # Penalizes repeated tokens
    no_repeat_ngram_size=3     # Prevents repeating 3-grams
)

# max_new_tokens: Maximum number of new tokens to generate (not including input prompt)
# - Higher values allow longer responses but take more time and memory
# - 32768 is very high (allows ~25K words) - typical values: 512-4096
# - Model will stop early if it generates an end-of-sequence token

# temperature: Controls randomness in token selection (higher = more creative/random)
# - 0.0-0.3: Very focused, deterministic, factual (good for accuracy-critical tasks)
# - 0.5-0.7: Balanced creativity and coherence (good default)
# - 0.8-1.2: More creative and diverse outputs
# - 1.5-2.0: Very creative/random (can become incoherent)
# Your 0.7 is a good balanced setting

# top_p (nucleus sampling): Considers only tokens whose cumulative probability >= p
# - 0.9: Only samples from top 90% probability mass (balanced, recommended)
# - 0.95: More diverse, includes less likely tokens
# - 0.5: Very focused, only highest probability tokens
# - 1.0: Considers all tokens (no filtering)
# Works together with temperature to control randomness

# top_k: Only samples from the K most likely tokens at each step
# - 50: Good balanced setting (your current value)
# - 20-40: More focused, predictable outputs
# - 80-100: More diverse outputs
# - 0 or None: No filtering (considers all tokens)
# Filters before top_p is applied

# do_sample: Enables probabilistic sampling instead of greedy decoding
# - True: Samples from probability distribution (uses temperature, top_p, top_k)
# - False: Always picks highest probability token (deterministic, ignores temp/top_p/top_k)
# Set to True for creative/natural outputs, False for deterministic/factual tasks

# repetition_penalty: Reduces probability of tokens that have already been generated
# - 1.0: No penalty (tokens can repeat freely)
# - 1.1-1.2: Mild penalty, reduces repetition while staying natural (your 1.1 is good)
# - 1.5+: Strong penalty, may affect coherence
# - <1.0: Encourages repetition (rarely used)
# Helps prevent the model from repeating the same words/phrases

# no_repeat_ngram_size: Prevents repeating sequences of N consecutive tokens
# - 0: No n-gram blocking (allows any repetition)
# - 3: Prevents repeating any 3-word phrase (your setting - good balance)
# - 2: Stricter, prevents repeating any 2-word phrase
# - 4+: More lenient, only blocks longer repeated phrases
# Complements repetition_penalty by preventing exact phrase repetition


output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

try:
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)


