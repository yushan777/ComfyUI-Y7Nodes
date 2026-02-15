# ComfyUI-Y7Nodes

A collection of utility / quality-of-life nodes for ComfyUI - Probably only useful to me.  There's really nothing new here. Some nodes are modifications of of existing custom nodes with additional features that suit my particular needs.

## Installation (ComfyUI Manager)

 #### This is the better way to install: 
 - Open `ComfyUI Manager` 
   - → `Custom Nodes Manager` 
   - → Search for `Y7` or `Y7Nodes`. 
   - Install. 
   - Restart Restart ComfyUI

------

## Installation (Manual)

1. Clone this repository into your ComfyUI custom_nodes directory:
   ```bash
   cd /path/to/ComfyUI/custom_nodes
   git clone https://github.com/yushan777/ComfyUI-Y7Nodes
   
   ```

2. Install Dependencies
   ```bash
   # activate your venv (if you have one)
   # Linux/macOS
   source venv/bin/activate
   or 
   # Windows
   venv/Scripts/activate
   
   pip install -r requirements.txt   
   ```

3. Restart ComfyUI if it's already running.

------

## Nodes

### Show Anything

> Takes input from any (most?) nodes and displays it in a readable format and provides a Copy Text button for easily copying the displayed content.
>
> <img src="assets/show_anything.jpg" alt="show anything" width="100%"/>
> <details>
>   <summary>ℹ️ <i>See More Information</i></summary>
>   
>   - Based on other nodes that already work just fine. I just always wanted one with a `copy text` button for easy copying of long generated prompts (for editing or use elsewhere). It will primarily show `string, integer, float and boolean` values directly but will also (try to) display tensor data.
>
> </details>

---
### Image Row
> Takes up to 4 images and concats them together horizontally in a row with captions:
> 
> <img src="assets/image_row.jpg" alt="image row" width="100%"/>
> <details>
>   <summary>ℹ️ <i>See More Information</i></summary>
>
>   - Captions list will map too whatever images has been inputed
>   - Preview mode by default, switch to save mode.
> </details>

---

### CLIP Token Counter

> Takes text (string) as input and, using the CLIP tokenizer, displays token count and more:
> 
> <img src="assets/clip_token_count.jpg" alt="clip token counter" width="100%"/>
> <details>
>   <summary>ℹ️ <i>See More Information</i></summary>
>   
>   - Displays the number of tokens in the text  
>   - Whether the input exceeds the model's token limit  (77 - Including BOS and EOS)
>   - The final token within the range, along with surrounding context  
>   - All tokens within the limit, plus any overflow tokens beyond it
>   - Copy Text button: copies the contents displayed in the text widget
>   - Pass-though output for original text
>
> </details>

---

### T5 Token Counter

> Takes text (string) as input and, using the T5 XXL tokenizer, displays token count and more:
> 
> <img src="assets/t5_token_count.jpg" alt="t5 token counter" width="100%"/>
> <details>
>   <summary>ℹ️ <i>See More Information</i></summary>
>   
>   - Displays the number of tokens in the text  
>   - Whether the input exceeds the model's token limit  (256 or 512 - Including EOS)
>   - The final token within the range, along with surrounding context  
>   - All tokens within the limit, plus any overflow tokens beyond it
>   - Copy Text button: copies the contents displayed in the text widget
>   - Pass-though output for original text
>
> </details>

------

### Catch and Edit Text (Dual)
> Based on the original ![CatchEditTextNode by ImagineerNL](https://github.com/ImagineerNL/ComfyUI-IMGNR-Utils)
> A node that catches and shows text(s) generated from a previous node and enables editing the text for subsequent runs. Using the edited text also mutes the input node.  Modified from the original to take two text inputs to work with the Prompt Enhancer (shown below) and provide two text outputs.
> 
> <img src="assets/prompt_enhancer_flux_with_catch_edit_text.jpg" alt="catch edit text" width="100%"/>
> <details>
>   <summary>ℹ️ <i>See More Information</i></summary>
>
>   See screenshot for Prompt Enhancer below. 
>
>   This node acts as a receiver and editor for text sent from two sources.
action widget:
- use_input: pass the input text as it without modification.
- use_edit_mute_input: pass the edited texts in the widgets and mute the source node of the input(s).

If you just need one text input then I recommend using [ImagineerNL's original node](https://github.com/ImagineerNL/ComfyUI-IMGNR-Utils).
> </details>

------

### Y7 Prompt Enhancer (Flux)

> Takes any basic prompt and enhances it and produces T5 and CLIP friendly variants of the enhanced prompt. token / trigger words can be used in sq. brackets
> Example: [ohwx man], [agg woman], [sks dog]
>
> <img src="assets/prompt_enhancer_flux_with_catch_edit_text.jpg" alt="catch edit text" width="100%"/>
> <details>
>   <summary>ℹ️ <i>See More Information</i></summary>
>   
>   ![Prompt Enhancer (Flux) ](assets/prompt_enhancer_flux_with_catch_edit_text.jpg)
>   
>   Flux.1 uses two encoders: CLIP and T5 XXL. CLIP processes only the first 77 tokens (including <bos>/<eos>), and anything beyond that depends on the implementation. In ComfyUI, long prompts are split into 77-token chunks for CLIP, which are then batched and concatenated. On the other hand, T5, supports up to 512 tokens (or 256 in the "schnell" version) and works well with natural, descriptive language.
>   
>   Most users simply feed the same (T5) prompt into both encoders, as it's the most straightforward approach. However, because the first 77 tokens are shared by both encoders—and the rest are exclusive to T5—how you structure your prompt can make a big difference.
>   
>   Front-loading long prose too early can reduce CLIP's effectiveness, while cramming too many keywords up front may limit T5's ability to build nuance throughout the rest of the prompt.
>   
>   For (possibly) better results, a hybrid approach of starting with high-impact keywords to guide CLIP, then follow with flowing, descriptive language tailored for T5. This approach plays to the strengths of both encoders (again, possibly).
>   
>   **Token/Trigger words** are handled by enclosing them inside square brackets `[ohwx man]`, but occasionally it might not work.
>
>  There can be quirks in some of the responses generated, but it will get you most of the way in producing prompts in both formats very quickly and you can always edit them afterwards (in your own editor). 
>   
>   Four LLM models are available, offering a balance of knowledge, instruction-following, and minimal censorship.
> 
>   If you're using a GPU with limited VRAM, consider switching to 8-bit or 4-bit quantization to reduce memory usage (with some trade-offs in quality).
**Note: This requires BitsandBytes** which is primarily Linux-focused. Support for Windows and macOS can be tricky — and there might be workarounds, but they’re beyond the scope of this note.
If you're running ComfyUI inside WSL (Windows Subsystem for Linux), you should be fine
>
>   Additionally, you can choose to unload all models before each run — helpful for workflows involving other large models that remain cached. Alternatively, you can always use [SeanScripts's Unload Model custom nodes](https://github.com/SeanScripts/ComfyUI-Unload-Model) which provide a convenient way to handle this dynamically.
>
>   The node will attempt to download the selected model (approx 14.5GB) if it can't be found.  
>
>   If you wish to download the model(s) manually, links and paths shown below:
>   https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B 
>   
>   ```
>   ComfyUI
>   └── models
>       └── LLM
>           └── OpenHermes-2.5-Mistral-7B
>           |   ├── added_tokens.json
>           |   ├── config.json
>           |   ├── generation_config.json
>           |   ├── model-00001-of-00002.safetensors
>           |   ├── model-00002-of-00002.safetensors
>           |   ├── model.safetensors.index.json
>           |   ├── special_tokens_map.json
>           |   ├── tokenizer.model
>           |   ├── tokenizer_config.json
>           |   ├── transformers_inference.py>           
>   ```
>   
>   For https://huggingface.co/teknium/Hermes-Trismegistus-Mistral-7B (approx. 14.5GB)
>   
>   ```
>   ComfyUI
>   └── models
>       └── LLM        
>           └── Hermes-Trismegistus-Mistral-7B        
>           |   ├── added_tokens.json
>           |   ├── config.json
>           |   ├── generation_config.json
>           |   ├── pytorch_model-00001-of-00002.bin
>           |   ├── pytorch_model-00002-of-00002.bin
>           |   ├── pytorch_model.bin.index.json
>           |   ├── special_tokens_map.json
>           |   ├── tokenizer.model
>           |   ├── tokenizer_config.json
>   ```
>
>   For https://huggingface.co/cognitivecomputations/Dolphin3.0-Llama3.1-8B (approx. 16GB)
>   ```
>   ComfyUI
>   └── models
>       └── LLM        
>           └── Dolphin3.0-Llama3.1-8B        
>           |   ├── config.json
>           |   ├── generation_config.json
>           |   ├── model-00001-of-00004.safetensors
>           |   ├── model-00002-of-00004.safetensors
>           |   ├── model-00003-of-00004.safetensors
>           |   ├── model-00004-of-00004.safetensors
>           |   ├── model.safetensors.index.json
>           |   ├── special_tokens_map.json
>           |   ├── tokenizer_config.json
>           |   ├── tokenizer.json
>           |   ├── trainer_state.json
>   ```

>   For https://huggingface.co/Qwen/Qwen2.5-7B-Instruct (approx. 15.2GB)
>   ```
>   ComfyUI
>   └── models
>       └── LLM        
>           └── Qwen2.5-7B-Instruct      
>           |   ├── config.json
>           |   ├── generation_config.json
>           |   ├── merges.txt
>           |   ├── model-00001-of-00004.safetensors
>           |   ├── model-00002-of-00004.safetensors
>           |   ├── model-00003-of-00004.safetensors
>           |   ├── model-00004-of-00004.safetensors
>           |   ├── model.safetensors.index.json
>           |   ├── tokenizer.json
>           |   ├── tokenizer_config (1).json
>           |   ├── tokenizer_config.json
>           |   ├── vocab.json
>   ```


> </details>

------

### Y7 Prompt Enhancer (Flux2 Klein)

> Takes any basic prompt and enhances it specifically for FLUX.2 [klein] using the Qwen3-8B LLM model or an abliterated version. Features include customizable prompt instructions, thinking mode, and advanced generation parameters.
>
> <details>
>   <summary>ℹ️ <i>See More Information</i></summary>
>   
>   This node is designed specifically for FLUX.2 [klein] image generation and uses the Qwen3-8B model to transform basic prompts into detailed, high-quality prompts.
>   
>   **Key Features:**
>   
>   - **Qwen3-8B Model**: Uses advanced LLM for prompt enhancement
>   - **Josiefied-Qwen3-8B-abliterated Model**: Abliterated variant of the Qwen3-8B Model
>   - **Thinking Mode**: Enable/disable the model's reasoning process output
>   - **Customizable Instructions**: Load custom prompt enhancement instructions from `prompt_instructions/prompt_instruction_flux2_klein.txt`
>   - **Quantization Support**: Choose between none, 8-bit, or 4-bit quantization (requires bitsandbytes - primarily Linux)
>   - **Platform Support**: Works on CUDA, Apple Silicon (MPS), and CPU
>   - **Advanced Parameters**: Control temperature, top_p, top_k, and max_new_tokens for fine-tuned generation
>   - **Memory Management**: Option to keep model loaded for faster batch processing
>   - **Numerical Stability**: Built-in handling for numerical edge cases during generation
>   
>   **Outputs:**
>   - `thinking_output`: The model's reasoning process (when thinking mode is enabled)
>   - `enhanced_prompt`: The final enhanced prompt for FLUX.2 [klein]
>   
>   **Customizing Prompt Instructions:**
>   
>   To customize how the model enhances prompts:
>   1. Copy `example_system_messages.py`
>   2. Rename it to `system_messages.py`
>   3. Edit the file with your custom instructions
>   4. The node will automatically load your custom version
>   
>   **Model Information:**
>   
>   The HuggingFace transformers version of Qwen3-8B is Required and cannot use the Comfy-Org packaged version.
>   This node performs runtime text generation for prompt enhancement and reasoning using AutoModelForCausalLM and AutoTokenizer. These capabilities—tokenizer access, generation control, and model.generate()—are only available through the HuggingFace transformers API.
>   Comfy-Org’s Qwen models are optimized for inference-only graph execution inside ComfyUI and do not expose the full language-model interfaces required for programmatic text generation outside the standard sampling flow.
>   If Qwen3-8B is not found locally, the node will automatically download it (~16GB) from HuggingFace to models/LLM/Qwen3-8B.

>   
>   Manual download location (if needed):
>   https://huggingface.co/Qwen/Qwen3-8B  
>   https://huggingface.co/Goekdeniz-Guelmez/Josiefied-Qwen3-8B-abliterated-v1
>   
>   ```
>   ComfyUI
>   └── models
>       └── LLM
>           └── Qwen3-8B
>               ├── config.json
>               ├── generation_config.json
>               ├── merges.txt
>               ├── model.safetensors.index.json
>               ├── model-00001-of-00004.safetensors
>               ├── model-00002-of-00004.safetensors
>               ├── model-00003-of-00004.safetensors
>               ├── model-00004-of-00004.safetensors
>               ├── tokenizer.json
>               ├── tokenizer_config.json
>               └── vocab.json
>
>           └── Josiefied-Qwen3-8B-abliterated-v1
>               ├── added_tokens.json
>               ├── config.json
>               ├── generation_config.json
>               ├── merges.txt
>               ├── model-00001-of-00004.safetensors
>               ├── model-00002-of-00004.safetensors
>               ├── model-00003-of-00004.safetensors
>               ├── model-00004-of-00004.safetensors
>               ├── model.safetensors.index.json
>               ├── special_tokens_map.json
>               ├── tokenizer_config.json
>               ├── tokenizer.json
>               └── vocab.json
>   ```
>   
>   **Performance Tips:**
>   
>   - Use quantization (8-bit or 4-bit) if you have limited VRAM
>   - Enable "keep_model_loaded" when processing multiple prompts in succession
>   - Lower temperature values (0.4-0.7) produce more consistent results
>   - Adjust max_new_tokens based on desired prompt length
>
> </details>

------

### LM Studio Nodes — Prerequisites

> The **LM Studio (Text)**, **LM Studio (Vision)**, and **Select LMS Model** nodes all require a running [LM Studio](https://lmstudio.ai/) server. LM Studio is a free desktop application for running LLMs locally.
>
> <details>
>   <summary>ℹ️ <i>LM Studio Server Setup</i></summary>
>   
>   **Local Setup (same machine as ComfyUI):**
>   
>   1. Download and install [LM Studio](https://lmstudio.ai/)
>   2. Download a model through the LM Studio interface (for vision nodes, ensure you pick a VL model, e.g. Qwen2.5-VL, Gemma3, etc.)
>   3. Load the model in LM Studio
>   4. Start the local server: go to the **Developer** tab (or **Local Server** in older versions) and click **Start Server**
>   5. By default, the server runs on `localhost:1234` — this matches the default `ip` and `port` values in the nodes
>   
>   **Network Setup (LM Studio on a different machine):**
>   
>   If LM Studio is running on another machine on your network:
>   
>   1. In LM Studio's server settings, enable **Serve on Local Network** (this binds the server to `0.0.0.0` instead of `127.0.0.1`)
>   2. Note the IP address of the machine running LM Studio (e.g., `192.168.1.100`)
>   3. In the ComfyUI node, set the `ip` field to that machine's IP address and ensure the `port` matches (default: `1234`)
>   4. Make sure there are no firewall rules blocking the port between the two machines
>   
>   **Model Identifier:**
>   
>   The `model_identifier` should match the model name as it appears in LM Studio. You can use the **Select LMS Model** node to pick from a predefined list stored in `comfyui-y7nodes/lms_config/models.txt` (one model name per line).
>   
>   **Python Package:**
>   
>   These nodes require the `lmstudio` Python SDK: `pip install lmstudio`
>
> </details>

---

### Y7 LM Studio (Text)

> Send text prompts to a local LM Studio server for text generation and prompt enhancement using any LLM loaded in LM Studio. Supports speculative decoding via a draft model.
>
> <details>
>   <summary>ℹ️ <i>See More Information</i></summary>
>   
>   Connects to an LM Studio server and sends a text prompt along with a system message to guide the model's behavior. The default system message is optimized for AI image prompt enhancement, but can be customized for any text generation task.
>   
>   **Key Features:**
>   
>   - **System Message**: Customizable system prompt that guides the LLM's behavior (default: image prompt enhancement)
>   - **Draft Model**: Optional speculative decoding support for faster generation
>   - **Reasoning Extraction**: Automatically separates thinking/reasoning blocks from the response
>   - **Memory Management**: Options to unload the LLM after generation and/or free ComfyUI VRAM beforehand
>   - **Fallback Handling**: Automatically retries with an alternative chat template if the first attempt fails
>   
>   **Inputs:**
>   
>   - `prompt`: The text prompt to send to the LLM (connected from another node)
>   - `model_identifier`: The model name/identifier loaded in LM Studio (connect a Select LMS Model node or type manually)
>   - `draft_model`: Optional speculative decoding draft model name (leave empty to disable)
>   - `system_message`: System prompt that guides the LLM's behavior
>   - `reasoning_tag`: Tag name used to extract reasoning blocks (e.g., `think` for `<think>...</think>`)
>   - `ip` / `port`: LM Studio server address (default: localhost:1234)
>   - `temperature`: Controls randomness (0.01–1.0, default 0.7)
>   - `max_tokens`: Maximum tokens to generate (-1 for unlimited)
>   - `unload_llm`: Unload the LLM from LM Studio after generation
>   - `unload_comfy_models`: Free VRAM by unloading ComfyUI models before running the LLM
>   
>   **Outputs:**
>   
>   - `Extended Prompt`: The generated text with reasoning blocks removed
>   - `Reasoning`: The extracted reasoning content (if present)
>   
>   **Requirements:**
>   
>   - LM Studio running locally (or on a network-accessible machine)
>
> </details>

---

### Y7 LM Studio (Vision)

> Send an image to a vision-capable LLM (VL model) in LM Studio for analysis and description. The instruction is provided via the system message — no separate text prompt input.
>
> <details>
>   <summary>ℹ️ <i>See More Information</i></summary>
>   
>   Connects to an LM Studio server and sends an image along with an instruction to a vision-language (VL) model. The system message acts as the sole instruction for how the model should interpret the image. The model must be vision-enabled or an error will be raised.
>   
>   **Key Features:**
>   
>   - **Vision-First Design**: Image is a required input — purpose-built for VL models
>   - **Instruction via System Message**: The system message is sent alongside the image as the user instruction (default: detailed image description)
>   - **Model Validation**: Checks that the loaded model supports vision before proceeding
>   - **Reasoning Extraction**: Automatically separates thinking/reasoning blocks from the response
>   - **Memory Management**: Options to unload the LLM after generation and/or free ComfyUI VRAM beforehand
>   
>   **Inputs:**
>   
>   - `image`: The image to analyze (required)
>   - `model_identifier`: The VL model name/identifier loaded in LM Studio (connect a Select LMS Model node or type manually)
>   - `system_message`: The instruction sent alongside the image (default: describe the image in detail)
>   - `reasoning_tag`: Tag name used to extract reasoning blocks (e.g., `think` for `<think>...</think>`)
>   - `ip` / `port`: LM Studio server address (default: localhost:1234)
>   - `temperature`: Controls randomness (0.01–1.0, default 0.7)
>   - `max_tokens`: Maximum tokens to generate (-1 for unlimited)
>   - `unload_llm`: Unload the LLM from LM Studio after generation
>   - `unload_comfy_models`: Free VRAM by unloading ComfyUI models before running the LLM
>   
>   **Outputs:**
>   
>   - `Response`: The model's analysis/description with reasoning blocks removed
>   - `Reasoning`: The extracted reasoning content (if present)
>   
>   **Requirements:**
>   
>   - LM Studio running locally (or on a network-accessible machine)
>   - A vision-capable model loaded in LM Studio (non-vision models will raise an error)`
>
> </details>

---

### Y7 Select LMS Model

> Select an LM Studio model from a predefined list stored in a text file. Outputs the model identifier string to connect to the LM Studio Text or Vision nodes.
>
> <details>
>   <summary>ℹ️ <i>See More Information</i></summary>
>   
>   Provides a dropdown of model identifiers loaded from `comfyui-y7nodes/lms_config/models.txt`. Add your favorite model names (one per line) to this file.
>   
>   **Output:**
>   
>   - `model_id`: The selected model identifier string
>
> </details>

---

### Y7 Image Size Presets
> Select predefined image size/aspect ratios loaded from a custom json file. Provides width and height outputs.
>
> <details>
>   <summary>ℹ️ <i>See More Information</i></summary>
>   - Provides outputs for `width` and `height` (INT).
>   - Looks for and loads presets from `custom_dimensions.json` located in the `nodes` directory. You can edit this file to add your own presets.
> </details>

---

### Y7 Crop to Resolution
> Automatically crops images to ensure dimensions are divisible by a specified value (e.g., 8 or 16), with visual preview of crop areas and independent horizontal/vertical control.
>
> <details>
>   <summary>ℹ️ <i>See More Information</i></summary>
>   
>   Many AI models require image dimensions to be divisible by specific values (typically 8 or 16) for proper processing. This node automatically checks image dimensions and crops them to meet these requirements while giving you precise control over where the crop occurs.
>   
>   **Key Features:**
>   
>   - **Visual Preview**: Shows original image with semi-transparent red overlay indicating areas that will be cropped
>   - **Independent Control**: Separate horizontal and vertical crop position settings
>   - **Smart Logic**: Only crops dimensions that need adjustment, ignoring dimensions that are already aligned
>   - **Flexible Positioning**: Choose `center`, `left`, `right`, `top`, `bottom`, or `none` for each axis
>   - **Informative Output**: Provides detailed status messages about dimensions and any cropping performed
>   - **On-Node Display**: Shows the cropped dimensions (e.g., '1024 x 768') directly on the node after execution
>   
>   **Inputs:**
>   
>   - `multiple`: The value dimensions must be a multiple of (default: 16). Common values are 8 or 16 for most AI models
>   - `horizontal_crop`: Where to keep content when width needs adjustment - `center`, `left`, `right`, or `none`
>   - `vertical_crop`: Where to keep content when height needs adjustment - `center`, `top`, `bottom`, or `none`
>   
>   **Outputs:**
>   
>   - `crop_preview`: Original image with red overlay showing what will be cropped (useful for previewing before committing)
>   - `image`: The cropped result (or original if no cropping needed)
>   - `info`: Status message with dimension details and cropping information
>   
>   **Behavior Notes:**
>   
>   - Crops to the nearest multiple down (e.g., 721 → 720 with multiple=16)
>   - Setting a crop position to `none` disables cropping for that dimension
>   - When using `center` with odd-numbered pixel differences, integer division rounds down (e.g., width=721 removes 1px from right only)
>   - This slight bias is standard in image processing and is minimal (max 1 pixel difference)
>   
>   **Use Cases:**
>   
>   - Preparing images for models that require specific dimension constraints
>   - Cropping images from one dimension while keeping the other intact
>   - Quick visual verification of crop areas before applying
>
> </details>

---

## Example Workflows

Example workflows can be found in the `workflows` directory. 

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Credits

- ShowAnything node is based on "Show Any" from yolain's ComfyUI-Easy-Use custom nodes and "Show Any To JSON" from crystian's ComfyUI-Crystools custom nodes, with additional formatting controls and a Copy Text button.
- Help popup system is based on the implementation from Kosinkadink's ComfyUI-VideoHelperSuite.
