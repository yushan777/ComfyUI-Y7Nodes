from ..utils.logger import logger

def title(text):
    """Title text element"""
    return f'<div style="margin-bottom: 10px;">{text}</div>'

def short_desc(desc):
    """Create a short description element with the special ID"""
    return f'<div id="Y7_shortdesc" style="margin-bottom: 15px;">{desc}</div>'

def process_highlights(text):
    """Process text and convert `highlighted` parts to code style that works in both light and dark themes"""
    import re
    pattern = r'`([^`]+)`'
    # Theme-agnostic styling:
    return re.sub(pattern, r'<code style="border: 1px solid #666; border-radius: 3px; padding: 0px 1px; font-family: monospace; display: inline-block;">\1</code>', text)
    
# Then modify normal to use this
def normal(text, indent_level=0, font_size="12px"):
    """Normal text element with optional indentation and font size"""
    indent_px = indent_level * 20  # 20px per indent level
    processed_text = process_highlights(text)
    return f'<div style="margin-bottom: 8px; margin-left: {indent_px}px; font-size: {font_size};">{processed_text}</div>'

descriptions = {
    # "Y7Nodes_Brightness": [
    #     "Image Brightness Adjustment (demo node)",
    #     short_desc("Adjusts the brightness of an image by multiplying pixel values"),
    #     normal("Control the brightness level with the `strength` parameter:"),
    #     normal("- Values > 1.0 increase brightness", 1),
    #     normal("- Values < 1.0 decrease brightness", 1),
    #     normal("- Value of 1.0 leaves the image unchanged", 1),
    #     normal("Optionally provide a `filename_prefix` to use when saving the processed image.")
    # ],

    "Y7Nodes_ImageStitcher": [
        "Image Stitcher",
        short_desc("Stitches 2–8 images side-by-side or top-and-bottom"),
        normal("Use the `image_count` widget to choose how many image inputs are shown (2–8)."),
        normal("All images are resized to match the first image's height (horizontal) or width (vertical) before concatenation."),
        normal("Inputs:"),
        normal("- `image_count`: Number of image sockets to display (2–8)", 1),
        normal("- `orientation`: `Side-by-Side (Horizontal)` or `Top-and-Bottom (Vertical)`", 1),
        normal("- `image1` … `imageN`: Images to stitch together", 1),
    ],

    "Y7Nodes_ImageCompare": [
        "Image Compare",
        short_desc("Compare two images with a draggable slider and selectable blend modes."),
        normal("Provides an interactive side-by-side comparison directly on the node. Drag the slider to reveal `image_a` over `image_b`, and switch blend modes to analyse differences."),
        normal("The preview updates live as the slider is moved or the blend mode is changed, and persists across workflow-tab switches."),
        normal("Blend modes:"),
        normal("- `normal`: Slider reveals `image_a` over `image_b`", 1),
        normal("- `difference`: Blended comparison for visual analysis of variations", 1),
        normal("Inputs:"),
        normal("- `image_a`: First image (required)", 1),
        normal("- `image_b`: Second image (optional)", 1),
        normal("This node has no outputs; it is a preview-only node for on-canvas comparison."),
        normal("Quality of life:"),
        normal("- The node auto-resizes to match the aspect ratio of the input images", 1),
        normal("- Slider position and blend mode are saved with the workflow", 1),
        normal(""),
        normal("Based on `Eses Image Compare` by Eses Nodes."),
    ],

    "Y7Nodes_Text":[
        "Basic Text Input With Copy Button",
        short_desc("Basic Text Input With Copy Button"),
    ],
    
    "Y7Nodes_ShowAnything": [
        "Show Anything",
        short_desc("Display the content of any input, regardless of its type."),
        normal("A debugging tool that displays information about any input in the ComfyUI interface."),
        normal("For string, integer, float, boolean values: Displays the content directly"),
        normal("For IMAGE and MASK tensors: Shows shape, data type, value range, mean, and std dev."),
        normal("For other tensors: Displays shape, data type, and value range"),                
        normal("For other types: Converts to JSON or string representation"),
        normal("Pass-through for the input.")
    ],
    
    "Y7Nodes_T5_TokenCounter": [
        "T5 V1.1 XXL Token Counter",
        short_desc("Counts tokens in a text using the T5 XXL tokenizer."),
        normal("Up to the first 256 or 512 tokens (default) will be displayed, with any overflow tokens shown below."),
        normal("The actual limit is 256 or 512 minus one special token reserved for the End-of-Sequence token `&lt;/s&gt;`"),
        normal("Some models such as Flux.1 Schnell impose a 256-token sequence limit."),
        normal("The final token in the sequence will be shown, along with a brief context of words leading up to it."),
        normal("Tokens prefixed with an underscore '_' represent a word boundary (New sentence or a space)."),
        normal("Tokens without an underscore '_' are usually subword pieces that continue from the previous token."),        
        normal("Inputs:"),
        normal("- text_in: Any text (string) input.", 1),
        normal("- show_tokens: Displays tokenized version of text (requires re-run).", 1),
        normal("- tokens_per_line: Number of token words per line (requires re-run).", 1),        
        normal("Output:"),
        normal("- text_out: A pass-through output for the input string", 1),
        normal("Widgets:"),
        normal("- font_size: Change font size used in the text widget. ", 1),
        normal("- Copy Text: Copy contents of the text widget.", 1),        
        normal(""),
        normal("Note: Longer prompts are supported, but how they are handled depends entirely on the specific "
        "implementation of the model and tokenizer. Some implementations may truncate, segment, or otherwise process longer inputs differently.")        
    ],

    "Y7Nodes_CLIP_TokenCounter": [
        "CLIP Token Counter",
        short_desc("Counts tokens in a text using the CLIP tokenizer."),
        normal("Up to the first 77 tokens will be displayed, with any overflow tokens shown below."),
        normal("The actual limit is 77 tokens, including two special tokens: `&lt;|startoftext|&gt;` (Beginning-of-Sequence) and `&lt;|endoftext|&gt;` (End-of-Sequence)."),
        normal("The final token in the sequence will be shown, along with a brief context of words leading up to it."),
        normal("The `&lt;/w&gt;` marker indicates a word boundary, typically where a space followed the word in the original text."),
        normal("Inputs:"),
        normal("- text_in: Any text (string) input.", 1),
        normal("- show_tokens: Displays tokenized version of text (requires re-run).", 1),
        normal("- tokens_per_line: Number of token words per line (requires re-run).", 1),        
        normal("Output:"),
        normal("- text_out: A pass-through output for the input string", 1),
        normal("Other Widgets:"),
        normal("- font_size: Change font size used in the text widget. ", 1),
        normal("- Copy Text: Copy contents of the text widget.", 1),
        normal(""),
        normal("Note: Longer prompts are supported, but how they are handled depends entirely on the specific "
            "implementation of the model and tokenizer. Some implementations may truncate, segment, or otherwise process longer inputs differently.")
    ],

    "Y7Nodes_PromptEnhancerNative":[
        "Prompt Enhancer (Native)",
        short_desc("Takes a basic prompt and enhances it using any generation-capable text encoder already loaded by ComfyUI"),
        normal("Unlike the other Y7 prompt enhancers, this node downloads and loads nothing itself. It takes a `CLIP` input from a standard `CLIPLoader` and lets ComfyUI handle all model loading and VRAM management. Nothing stays resident that ComfyUI is not already managing."),
        normal("Supported models:"),
        normal("- `Gemma 4` - E2B, E4B, 31B and the 12B unified model. Also handles image, video and audio, though this node only sends text.", 1),
        normal("- `Gemma 3` - the 12B model, including the LTX-2 text encoder built on it.", 1),
        normal("- `Qwen3` - 0.6B, 2B, 4B and 8B.", 1),
        normal("- `Qwen3.5` - 0.8B, 2B, 4B, 9B and 27B.", 1),
        normal("- `Qwen3-VL` - 4B and 8B.", 1),
        normal("Not supported - these encoders have no text generation path and the node will say so rather than failing obscurely:"),
        normal("- `T5` (all sizes), `UMT5`, `CLIP-L`, `CLIP-G`, `Gemma 2`, `LLaMA-3.1`", 1),
        normal("Weights must be safetensors placed in `models/text_encoders/`. GGUF will not work: ComfyUI core cannot load GGUF at all, and for Gemma 4 the tokenizer is embedded inside the safetensors file itself. The `CLIPLoader` `type` dropdown is ignored for Gemma 4 - the model is detected from the weights, so any value works."),
        normal("Inputs:"),
        normal("- `clip`: A generation-capable text encoder from `CLIPLoader` - see the supported models above.", 1),
        normal("- `text`: Your basic prompt to enhance. Accepts a connection from any string node.", 1),
        normal("- `instruction`: The instruction placed before your text. Edit this to change the style of enhancement.", 1),
        normal("- `max_length`: Maximum NEW tokens to generate (64-32768, default 2048). This is not the context window. Reasoning is spent from the same budget, and the KV cache reserves ~84KB of VRAM per token up front.", 1),
        normal("- `temperature`: Controls randomness (0.0-2.0, default 1.0, Google's recommended value for Gemma). 0 switches to greedy decoding and ignores top_k/top_p.", 1),
        normal("- `top_k`: Limits selection to the k most likely tokens (0-1000, default 64, Google's recommended value). 0 disables.", 1),
        normal("- `top_p`: Nucleus sampling (0.0-1.0, default 0.95, Google's recommended value). 1.0 disables.", 1),
        normal("- `seed`: Random seed. Change it to re-roll - identical inputs return a cached result.", 1),
        normal("- `thinking`: Let the model reason before answering. Its reasoning is always separated out, never mixed into the prompt.", 1),
        normal("Outputs:"),
        normal("- `thinking_output`: The model's reasoning, if it produced any", 1),
        normal("- `enhanced_prompt`: The enhanced prompt, with all reasoning removed", 1),
        normal("On reasoning:"),
        normal("Gemma 4 emits reasoning inside a thought channel, and ComfyUI's decoder deliberately keeps that text rather than discarding it. Setting `thinking` to False only primes the model to skip it - Gemma 4 frequently reasons anyway, which is why the core `Generate Text` node can return a wall of planning notes ahead of the actual prompt."),
        normal("This node always splits the two apart, including the awkward case where the model reasons past the primed channel and closes it with an orphan tag. If `enhanced_prompt` ever comes back empty, the model spent the whole `max_length` budget reasoning - raise it, or lower `temperature`.")
    ],

    "Y7Nodes_CatchEditTextNodeDual": [
        "Catch and Edit Text (Dual)",
        short_desc("Catches text from two separate inputs making them editable"),
        normal("This node acts as a receiver and editor for text sent from two sources."),
        normal("action widget:"),
        normal("- use_input: pass the input text as it without modification.",1),
        normal("- use_edit_mute_input: pass the edited text in the widget and mute the source node of the input(s).",1),
        normal(""),
        normal("Based on the original CatchEditTextNode by ImagineerNL"),
        normal("https://github.com/ImagineerNL/ComfyUI-IMGNR-Utils"),
        normal("If you just need one text input then I recommend using his original node."),

    ],

    "Y7Nodes_ImageSizePresets": [
        "Image Size Presets",
        short_desc("Node to provide image width and height from a named preset set, with an optional custom dimensions file."),
        normal("Inputs:"),
        normal("- `preset`: Selects the active dimension set: `default`, `flux.2`, `qwen-image`, or `custom*`.", 1),
        normal("- `dimension`: Dropdown of dimensions for the selected preset set. Updates dynamically when preset changes.", 1),
        normal("- `custom_w`: Width to use when 'Custom' is selected from the dimension dropdown.", 1),
        normal("- `custom_h`: Height to use when 'Custom' is selected from the dimension dropdown.", 1),
        normal("Outputs:"),
        normal("- `Width`: The selected or custom width.", 1),
        normal("- `Height`: The selected or custom height.", 1),
        normal("The `custom*` preset loads from `custom_dimensions.json` in the `nodes` directory; falls back to `default` if missing or invalid."),
        normal("Examine `custom_dimensions_example.json` for the expected format."),
    ],

    "Y7Nodes_AspectRatioPicker": [
        "Aspect Ratio Picker",
        short_desc("Interactive 2D canvas for picking image width and height by dragging."),
        normal("Click or drag anywhere in the canvas to set width (X axis, left→right) and height (Y axis, bottom→top)."),
        normal("A filled rectangle shows the selected proportions; the dot marks the current position."),
        normal("The simplified aspect ratio (e.g. 16:9, 4:3, 1:1) is displayed at the bottom of the canvas."),
        normal("Current width and height values are shown in the right panel. Double-click either value to type a number directly."),
        normal("Click the Swap button (below the height value) to swap width and height (portrait ↔ landscape)."),
        normal("Snapping to step increments is on by default. Hold Shift while dragging to temporarily disable snapping."),
        normal("Node properties (right-click → Properties):"),
        normal("- `valueX` / `valueY`: Current width and height.", 1),
        normal("- `minX` / `maxX` / `minY` / `maxY`: Range for each axis.", 1),
        normal("- `stepX` / `stepY`: Snap increment for each axis.", 1),
        normal("- `snap`: Whether dragging snaps to step increments by default.", 1),
        normal("- `dots`: Show grid dots at each step position.", 1),
        normal("Outputs:"),
        normal("- `width`: Selected width in pixels (INT).", 1),
        normal("- `height`: Selected height in pixels (INT).", 1),
        normal("The right panel also shows the total megapixel count (e.g. `1.0MP` at 1024×1024)."),
    ],

    "Y7Nodes_ImagePadForOutpaint": [
        "Pad Image for Outpainting",
        short_desc("Clone of the built-in Pad Image for Outpainting node, with a `step` input that snaps each padding value to a chosen multiple."),
        normal("Pads an image on any/all sides ready for outpainting, generating a feathered mask over the new regions."),
        normal("Inputs:"),
        normal("- `image`: The input image to pad", 1),
        normal("- `left` / `top` / `right` / `bottom`: Pixels to add to each edge", 1),
        normal("- `feathering`: Width of the soft gradient at the border between original and padded regions", 1),
        normal("- `step`: Each of `left`/`top`/`right`/`bottom` is independently snapped to the nearest multiple of this value, ties rounding up, before padding is applied (default: 8)", 1),
        normal("Outputs:"),
        normal("- `IMAGE`: The padded image", 1),
        normal("- `MASK`: Mask covering the padded regions, feathered at the border", 1),
    ],

    "Y7Nodes_CropToNearestMultiple": [
        "Crop to Nearest Multiple",
        short_desc("Crops images to ensure dimensions are divisible by a specified multiple value - default 16px. You can always use a calculator if you prefer."),
        normal("Control over horizontal and vertical cropping, making it ideal for situations where only one dimension needs adjustment."),
        normal("Inputs:"),
        normal("- `image`: The input image to check and optionally crop", 1),
        normal("- `multiple`: The value that dimensions must be a multiple of (default: 16). Common values are 8, 16, 32", 1),
        normal("- `horizontal_crop`: Crop position for width adjustment - `center`, `left`, `right`, or `none`", 1),
        normal("- `vertical_crop`: Crop position for height adjustment - `center`, `top`, `bottom`, or `none`", 1),
        normal("Outputs:"),
        normal("- `crop_preview`: Original image with a red overlay showing the areas that will be cropped and removed", 1),
        normal("- `cropped_image`: The cropped image (or original if no cropping needed)", 1),
        normal("- `info`: Status message with dimension information and cropping details", 1),
        normal("Behavior:"),
        normal("- If both dimensions are already a multiple of the specified value, no cropping occurs", 1),
        normal("- Crops to the nearest multiple down (e.g., 721 → 720 with multiple=16)", 1),
        normal("- Setting crop position to `none` disables cropping for that dimension", 1),
        normal("- Smart logic only applies crop settings to dimensions that need adjustment", 1),
        normal("- The node displays the cropped dimensions (e.g., '1024 x 768') directly on the node after execution", 1),
        normal("Note on 'center' cropping:"),
        normal("When using `center` with odd-numbered pixel differences, integer division rounds down, causing a slight bias (max 1px).", 1),
        normal("Example: width=721, target=720, diff=1 → removes 1px from right only", 2),
        normal("Example: width=723, target=720, diff=3 → removes 1px from left, 2px from right", 2),
        normal("This is standard behavior in image processing and the bias is minimal.", 1)
    ],

    "Y7Nodes_ScaleImageBy": [
        "Scale Image By",
        short_desc("Scales an image by a multiplier while preserving aspect ratio."),
        normal("Multiplies both width and height by `scale_by`, then resamples using the chosen method."),
        normal("The scaled dimensions are displayed directly on the node after execution."),
        normal("Inputs:"),
        normal("- `image`: The input image to scale", 1),
        normal("- `upscale_method`: Resampling algorithm — `nearest-exact`, `bilinear`, `area`, `bicubic`, or `lanczos`", 1),
        normal("- `scale_by`: Multiplier applied to both dimensions (default: 1.0, range: 0.01–8.0)", 1),
        normal("- `resolution_steps`: Snap output dimensions to the nearest multiple of this value (default: 8). Common values: 8, 16, 64", 1),
        normal("Outputs:"),
        normal("- `image`: The scaled image", 1),
        normal("- `width`: Output width in pixels (INT)", 1),
        normal("- `height`: Output height in pixels (INT)", 1),
    ],

    "Y7Nodes_ScaleImageToTotalPixels": [
        "Scale Image to Total Pixels",
        short_desc("Scales an image to a target total pixel count (in megapixels) while preserving aspect ratio."),
        normal("Computes a uniform scale factor so that `width × height` equals the target megapixel count, then resamples using the chosen method."),
        normal("The scaled dimensions are displayed directly on the node after execution."),
        normal("Inputs:"),
        normal("- `image`: The input image to scale", 1),
        normal("- `upscale_method`: Resampling algorithm — `nearest-exact`, `bilinear`, `area`, `bicubic`, or `lanczos`", 1),
        normal("- `megapixels`: Target total pixel count in megapixels (default: 1.0, range: 0.01–16.0)", 1),
        normal("- `resolution_steps`: Snap output dimensions to the nearest multiple of this value (default: 8). Common values: 8, 16, 64", 1),
        normal("Outputs:"),
        normal("- `image`: The scaled image", 1),
        normal("- `width`: Output width in pixels (INT)", 1),
        normal("- `height`: Output height in pixels (INT)", 1),
    ],

    "Y7Nodes_ColorMatchMasked": [
        "Color Match (Masked)",
        short_desc("Color matches the target image to a reference while excluding masked regions from the calculation."),
        normal("Ideal for correcting color shifts after inpainting. The mask excludes regions (like the inpainted area) from BOTH images during color transfer calculation, preventing the original colors from bleeding into the result."),
        normal("Example: After changing a red car to blue via inpainting, the background may have a red tint. This node calculates color correction using only the non-masked areas, then applies it without affecting the inpainted region."),
        normal("Inputs:"),
        normal("- `image_ref`: Reference image (e.g., original before inpainting)", 1),
        normal("- `image_target`: Target image to color match (e.g., result after inpainting)", 1),
        normal("- `mask`: Mask where white (1.0) = areas to exclude from color matching", 1),
        normal("- `method`: Color transfer algorithm - `mkl` (Monge-Kantorovich), `hm` (histogram), `reinhard`, `mvgd` (Multi-Variate Gaussian)", 1),
        normal("- `strength`: Blend between original and color-matched result (0.0 = no change, 1.0 = full correction)", 1),
        normal("- `feather`: Blur radius for mask edges to create smooth transitions (0-100 pixels)", 1),
        normal("Output:"),
        normal("- `image`: The color-matched result with masked areas preserved unchanged", 1),
        normal("Requires the `color-matcher` library: `pip install color-matcher`")
    ],

    "Y7Nodes_LMStudioText": [
        "LM Studio (Text)",
        short_desc("Send text prompts to a local LM Studio server for text generation and prompt enhancement."),
        normal("Connects to an LM Studio server to generate or enhance text using a specified LLM. Uses a system message to guide the model's behavior and a user prompt as input."),
        normal("Inputs:"),
        normal("- `prompt`: The text prompt to send to the LLM", 1),
        normal("- `model_identifier`: The model name/identifier loaded in LM Studio (connect a Select LMS Model node or type manually)", 1),
        normal("- `draft_model`: Optional speculative decoding draft model name (leave empty to disable)", 1),
        normal("- `system_message`: System prompt that guides the LLM's behavior (default is optimized for image prompt enhancement)", 1),
        normal("- `reasoning_tag`: Tag name used to extract reasoning blocks (e.g., `think` for `<think>...</think>`)", 1),
        normal("- `ip` / `port`: LM Studio server address (default: localhost:1234)", 1),
        normal("- `temperature`: Controls randomness (0.01-1.0, default 0.7)", 1),
        normal("- `max_tokens`: Maximum tokens to generate (-1 for unlimited)", 1),
        normal("- `unload_llm`: Unload the LLM from LM Studio after generation", 1),
        normal("- `unload_comfy_models`: Free VRAM by unloading ComfyUI models before running the LLM", 1),
        normal("Outputs:"),
        normal("- `Extended Prompt`: The generated text with reasoning blocks removed", 1),
        normal("- `Reasoning`: The extracted reasoning content (if present)", 1),
        normal("Requires the `lmstudio` Python package: `pip install lmstudio`"),
    ],

    "Y7Nodes_LMStudioVision": [
        "LM Studio (Vision)",
        short_desc("Send an image to a vision-capable LLM in LM Studio for analysis and description."),
        normal("Connects to an LM Studio server and sends an image along with an instruction to a vision-language (VL) model. The model must be vision-enabled or an error will be raised."),
        normal("Inputs:"),
        normal("- `image`: The image to analyze (required)", 1),
        normal("- `model_identifier`: The VL model name/identifier loaded in LM Studio (connect a Select LMS Model node or type manually)", 1),
        normal("- `system_message`: The instruction sent alongside the image (default describes the image in detail)", 1),
        normal("- `reasoning_tag`: Tag name used to extract reasoning blocks (e.g., `think` for `<think>...</think>`)", 1),
        normal("- `ip` / `port`: LM Studio server address (default: localhost:1234)", 1),
        normal("- `temperature`: Controls randomness (0.01-1.0, default 0.7)", 1),
        normal("- `max_tokens`: Maximum tokens to generate (-1 for unlimited)", 1),
        normal("- `unload_llm`: Unload the LLM from LM Studio after generation", 1),
        normal("- `unload_comfy_models`: Free VRAM by unloading ComfyUI models before running the LLM", 1),
        normal("Outputs:"),
        normal("- `Response`: The model's analysis/description with reasoning blocks removed", 1),
        normal("- `Reasoning`: The extracted reasoning content (if present)", 1),
        normal("Requires the `lmstudio` Python package: `pip install lmstudio`"),
        normal("Note: The loaded model must support vision. Non-vision models will raise an error."),
    ],

    "Y7Nodes_SelectLMSModel": [
        "Select LMS Model",
        short_desc("Select an LM Studio model from a predefined list."),
        normal("Provides a dropdown of model identifiers loaded from a `models.txt` file."),
        normal("Add your favorite model names (one per line) to `comfyui-y7nodes/lms_config/models.txt`."),
        normal("Output:"),
        normal("- `model_id`: The selected model identifier string", 1),
    ],

    "SamplerSelect_Name": [
        "Sampler Select (Name)",
        short_desc("Select a sampler by name and output it as a linkable string."),
        normal("ComfyUI's built-in KSampler nodes define `sampler_name` as a COMBO widget with no input socket, so it can't receive a node connection. This node exposes the selection as a typed output that can be wired into any node that accepts a sampler name."),
        normal("Output:"),
        normal("- `sampler_name`: The selected sampler name", 1),
    ],

    "Y7Nodes_PasteCroppedImageBack": [
        "Paste Cropped Image Back",
        short_desc("Paste a crop image onto a base image at a region defined by edge-relative coordinates."),
        normal("Sometimes you may want to change or refine a specific area of an image without affecting the rest too much — for example, fixing a face, hand, or background detail after generation. A typical workflow is to crop the region, run it through img2img or inpainting, then paste the result back using this node."),
        normal("Works well with the `OLM Drag Crop` custom node, which lets you visually drag-select a crop region and outputs the crop coordinates directly — those coordinates can be wired into this node's `top`, `left`, `right`, and `bottom` inputs."),
        normal("Unlike the WAS equivalent, `right` and `bottom` are pixel offsets measured inward from the RIGHT and BOTTOM edges of the base image, rather than absolute coordinates from the top-left."),
        normal("Paste region calculated as:"),
        normal("- `x1 = left`", 1),
        normal("- `y1 = top`", 1),
        normal("- `x2 = image_width - right`", 1),
        normal("- `y2 = image_height - bottom`", 1),
        normal("The `image_crop` is always resized to exactly fit the paste region. If it was upscaled for editing (e.g. sent through img2img at a higher resolution), it will be scaled back down during pasting. There is no aspect-ratio preservation — if the aspect ratio of the crop image differs from the paste region, it will be stretched to fit and appear distorted."),
        normal("Inputs:"),
        normal("- `image_orig`: Base image to paste onto", 1),
        normal("- `image_crop`: Image to paste into the defined region (will be resized to fit)", 1),
        normal("- `left`: Pixels from the left edge to the left of the paste region", 1),
        normal("- `top`: Pixels from the top edge to the top of the paste region", 1),
        normal("- `right`: Pixels inward from the RIGHT edge to the right of the paste region", 1),
        normal("- `bottom`: Pixels inward from the BOTTOM edge to the bottom of the paste region", 1),
        normal("- `crop_blending`: Feathering amount at the edges of the pasted region (0.0–1.0)", 1),
        normal("- `crop_sharpening`: Number of sharpening passes applied to the crop before pasting (0–3)", 1),
        normal("Outputs:"),
        normal("- `IMAGE`: The base image with the crop pasted in", 1),
        normal("- `MASK`: The blended mask used for the paste operation", 1),
    ],

    "Y7Nodes_ImageBatchPath": [
        "Image Batch Path",
        short_desc("Load a batch of images from a directory and output them as a list of image tensors with matching file paths."),
        normal("Supports jpg, jpeg, png, and webp. Images are EXIF-transposed and converted to RGB float32 tensors."),
        normal("Designed to pair with CaptionSaver: the `IMAGE_PATH` output tells CaptionSaver where to write each .txt file, and `IMAGE` feeds into a VLM node for captioning."),
        normal("Inputs:"),
        normal("- `image_dir`: Path to the directory containing images", 1),
        normal("- `batch_size`: Number of images to load (0 = all images in the directory)", 1),
        normal("- `start_from`: 1-based index of the first image to load — useful for resuming from a specific point", 1),
        normal("- `sort_method`: Order to load images — `sequential` (alphabetical), `reverse`, or `random`", 1),
        normal("Outputs (both are lists):"),
        normal("- `IMAGE`: List of image tensors (one per image)", 1),
        normal("- `IMAGE_PATH`: List of full file paths matching each image tensor", 1),
        normal("Note: When `sort_method` is `random`, the node re-evaluates on every run."),
    ],

    "Y7Nodes_LoadImage": [
        "Load Image",
        short_desc("Load an image with support for subdirectories — otherwise identical to the native Load Image node."),
        normal("The native ComfyUI Load Image node only lists files directly in the `input` folder. This node walks the entire `input` directory tree so images organised into subfolders appear in the dropdown."),
        normal("Outputs:"),
        normal("- `image`: RGB image tensor", 1),
        normal("- `mask`: Alpha channel as a mask (zeros if no alpha channel present)", 1),
    ],

    "Y7Nodes_CaptionSaver": [
        "Caption Saver",
        short_desc("Save a caption string as a .txt file next to the source image, using the same filename stem."),
        normal("Designed to pair with ImageBatchPath and any VLM node that outputs a STRING: connect `IMAGE_PATH` from ImageBatchPath and the caption `STRING` from the VLM node."),
        normal("Compatible with any node that outputs a STRING. Examples: Florence2, MiniCPM, LLaVA, Qwen-VL, etc."),
        normal("Example: `cat.jpg` → `cat.txt` saved in the same directory."),
        normal("Inputs:"),
        normal("- `string`: The caption text to write (required, must be connected)", 1),
        normal("- `image_path`: Full path to the source image (required, must be connected — e.g. from ImageBatchPath)", 1),
        normal("- `overwrite`: If true, overwrites any existing .txt file. If false, appends a counter to avoid overwriting (e.g. `cat_01.txt`, `cat_02.txt`)", 1),
        normal("This node has no outputs — it is a terminal/output node."),
    ],

    "Y7Nodes_Flux2KleinEdit_Ref1": [
        "Y7 Flux.2 Klein Edit Ref 1",
        short_desc("Loads an image on-node and prepares Klein edit conditioning (reference latent + optional mask-driven inpaint conditioning)."),
        normal("Replaces the usual chain of Load Image → downscale → VAE Encode → mask processing → ReferenceLatent → conditioning patch nodes with a single node. Paint a mask directly on the image (right-click → Open in MaskEditor) to drive inpaint-style editing; leave it unpainted for a plain whole-image edit."),
        normal("Works with all flux.2-klein variants (base, distilled, 4B/8B text encoder), since ComfyUI routes them all through the same Flux2 model class."),
        normal("Inputs:"),
        normal("- `downscale_factor`: Shrinks the image (and mask) before encoding, `0.25`–`1.0`. Lowers VRAM use and latent size on large sources; `1.0` keeps the original resolution", 1),
        normal("- `crop_2_nearest_16px`: Centre-crops the image (and mask) down to the nearest multiple of 16. Flux.2 works best on dimensions that are multiples of 16 and the edit pipeline rounds them down anyway, so this makes what gets encoded match what the model actually sees. No-op if the image is already aligned", 1),
        normal("- `expand_mask`: Dilates the mask outward by this many pixels, so the edit region covers a little more than what was painted. `0` disables", 1),
        normal("- `feather_mask`: Gaussian-blurs the mask edges by this radius for a smoother blend between edited and preserved areas. `0` disables", 1),
        normal("- `binary_mask`: Hard-thresholds the finished mask to pure black/white, cutting at `0.5`. Applied last, after expand and feather, so the result is always crisp with no grey ramp", 1),
        normal("- `positive` (optional): Positive conditioning to patch. Left as an empty list if not connected", 1),
        normal("- `negative` (optional): Negative conditioning, patched the same way as positive", 1),
        normal("Outputs:"),
        normal("- `reference_latent`: VAE-encoded latent of the processed image, for use as the edit model's reference latent", 1),
        normal("- `positive`: Conditioning with `reference_latents` and `concat_latent_image` set (plus `concat_mask` when a mask was painted)", 1),
        normal("- `negative`: Conditioning patched the same way as positive", 1),
        normal("- `preview_image`: The image after downscale/crop, for on-canvas preview", 1),
        normal("- `preview_mask`: The mask after expand/feather/binarize. All-zero if no mask was painted", 1),
        normal("Notes:"),
        normal("- When a mask is present, masked regions are replaced with neutral grey in the concat conditioning, so Klein keeps the surrounding context as reference", 1),
        normal("- `crop_2_nearest_16px` also snaps the `downscale_factor` target to 16, so a downscale followed by a crop doesn't trim twice", 1),
        normal("- Mask processing runs in widget order: expand → feather → binarize. With `binary_mask` on, feathering still shapes the edge (rounding corners, smoothing jagged strokes) but the final cut leaves no soft gradient", 1),
    ],

    "Y7Nodes_Flux2KleinEdit_MultiRef": [
        "Y7 Flux.2 Klein Edit Multi-Ref",
        short_desc("Loads the edit image on-node, takes extra reference images on growable sockets, and prepares Klein edit conditioning (reference latents + optional mask-driven inpaint conditioning)."),
        normal("Same as the single-reference Klein Edit node, but Flux.2 / Klein accepts a list of reference latents, so extra images can be fed in as additional visual context — a character sheet, a style reference, a product shot — alongside the image actually being edited."),
        normal("Paint a mask directly on the on-node image (right-click → Open in MaskEditor) to drive inpaint-style editing; leave it unpainted for a plain whole-image edit."),
        normal("The extra references are IMAGE sockets rather than on-node file pickers because the mask editor is hard-wired to the widget named `image`, so only one on-node picker can ever carry a painted mask. As sockets they can come from anywhere — Load Image, an upscaler, another Y7 node."),
        normal("Prompting with multiple references:"),
        normal("- There is no special syntax. Klein taps the Qwen3-VL language model with the visual tower unused, so the text encoder never sees the reference images at all — they enter as VAE latents appended to the transformer's token sequence. Nothing like `<image1>`, `[Image 1]` or `@ref2` is a real token; refer to the images in plain English by their position", 1),
        normal("- Position is the socket number: the on-node `image` is reference 1, `ref_image_2` is reference 2, `ref_image_3` is reference 3, and so on. The sockets are numbered to match", 1),
        normal("- ComfyUI's own multi-reference Klein template words it as `Figure N`: \"Have the man in Figure 1 put on the clothes from Figure 2, wear a hat, and carry a bag. Then, change the background environment to an African savannah while keeping the man in the same posture...\"", 1),
        normal("- `image 1` / `the first image` is the same kind of plain positional reference and is the wording BFL's own material tends to use. Both forms are just words to the model, so either should work — if a prompt is not binding to the right reference, fix the seed and try the other phrasing", 1),
        normal("- Always pair the index with a noun — `the man in Figure 1`, `the clothes from Figure 2`, `the room from image 3`. A bare index has nothing to latch onto; the noun is what actually anchors the reference", 1),
        normal("- Say what to preserve as well as what to change: \"...preserve her facial identity, hairstyle and proportions from Figure 1\". Reference images are context, not constraints — nothing forces the model to keep them", 1),
        normal("Inputs:"),
        normal("- `image`: The image being edited, picked on the node. This is the only image a mask applies to, and it always leads the reference list", 1),
        normal("- `target_megapixels`: Resamples the image (and mask) to roughly this many megapixels before encoding, same maths as `ImageScaleToTotalPixels`. Applied independently to every reference image too, so each one lands on the budget whether it has to shrink or grow. `0` keeps every image at its original resolution", 1),
        normal("- `crop_2_nearest_16px`: Centre-crops the image (and mask) and every reference image down to the nearest multiple of 16, which Flux.2 prefers. No-op if the dimensions are already aligned", 1),
        normal("- `expand_mask`: Dilates the mask outward by this many pixels. `0` disables", 1),
        normal("- `feather_mask`: Gaussian-blurs the mask edges by this radius. `0` disables", 1),
        normal("- `binary_mask`: Hard-thresholds the finished mask to pure black/white at `0.5`, applied last so the result is crisp", 1),
        normal("- `ref_image_2` … `ref_image_8` (optional): Additional reference images, used as visual context only. One empty socket is shown to start with and a new one appears each time you connect the last, up to eight. These cannot be masked — see the note below", 1),
        normal("- `positive` (optional): Positive conditioning to patch. Left as an empty list if not connected", 1),
        normal("- `negative` (optional): Negative conditioning, patched the same way as positive", 1),
        normal("Outputs:"),
        normal("- `reference_latent`: VAE-encoded latent of the edited image only — the extra references go onto the conditioning, not into this latent", 1),
        normal("- `positive`: Conditioning with `reference_latents` (edited image first, then each reference in socket order) and `concat_latent_image` set, plus `concat_mask` when a mask was painted", 1),
        normal("- `negative`: Conditioning patched the same way as positive", 1),
        normal("- `preview_image`: The edited image after resize/crop, for on-canvas preview", 1),
        normal("- `preview_mask`: The mask after expand/feather/binarize. All-zero if no mask was painted", 1),
        normal("- `ref_count`: How many reference latents ended up on the conditioning, counting the edited image", 1),
        normal("Notes:"),
        normal("- Only the first image can be masked, and only if you want to — masking is optional. The mask belongs to the on-node `image` (the one being edited); the `ref_image_*` sockets take IMAGE only, have no mask input, and any alpha channel on them is dropped before encoding. There is deliberately no way to mask a reference", 1),
        normal("- That is how the model's inpaint conditioning works, not an arbitrary node choice: a mask drives the `concat_latent_image` / `concat_mask` inpaint conditioning, which has to line up pixel-for-pixel with the latent being denoised. The extra references are arbitrary images of arbitrary size, so a mask on one would have nothing to align to", 1),
        normal("- Reference order is edited image first, then `ref_image_2`, `ref_image_3`, … in socket order", 1),
        normal("- A socket carrying a batch of images is split into one reference latent per image, since the model treats each list entry as a separate reference rather than as a batch", 1),
        normal("- Flux.2 Klein is trained around 1.0 MP. Sampling far above that — an 8 MP source at `target_megapixels` 0 — degrades badly at the low step counts the distilled checkpoints use, so leave the default at `1.0` unless you have a reason not to", 1),
        normal("- Every reference image adds tokens to the model's context, so `target_megapixels` is the lever for both quality and VRAM: it caps the big images and, just as importantly, brings undersized references up to a resolution that actually contributes detail", 1),
    ],

    "Y7Nodes_Flux2Sampler": [
        "Flux.2 Sampler",
        short_desc("All-in-one Flux.2 sampler: RandomNoise + KSamplerSelect + Flux2Scheduler + CFGGuider + SamplerCustomAdvanced in a single node."),
        normal("Replaces the five-node chain normally needed to sample a Flux.2 or Klein model. Feed it a model, a starting latent and conditioning, and it returns the denoised latent ready for VAE decode."),
        normal("There are deliberately no `width`/`height` widgets: the Flux.2 sigma schedule is derived from `latent_image`'s own dimensions, so it can never drift out of sync with the latent actually being sampled."),
        normal("Inputs:"),
        normal("- `model`: Diffusion model to sample with", 1),
        normal("- `latent_image`: Starting latent to denoise — e.g. from an Empty Flux.2 Latent Image, or the `reference_latent` output of the Klein Edit node. Its dimensions also set the resolution used for the sigma schedule", 1),
        normal("- `positive`: Positive conditioning (what to steer generation towards)", 1),
        normal("- `negative`: Negative conditioning (what to steer away from). Effectively ignored when `cfg` is `1.0`", 1),
        normal("- `seed`: Seed for the initial noise, with the standard randomize/increment/decrement/fixed control", 1),
        normal("- `cfg`: Classifier-free guidance scale. Klein checkpoints are usually guidance-distilled, so the default `1.0` (negative conditioning skipped) is normally what you want. Raise it only on models that expect real CFG", 1),
        normal("- `sampler_name`: Which k-diffusion sampler algorithm to step with. `euler` is the default", 1),
        normal("- `steps`: Number of sampling steps. Distilled Klein checkpoints commonly need only ~4; non-distilled Flux.2 wants considerably more", 1),
        normal("Outputs:"),
        normal("- `output`: The denoised latent, ready for VAE decode", 1),
        normal("Notes:"),
        normal("- Any `noise_mask` carried on the incoming latent (as set by the Klein Edit node when a mask is painted) is honoured, so inpaint-style edits work without extra wiring", 1),
        normal("- The Flux.2 sigma schedule math is vendored from `Flux2Scheduler`, so this node doesn't depend on ComfyUI's internal `comfy_extras` module", 1),
    ],

    # Add more node descriptions here
}

def as_html(entry, depth=0):
    """Convert structured documentation into HTML with collapsible sections"""
    if isinstance(entry, dict):
        size = 0.8 if depth < 2 else 1
        html = ''
        for k in entry:
            if k == "collapsed":
                continue
            collapse_single = k.endswith("_collapsed")
            if collapse_single:
                name = k[:-len("_collapsed")]
            else:
                name = k
            if collapse_single:
                name = k[:-len("_collapsed")]
            else:
                name = k
            collapse_flag = ' Y7Nodes_precollapse' if entry.get("collapsed", False) or collapse_single else ''
            html += f'<div Y7Nodes_title=\"{name}\" style=\"display: flex; font-size: {size}em\" class=\"Y7Nodes_collapse{collapse_flag}\"><div style=\"color: #AAA; height: 1.5em;\">[<span style=\"font-family: monospace\">-</span>]</div><div style=\"width: 100%\">{name}: {as_html(entry[k], depth=depth+1)}</div></div>'
        return html
    if isinstance(entry, list):
        if depth == 0:
            depth += 1
            size = .8
        else:
            size = 1
        html = ''
        html += entry[0]
        for i in entry[1:]:
            html += f'<div style=\"font-size: {size}em\">{as_html(i, depth=depth)}</div>'
        return html
    return str(entry)

def _apply_v3_description(node_cls, html):
    """Attach documentation HTML to a V3 (io.ComfyNode) node class.

    V3 nodes don't serve the DESCRIPTION class attribute: ComfyUI builds their node info from a
    fresh define_schema() call (Schema.get_v1_info uses schema.description), and DESCRIPTION itself
    is a classproperty backed by _DESCRIPTION. Assigning DESCRIPTION on such a class silently
    shadows the classproperty and is then ignored by the frontend, so patch the schema on its way
    out of define_schema instead.
    """
    node_cls._y7_doc_html = html
    node_cls._DESCRIPTION = html  # so cls.DESCRIPTION reads back the docs too

    if "_y7_doc_patched" in node_cls.__dict__:
        return  # already wrapped; the refreshed _y7_doc_html above is all that's needed

    original_define_schema = node_cls.define_schema

    def define_schema(cls):
        schema = original_define_schema()
        schema.description = cls._y7_doc_html
        return schema

    node_cls.define_schema = classmethod(define_schema)
    node_cls._y7_doc_patched = True


def format_descriptions(nodes):
    """Applies HTML documentation to node classes"""
    logger.info(f"Formatting descriptions for nodes: {list(nodes.keys())}")
    logger.info(f"Available descriptions: {list(descriptions.keys())}")
    
    for k in descriptions:
        if k in nodes:
            logger.info(f"Setting DESCRIPTION for {k}")
            html = as_html(descriptions[k])
            if hasattr(nodes[k], "define_schema"):
                _apply_v3_description(nodes[k], html)
            else:
                nodes[k].DESCRIPTION = html
                # Also set a direct description property for easier access
                nodes[k].description = html
        else:
            logger.warning(f"Node {k} has a description but is not in the nodes dictionary")
    
    # Optionally, log any undocumented nodes
    undocumented_nodes = []
    for k in nodes:
        if k.startswith("Y7_") and not hasattr(nodes[k], "DESCRIPTION"):
            undocumented_nodes.append(k)
    
    if len(undocumented_nodes) > 0:
        logger.warning(f"Some nodes have not been documented: {undocumented_nodes}")
    
    # Return the number of descriptions applied for confirmation
    return len([k for k in descriptions if k in nodes])
