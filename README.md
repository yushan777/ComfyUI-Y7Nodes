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

### Y7 Aspect Ratio Picker

> Interactive 2D canvas for picking image width and height by dragging.
>
> <img src="assets/aspect_ratio_picker.jpg" alt="aspect ratio picker" width="50%"/>
> <details>
>   <summary>ℹ️ <i>See More Information</i></summary>
>
>   - Click or drag anywhere in the canvas to set width (X axis, left→right) and height (Y axis, bottom→top).
>   - A filled rectangle shows the selected proportions; the dot marks the current position.
>   - The simplified aspect ratio (e.g. 16:9, 4:3, 1:1) is displayed at the bottom of the canvas.
>   - Current width and height values are shown in the right panel. Double-click either value to type a number directly.
>   - Click the **Swap** button (below the height value) to swap width and height (portrait ↔ landscape).
>   - Snapping to step increments is on by default. Hold Shift while dragging to temporarily disable snapping.
>   - Node properties (right-click → Properties): `valueX`, `valueY`, `minX`, `maxX`, `minY`, `maxY`, `stepX`, `stepY`, `snap`, `dots`.
>   - The right panel shows width, height, simplified ratio (e.g. 16:9), and total megapixels (e.g. 1.0MP).
>   - Outputs `width` and `height` as INT.
>
> </details>

---

### Y7 Show Anything

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

### Y7 Text

> A plain multiline text box that passes its contents straight through to a `STRING` output. Useful as a shared prompt source that several nodes can read from.

---

### Y7 Image Stitcher

> Stitches 2–8 images side-by-side or top-to-bottom. The `image_count` widget controls how many image sockets are shown on the node.
>
> <details>
>   <summary>ℹ️ <i>See More Information</i></summary>
>
>   - Set `image_count` (2–8) to show exactly that many image input sockets. Sockets are added or removed live as you change the value.
>   - **Side-by-Side (Horizontal)**: all images are resized to match the first image's height, then concatenated left-to-right.
>   - **Top-and-Bottom (Vertical)**: all images are resized to match the first image's width, then concatenated top-to-bottom.
>   - Unconnected sockets within the count are skipped gracefully.
>   - Output is a single `IMAGE` tensor.
>
> </details>

---

### Y7 Image Compare

> Compares two images with a draggable slider for interactive side-by-side comparison directly on the node.
>
> <details>
>   <summary>ℹ️ <i>See More Information</i></summary>
>
>   - Connect `image_a` (required) and `image_b` (optional), then drag the slider to reveal `image_a` over `image_b`.
>   - Two blend modes: `normal` (slider wipe) and `difference` (highlights variations between the two images).
>   - The preview updates live as the slider is moved or the blend mode is changed, and persists across workflow-tab switches.
>   - The node auto-resizes to match the aspect ratio of the input images; slider position and blend mode are saved with the workflow.
>   - Right-click over the image for `Open Image` / `Save Image` options.
>   - Preview-only node with no outputs.
>   - Based on `Eses Image Compare` by Eses Nodes.
>
> </details>

---

### Y7 CLIP Token Counter

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

### Y7 T5 Token Counter

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

### Y7 Prompt Enhancer (Native)

> Takes any basic prompt and enhances it using a text encoder that ComfyUI has already loaded, via a `CLIP` input. Downloads nothing, loads nothing, and leaves all VRAM management to ComfyUI. Reasoning is separated from the prompt automatically.
>
> <details>
>   <summary>ℹ️ <i>See More Information</i></summary>
>
>   Where a typical prompt enhancer node owns its model (downloading it, loading it through HuggingFace transformers, and managing when to free it), this node does none of that. You load a text encoder with a standard `CLIPLoader` and connect it. ComfyUI handles loading, device placement and offloading exactly as it does for any other model in your workflow.
>
>   **Supported models**
>
>   Any text encoder with a native generation path:
>
>   | Model | Variants |
>   | --- | --- |
>   | Gemma 4 | E2B, E4B, 31B, 12B unified |
>   | Gemma 3 | 12B (including the LTX-2 text encoder built on it) |
>   | Qwen3 | 0.6B, 2B, 4B, 8B |
>   | Qwen3.5 | 0.8B, 2B, 4B, 9B, 27B |
>   | Qwen3-VL | 4B, 8B |
>
>   **Not supported** — T5 (all sizes), UMT5, CLIP-L, CLIP-G, Gemma 2, LLaMA-3.1. These have no text generation path; the node detects this and tells you, rather than failing with an obscure error deep inside the encoder.
>
>   **Model format**
>
>   Safetensors only, placed in `models/text_encoders/`. GGUF will **not** work — ComfyUI core has no GGUF loader (`.gguf` is not even a recognised model extension), and for Gemma 4 the tokenizer is embedded inside the safetensors file, so a converted GGUF would be missing it.
>
>   For Gemma 4 the `CLIPLoader` `type` dropdown is ignored — the model is detected from the weights themselves, so any value in that dropdown works.
>
>   Gemma 4 weights: https://huggingface.co/Comfy-Org/gemma-4/tree/main/text_encoders
>
>   ```
>   ComfyUI
>   └── models
>       └── text_encoders
>           └── gemma4_e4b_it_bf16.safetensors
>   ```
>
>   **Inputs**
>
>   - `clip`: A generation-capable text encoder from `CLIPLoader`
>   - `text`: Your basic prompt. Accepts a connection from any string node.
>   - `instruction`: The instruction placed before your text. Edit to change the enhancement style.
>   - `max_length`: Maximum **new** tokens to generate (64–32768, default 2048). Not the context window. Reasoning is spent from the same budget, and the KV cache reserves roughly 84KB of VRAM per token up front.
>   - `temperature`: 0.0–2.0, default 1.0 (Google's recommended value for Gemma). 0 switches to greedy decoding.
>   - `top_k`: 0–1000, default 64 (Google's recommended value). 0 disables.
>   - `top_p`: 0.0–1.0, default 0.95 (Google's recommended value). 1.0 disables.
>   - `seed`: Change it to re-roll — identical inputs return a cached result.
>   - `thinking`: Let the model reason before answering.
>
>   **Outputs**
>
>   - `thinking_output`: The model's reasoning, if it produced any
>   - `enhanced_prompt`: The enhanced prompt, with all reasoning removed
>
>   **On reasoning**
>
>   Gemma 4 writes its reasoning into a thought channel, and ComfyUI's decoder deliberately preserves that text instead of discarding it. Turning `thinking` off only *primes* the model to skip reasoning — Gemma 4 often reasons anyway, which is why ComfyUI's own **Generate Text** node can hand back a wall of planning notes with the actual prompt buried at the end.
>
>   This node always separates the two, including the awkward case where the model reasons straight past the primed channel and closes it with an unmatched tag. If `enhanced_prompt` comes back empty, the model spent the entire `max_length` budget reasoning — raise it, or lower `temperature`.
>
> </details>

------

### Y7 Image Batch Path

> Load a batch of images from a directory and output them as a list of image tensors with matching file paths. Designed to pair with Caption Saver and a VLM node for batch captioning workflows.
>
> <details>
>   <summary>ℹ️ <i>See More Information</i></summary>
>
>   Supports jpg, jpeg, png, and webp. Images are EXIF-transposed and converted to RGB float32 tensors.
>
>   Connect `IMAGE` to a VLM node and `IMAGE_PATH` to Caption Saver. The path list tells Caption Saver exactly where to write each `.txt` file.
>
>   **Inputs:**
>
>   - `image_dir`: Path to the directory containing images
>   - `batch_size`: Number of images to load (0 = all)
>   - `start_from`: 1-based index of the first image to load — useful for resuming part-way through a directory
>   - `sort_method`: Load order — `sequential` (alphabetical), `reverse`, or `random`
>
>   **Outputs** (both are lists):
>
>   - `IMAGE`: List of image tensors, one per file
>   - `IMAGE_PATH`: List of full file paths matching each image tensor
>
>   Note: When `sort_method` is `random`, the node re-evaluates on every run.
>
> </details>

---

### Y7 Caption Saver

> Save a caption string as a `.txt` file next to the source image, using the same filename stem (e.g. `cat.jpg` → `cat.txt`).
>
> <details>
>   <summary>ℹ️ <i>See More Information</i></summary>
>
>   Designed to pair with **Image Batch Path** and a VLM node: connect `IMAGE_PATH` from Image Batch Path and the caption `STRING` from the VLM node.
>
>   Compatible with any node that outputs a STRING. Examples: Florence2, MiniCPM, LLaVA, Qwen-VL, etc.
>
>   **Inputs:**
>
>   - `string`: The caption text to write (must be connected)
>   - `image_path`: Full path to the source image (must be connected — e.g. from Image Batch Path)
>   - `overwrite`: If true, overwrites any existing `.txt` file. If false, appends a counter to avoid overwriting (e.g. `cat_01.txt`, `cat_02.txt`)
>
>   This node has no outputs — it is a terminal/output node.
>
> </details>

---

### Y7 Image Size (Presets)
> Select predefined image size/aspect ratios from a named preset set. Provides width and height outputs.
>
> <details>
>   <summary>ℹ️ <i>See More Information</i></summary>
>
>   - Provides outputs for `width` and `height` (INT).
>   - The `preset` widget selects the active dimension set: `default`, `flux.2`, `qwen-image`, or `custom*`.
>   - The `dimension` dropdown updates dynamically to show only the dimensions for the selected preset.
>   - Select `Custom` from the dimension dropdown to use manually entered `custom_w` / `custom_h` values.
>   - The `custom*` preset loads from `custom_dimensions.json` in the `nodes` directory; falls back to `default` if the file is missing or invalid. See `custom_dimensions_example.json` for the expected format.
> </details>

---

### Y7 Resolution Selector

> Calculate width and height from an aspect ratio and a megapixel target, rounded to the nearest multiple.
>
> <details>
>   <summary>ℹ️ <i>See More Information</i></summary>
>
>   A clone of ComfyUI's built-in V3 `ResolutionSelector` node, with a finer-grained `megapixels` step (0.01 instead of 0.1) and the resolution displayed directly on the node after execution.
>
>   **Inputs:**
>
>   - `aspect_ratio`: The target aspect ratio (e.g. `1:1 (Square)`, `16:9 (Widescreen)`)
>   - `megapixels`: Target total megapixels (default: 1.0, range: 0.1–16.0)
>   - `multiple`: Rounds the result to the nearest multiple of this value (default: 8)
>
>   **Outputs:**
>
>   - `width`: Calculated width in pixels (INT)
>   - `height`: Calculated height in pixels (INT)
>
> </details>

---

### Y7 Scale Image By

> Scales an image by a multiplier while preserving aspect ratio.
>
> <details>
>   <summary>ℹ️ <i>See More Information</i></summary>
>
>   Multiplies both width and height by `scale_by`, then resamples using the chosen method. The output resolution is displayed directly on the node after execution.
>
>   **Inputs:**
>
>   - `image`: The input image to scale
>   - `upscale_method`: Resampling algorithm — `nearest-exact`, `bilinear`, `area`, `bicubic`, or `lanczos`
>   - `scale_by`: Multiplier applied to both dimensions (default: 1.0, range: 0.01–8.0)
>   - `resolution_steps`: Snap output dimensions to the nearest multiple of this value (default: 8). Common values: 8, 16, 64
>
>   **Outputs:**
>
>   - `image`: The scaled image
>   - `width`: Output width in pixels (INT)
>   - `height`: Output height in pixels (INT)
>
> </details>

---

### Y7 Scale Image to Total Pixels

> Scales an image to a target total pixel count (in megapixels) while preserving aspect ratio.
>
> <details>
>   <summary>ℹ️ <i>See More Information</i></summary>
>
>   Computes a uniform scale factor so that `width × height` equals the target megapixel count, then resamples using the chosen method. The output resolution is displayed directly on the node after execution.
>
>   **Inputs:**
>
>   - `image`: The input image to scale
>   - `upscale_method`: Resampling algorithm — `nearest-exact`, `bilinear`, `area`, `bicubic`, or `lanczos`
>   - `megapixels`: Target total pixel count in megapixels (default: 1.0, range: 0.01–16.0)
>   - `resolution_steps`: Snap output dimensions to the nearest multiple of this value (default: 8). Common values: 8, 16, 64
>
>   **Outputs:**
>
>   - `image`: The scaled image
>   - `width`: Output width in pixels (INT)
>   - `height`: Output height in pixels (INT)
>
> </details>

---

### Y7 Crop to Nearest Multiple
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
>   - `h_crop`: Where to keep content when width needs adjustment - `center`, `left`, `right`, or `none`
>   - `v_crop`: Where to keep content when height needs adjustment - `center`, `top`, `bottom`, or `none`
>   
>   **Outputs:**
>   
>   - `crop_preview`: Original image with red overlay showing what will be cropped (useful for previewing before committing)
>   - `cropped_image`: The cropped result (or original if no cropping needed)
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

### Y7 Color Match (Masked)

> Colour matches a target image to a reference while excluding masked regions from the calculation on **both** images.
>
> <details>
>   <summary>ℹ️ <i>See More Information</i></summary>
>
>   The use case is inpainting. Change a red car to blue and the rest of the image often picks up a colour shift, but a plain colour-match node will read the new blue car as part of the picture and skew the correction. This node calculates the transfer from the non-masked areas only (the background), applies it to those same areas, and leaves the inpainted region untouched.
>
>   **Inputs:**
>
>   - `image_ref`: Reference image — e.g. the original, before inpainting
>   - `image_target`: Image to correct — e.g. the result after inpainting
>   - `method`: Colour transfer method — `mkl`, `hm`, `reinhard`, `mvgd`, `hm-mvgd-hm`, or `hm-mkl-hm`
>   - `mask` (optional): White (1.0) marks the region to **exclude** from matching. With no mask connected, the whole image is matched
>   - `strength` (optional): Blend between the original and the corrected result (0.0–1.0, default 1.0)
>   - `feather` (optional): Blur radius in pixels for the mask edge transition (0–100, default 0)
>
>   **Outputs:**
>
>   - `image`: The colour-corrected image
>
>   Requires the `color-matcher` package (included in `requirements.txt`). Based on the ColorMatch node from kijai's ComfyUI-KJNodes.
>
> </details>

---

### Y7 Paste Cropped Image Back

> <img src="assets/paste_cropped_image_back.jpg" alt="paste cropped image back" width="100%"/>
>
> Paste a cropped image back onto a base image post-editing at a region defined by edge-relative coordinates. Varient of the WAS equivalent, except `right` and `bottom` are offsets measured inward from the right and bottom edges.
>
> <details>
>   <summary>ℹ️ <i>See More Information</i></summary>
>
>   Sometimes you may want to change or refine a specific area of an image without affecting the rest too much — for example, fixing a face, hand, or background detail after generation. A typical workflow is to crop the region, run it through img2img or inpainting, then paste the result back using this node.
>
>   Works well with the **OLM Drag Crop** custom node, which lets you visually drag-select a crop region and outputs the crop coordinates directly — those coordinates can be wired into this node's `top`, `left`, `right`, and `bottom` inputs.
>
>   The paste region is computed as:
>   - `x1 = left`
>   - `y1 = top`
>   - `x2 = image_width - right`
>   - `y2 = image_height - bottom`
>
>   This makes it easy to target edge-relative regions without knowing the image dimensions in advance — e.g., set `right=256, bottom=256` to always paste into the bottom-right 256-wide strip of any image.
>
>   The `image_crop` is always resized to exactly fit the paste region. If it was upscaled for editing (e.g. sent through img2img at a higher resolution), it will be scaled back down during pasting. There is no aspect-ratio preservation — if the aspect ratio of the crop image differs from the paste region, it will be stretched to fit and appear distorted.
>
>   **Inputs:**
>
>   - `image_orig`: Base image to paste onto
>   - `image_crop`: Image to paste into the defined region (resized to fit)
>   - `left` / `top`: Pixel offsets from the left and top edges
>   - `right` / `bottom`: Pixel offsets inward from the right and bottom edges
>   - `crop_blending`: Feathering/blending amount at paste edges (0.0–1.0)
>   - `crop_sharpening`: Number of sharpening passes applied before pasting (0–3)
>
>   **Outputs:**
>
>   - `IMAGE`: Base image with crop pasted in
>   - `MASK`: The blended mask used for the paste operation
>
> </details>

---

### Y7 Pad Image for Outpainting

> Pads an image ready for outpainting, with the padding snapped to a chosen step and a choice of what to fill the new area with.
>
> <details>
>   <summary>ℹ️ <i>See More Information</i></summary>
>
>   A rework of ComfyUI's built-in `ImagePadForOutpaint`. Three differences: padding is rounded to a multiple of `step` so the padded canvas stays a size the model is happy with, the new area can be filled with something more useful than flat grey, and the feathering is vectorised rather than looping over every pixel in Python.
>
>   The fill matters more than it sounds like it should — it all gets painted over during the outpaint, but a plausible starting colour tends to give a better result than flat grey.
>
>   **Inputs:**
>
>   - `image`: The image to pad
>   - `step`: Padding is rounded to a multiple of this value (default 16). Also sets how far the +/- arrows move each side
>   - `fill`: What to put in the new area — `grey` (flat 50%, same as the built-in node), `edge replicate` (smears the outermost pixels outwards), `mirror` (reflects the image back on itself), `blurred edge` (a soft blur of the nearby colours), or `noise` (random noise in the image's own colours)
>   - `feathering`: Width in pixels of the soft fade at the join between the original image and the new area (default 40; 0 gives a hard edge)
>   - `left` / `top` / `right` / `bottom`: Pixels to add to each edge
>
>   **Outputs:**
>
>   - `image (original)`: The image exactly as it came in, unpadded
>   - `image (padded)`: The image on the enlarged canvas, with the new area filled
>   - `mask`: White over the new area, black over the original image, feathered ramp in between
>   - `width` / `height`: Dimensions of the padded image (INT)
>
> </details>

---

### Sampler Select (Name)

> Select a sampler by name and output it as a linkable string — works around ComfyUI's built-in KSampler nodes not exposing sampler name as a connectable input.

---

### Y7 Flux.2 Klein Edit Multi-Ref

> All-in-one image editing for Flux.2 Klein with multiple reference images. Pick the image to edit on the node, mask it, wire in extra references, and get back a ready-to-sample reference latent plus patched positive/negative conditioning.
>
> <details>
>   <summary>ℹ️ <i>See More Information</i></summary>
>
>   Flux.2 / Klein accepts a *list* of reference latents on the conditioning, so extra images can be wired in as additional visual context — a character sheet, a style reference, a product shot — alongside the image actually being edited. The list is ordered: the on-node `image` is reference 1, then `ref_image_2`, `ref_image_3` and so on in socket order. The extra sockets grow as you connect them, up to `ref_image_8`.
>
>   **Masking** applies to the on-node image and nothing else, whichever route it arrives by — painted in the mask editor, or wired into `external_mask`. The extra references are IMAGE-only and have no mask of their own; a mask ends up as the latent's `noise_mask`, which has to line up cell-for-cell with the latent being denoised, and an arbitrary reference image at an arbitrary size has nothing to align to.
>
>   Klein has no mask input of its own and is not a fill/inpaint model, so the mask survives only as that `noise_mask`: the sampler restores everything outside the painted area after each step, in latent space at 1/16 resolution. Two consequences that pull in opposite directions:
>
>   - For sampling, feathering is close to a no-op — 4px is bit-identical to no feather once the mask is downsampled 16x, and only ~32px puts real intermediate values into more than one latent cell (where it crossfades partially-denoised latents into a smeared seam rather than a blend).
>   - For a pixel-space composite downstream (`ImageCompositeMasked`, or **Y7 Paste Cropped Image Back**), the same feather survives at full resolution, where 4–8px is the difference between a hard cut and a soft join. That is why `binary_mask` defaults to off — it would flatten the ramp before either consumer sees it.
>
>   Leave the sampler's `denoise` at 1.0. The unmasked area is restored every step regardless, so lowering it does nothing but weaken the edit.
>
>   **Prompting:** there is no special syntax for addressing the references. The text encoder never sees the images — they reach the transformer only as latent tokens appended to the sequence, in list order. Refer to them in plain English by position, always paired with a noun: *"Have the man in Figure 1 put on the clothes from Figure 2, and change the background to a savannah"*. `Figure N` is the wording in ComfyUI's own multi-reference Klein template; `image 1` / `the first image` is the same kind of positional reference. Neither is a real token, so both are worth trying if a prompt is not binding to the reference you meant.
>
>   **Inputs:**
>
>   - `vae`: Used to encode the edited image and every reference
>   - `image`: The image to edit, picked from the ComfyUI input folder (with upload button and preview). Right-click → Open in MaskEditor to paint a mask
>   - `target_megapixels`: Resamples the edited image to this pixel budget before encoding (default 1.0; 0 leaves it alone). This is the latent actually being denoised, so it also sets the output resolution. Klein is built for around 1.0
>   - `ref_megapixels`: The same budget, applied to each reference separately (default 1.0; 0 leaves them alone). It does *not* affect output size. One image plus six references at 1.0 is seven megapixels of work — turn this down first (try 0.5) if you run slow or run out of VRAM
>   - `crop_2_nearest_16px`: Centre-crops the edited image, its mask and every reference down to a multiple of 16 before encoding (default on)
>   - `expand_mask`: Grows the painted mask outwards by this many pixels (default 16 — one full latent cell, which compensates for a tightly painted edge being clipped inward by the downsample). Try 24–32 for hair, fur or soft edges
>   - `feather_mask`: Gaussian softening of the mask edge (default 8). See the note above on where this actually matters
>   - `binary_mask`: Applied last — thresholds the mask to hard edges (default off)
>   - `external_mask` (optional): A mask from anywhere else, used instead of a painted one. Note: with Load Image (as Mask) on a plain black-and-white file, set its channel to `red` — on `alpha` it hands back an empty mask
>   - `ref_image_2` … `ref_image_8` (optional): Extra reference images. Connect one and another empty socket appears
>   - `positive` / `negative` (optional): Conditioning to extend with the reference latents
>
>   **Outputs:**
>
>   - `reference_latent`: The encoded (and resampled) edited image, ready to sample
>   - `positive` / `negative`: Conditioning with the reference latents attached
>   - `preview_image`: The edited image as it was actually encoded
>   - `preview_mask`: The processed mask, after expand/feather/binary
>   - `debug`: What the node actually did — sizes in and out, mask state, every reference socket, and the token cost of the whole reference list. Wire it into **Y7 Show Anything** when something looks wrong
>
>   Works with all flux.2-klein variants (base, distilled, 4B/8B text encoder).
>
> </details>

---

### Y7 Flux.2 Sampler

> All-in-one Flux.2 sampler: `RandomNoise`, `KSamplerSelect`, `Flux2Scheduler`, `CFGGuider` and `SamplerCustomAdvanced` compacted into a single node.
>
> <details>
>   <summary>ℹ️ <i>See More Information</i></summary>
>
>   Sampling a Flux.2 (including Klein) model normally means wiring five nodes together. This node takes one input side — model, positive, negative, latent — and gives back the denoised latent, ready for VAE decode.
>
>   There are no width/height widgets. `Flux2Scheduler`'s are dropped because the sigma schedule's `seq_len` is derived from `latent_image`'s own dimensions instead, so it cannot drift out of sync with the latent being sampled.
>
>   **Inputs:**
>
>   - `model`: The diffusion model to sample with
>   - `latent_image`: The latent to denoise — e.g. from Empty Flux.2 Latent Image, or from the Klein Edit node
>   - `positive` / `negative`: Conditioning. The negative is ignored by the guider when `cfg` is 1.0
>   - `seed`: Seed for the initial noise
>   - `cfg`: Guidance scale (default 1.0 — Klein checkpoints are typically guidance-distilled)
>   - `sampler_name`: Which sampler algorithm to step with (default `euler`)
>   - `steps`: Sampling steps (default 4 — distilled Klein checkpoints commonly need no more)
>   - `denoise`: How much of the schedule to run (default 1.0). It shortens the run as well as the noise level, so at 4 steps only a handful of settings are reachable, and Flux.2's schedule means even 0.5 keeps under 10% of the incoming image. Leave it at 1.0 unless you specifically want this
>
>   **Outputs:**
>
>   - `output`: The denoised latent
>
> </details>

---

### Y7 Flux.2 Klein Upscaler (Tiled)

> Enlarges an image and repaints it one tile at a time with Flux.2 Klein, so you can go far past the size the model or the card could manage in one pass. The sampler, scheduler and guider are built in.
>
> <details>
>   <summary>ℹ️ <i>See More Information</i></summary>
>
>   Each tile is treated as a small inpaint: the middle is regenerated while a band of surrounding pixels is held frozen as context, so neighbouring tiles agree with each other instead of each inventing its own idea of what is there. Nothing larger than a single tile is ever encoded or decoded, which is what keeps VRAM flat as the canvas grows — `tile_size`, not the finished size, is what determines peak memory.
>
>   Ported from [ComfyUI_KleinTiledUpscaler](https://github.com/Gavr728/ComfyUI_KleinTiledUpscaler) by Gavr728 (MIT). The frozen-padding anchoring, the separable blend profiles and the shared canvas noise field are that project's ideas. What is done differently here:
>
>   - **All-in-one.** Upstream needs CFGGuider + KSamplerSelect + Flux2Scheduler (and usually SplitSigmasDenoise) wired in front of it. This node builds the guider, the sampler and the schedule itself, and gives the negative conditioning the same per-tile treatment as the positive.
>   - **Per-tile sigma schedule.** Flux.2's schedule shift is a function of how many tokens are being sampled, and a tile is not the whole canvas. Every tile gets a schedule built from its own latent size.
>   - **Partition-of-unity blending** instead of sequential lerp plus a `mask_blur` widget. Tile weights are complementary raised-cosine ramps centred on each shared boundary, summing to exactly 1.0, so seams are correct by construction and there is nothing to tune.
>   - **No full-canvas VAE encode**, which is the exact thing tiling exists to avoid.
>   - **Tile layout rewritten** — the grid is derived once, the remainder spread evenly across tiles, and the result printed in the report.
>   - **Per-tile sampler seed**, so ancestral and SDE samplers do not repeat the same texture across the picture.
>
>   **Inputs:**
>
>   - `model` / `vae`: The Klein model and the VAE used for every tile encode/decode
>   - `image`: The picture to enlarge. A batch is processed one image at a time
>   - `positive`: Keep it a short description of the picture as a whole, or leave it empty — it is applied to every tile, so anything naming a specific object will try to put that object in all of them
>   - `negative`: Ignored unless `cfg` is above 1.0
>   - `seed`
>   - `scale_factor`: How much bigger to make the picture (default 2.0). **1.0 is a useful setting** — no enlargement, but every tile is still repainted, which turns the node into a sharpen/refine pass. With 1.0, leave `upscale_model` unplugged and use `refine_strength` (0.85–0.92) to control how much changes
>   - `tile_size`: Roughly how big each tile is (default 1024). This is what decides VRAM. Turn it down if you run out of memory, up if tiles are inventing detail that does not belong
>   - `overlap`: How far each tile looks past its own edges (default 128). Raising it helps things that run across a seam — a roofline, a horizon — stay joined up
>   - `steps`: Per tile (default 4)
>   - `cfg`: Leave at 1.0 for distilled checkpoints. Above 1.0 every tile costs twice as much
>   - `sampler_name`
>   - `denoise`: How much each tile may change (default 1.0, which is normally what you want here)
>   - `color_match`: Nudges each finished tile back towards the colours of the enlarged original so tiles do not drift apart in tone (default `mkl`; `off` disables)
>   - `refine_strength` *(advanced)*: Restrains the model at every sampler step rather than fading the finished tile afterwards, which makes it a better detail dial than `denoise`. 1.0 repaints completely; 0.92 is a good first try; 0.85 is strongly held back
>   - `reference_source` *(advanced)*: What each tile is told it is looking at — `enlarged` (always the plain enlargement, predictable) or `progressive` (the tile's surroundings as they stand, including finished neighbours; more consistent, but mistakes can compound). Try `enlarged` first
>   - `upscale_model` *(optional)*: An ESRGAN-style upscaler to enlarge with before the tiles are repainted; plain bicubic is used without one. Pick a clean, neutral model rather than a sharp one — the enlargement becomes the frozen context and the reference every tile works from, so halos and invented texture get copied rather than fixed. `4xRealWebPhoto_v4_dat2`, `4xNomosUniDAT_otf` and `RealESRGAN_x4plus` are good; anything with "Sharp" in the name is not. Its result is always resized to exactly `scale_factor`, so a 4x model at scale_factor 2 is fine (better, even), but a 2x model at scale_factor 4 leaves half the enlargement to bicubic
>
>   **Outputs:**
>
>   - `image`: The finished picture
>   - `latent`: The same picture in latent form, blended with identical geometry, so it can be fed onward without a re-encode
>   - `report`: Canvas size, the grid chosen, tile and crop sizes, and a check that the blend weights really do sum to 1
>
> </details>

---

### Y7 Load Image (subfolders)

> The native ComfyUI Load Image node only lists files directly in the `input` folder. This node is identical except it walks the full `input` directory tree, so images organised into subdirectories appear in the dropdown.
>
> Outputs `image` (IMAGE) and `mask` (MASK), same as the built-in node.

---

## Example Workflows

Example workflows can be found in the `example_workflows` directory. 

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Credits

- ShowAnything node is based on "Show Any" from yolain's ComfyUI-Easy-Use custom nodes and "Show Any To JSON" from crystian's ComfyUI-Crystools custom nodes, with additional formatting controls and a Copy Text button.
- Help popup system is based on the implementation from Kosinkadink's ComfyUI-VideoHelperSuite.
- Color Match (Masked) is based on the ColorMatch node from kijai's ComfyUI-KJNodes, using the [color-matcher](https://github.com/hahnec/color-matcher) library.
- Flux.2 Klein Upscaler (Tiled) is ported from [ComfyUI_KleinTiledUpscaler](https://github.com/Gavr728/ComfyUI_KleinTiledUpscaler) by Gavr728 (MIT).
