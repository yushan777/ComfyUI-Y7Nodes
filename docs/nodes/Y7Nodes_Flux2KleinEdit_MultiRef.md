# Y7 Flux.2 Klein Edit Multi-Ref

Loads the edit image on-node, and takes extra reference images on growable list of sockets,so extra images can be fed in as additional visual context — a character sheet, a style reference, a product shot — alongside the image actually being edited. and prepares Klein edit conditioning (reference latents + optional mask-driven inpaint conditioning).

Paint a mask directly on the on-node image (right-click → Open in MaskEditor) to drive inpaint-style editing; leave it unpainted for a plain whole-image edit.

The extra references are IMAGE sockets rather than on-node file pickers because the mask editor is hard-wired to the widget named `image`, so only one on-node picker can ever carry a painted mask. As sockets they can come from anywhere — Load Image, an upscaler, another Y7 node.

Prompting with multiple references:

  - There is no special syntax. Klein taps the Qwen3-VL language model with the visual tower unused, so the text encoder never sees the reference images at all — they enter as VAE latents appended to the transformer's token sequence. Nothing like `<image1>`, `[Image 1]` or `@ref2` is a real token; refer to the images in plain English by their position
  - Position is the socket number: the on-node `image` is reference 1, `ref_image_2` is reference 2, `ref_image_3` is reference 3, and so on. The sockets are numbered to match
  - ComfyUI's own multi-reference Klein template words it as `Figure N`: "Have the man in Figure 1 put on the clothes from Figure 2, wear a hat, and carry a bag. Then, change the background environment to an African savannah while keeping the man in the same posture..."
  - `image 1` / `the first image` is the same kind of plain positional reference and is the wording BFL's own material tends to use. Both forms are just words to the model, so either should work — if a prompt is not binding to the right reference, fix the seed and try the other phrasing
  - Always pair the index with a noun — `the man in Figure 1`, `the clothes from Figure 2`, `the room from image 3`. A bare index has nothing to latch onto; the noun is what actually anchors the reference
  - Say what to preserve as well as what to change: "...preserve her facial identity, hairstyle and proportions from Figure 1". Reference images are context, not constraints — nothing forces the model to keep them

Inputs:

  - `image`: The image being edited, picked on the node. This is the only image a mask applies to, and it always leads the reference list
  - `target_megapixels`: Resamples the image (and mask) to roughly this many megapixels before encoding, same maths as `ImageScaleToTotalPixels`. Applied independently to every reference image too, so each one lands on the budget whether it has to shrink or grow. `0` keeps every image at its original resolution
  - `crop_2_nearest_16px`: Centre-crops the image (and mask) and every reference image down to the nearest multiple of 16, which Flux.2 prefers. No-op if the dimensions are already aligned
  - `expand_mask`: Dilates the mask outward by this many pixels. `0` disables
  - `feather_mask`: Gaussian-blurs the mask edges by this radius. `0` disables
  - `binary_mask`: Hard-thresholds the finished mask to pure black/white at `0.5`, applied last so the result is crisp
  - `ref_image_2` … `ref_image_8` (optional): Additional reference images, used as visual context only. One empty socket is shown to start with and a new one appears each time you connect the last, up to eight. These cannot be masked — see the note below
  - `positive` (optional): Positive conditioning to patch. Left as an empty list if not connected
  - `negative` (optional): Negative conditioning, patched the same way as positive

Outputs:

  - `reference_latent`: VAE-encoded latent of the edited image only — the extra references go onto the conditioning, not into this latent
  - `positive`: Conditioning with `reference_latents` (edited image first, then each reference in socket order) and `concat_latent_image` set, plus `concat_mask` when a mask was painted
  - `negative`: Conditioning patched the same way as positive
  - `preview_image`: The edited image after resize/crop, for on-canvas preview
  - `preview_mask`: The mask after expand/feather/binarize. All-zero if no mask was painted
  - `ref_count`: How many reference latents ended up on the conditioning, counting the edited image

Notes:

  - Only the first image can be masked, and only if you want to — masking is optional. The mask belongs to the on-node `image` (the one being edited); the `ref_image_*` sockets take IMAGE only, have no mask input, and any alpha channel on them is dropped before encoding. There is deliberately no way to mask a reference
  - That is how the model's inpaint conditioning works, not an arbitrary node choice: a mask drives the `concat_latent_image` / `concat_mask` inpaint conditioning, which has to line up pixel-for-pixel with the latent being denoised. The extra references are arbitrary images of arbitrary size, so a mask on one would have nothing to align to
  - Reference order is edited image first, then `ref_image_2`, `ref_image_3`, … in socket order
  - A socket carrying a batch of images is split into one reference latent per image, since the model treats each list entry as a separate reference rather than as a batch
  - Flux.2 Klein is trained around 1.0 MP. Sampling far above that — an 8 MP source at `target_megapixels` 0 — degrades badly at the low step counts the distilled checkpoints use, so leave the default at `1.0` unless you have a reason not to
  - Every reference image adds tokens to the model's context, so `target_megapixels` is the lever for both quality and VRAM: it caps the big images and, just as importantly, brings undersized references up to a resolution that actually contributes detail
