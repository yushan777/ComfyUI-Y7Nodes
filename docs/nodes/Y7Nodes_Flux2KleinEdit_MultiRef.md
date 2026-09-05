# Y7 Flux.2 Klein Edit Multi-Ref

Loads the primary image to be edited on-node, and takes extra reference images on growable list of inputs, so extra images can be fed in as additional visual context — a character sheet, a style reference, a product shot. Then prepares Klein edit conditioning (reference latents + an optional mask that limits which part of the picture is redrawn).

Paint a mask directly on the on-node image (via MaskEditor) to confine the edit to that area, or feed one into the `external_mask` socket; leave it unmasked for a plain whole-image edit.

**A mask only ever applies to the primary image** — the one picked on the node, `reference_latents[0]`, the image the output is actually made from. This is true of both routes in, painted and `external_mask`, and of every mask setting (`expand_mask`, `feather_mask`, `binary_mask`). The `ref_image_*` sockets cannot be masked at all: they are IMAGE-only, have no mask input, and any alpha channel on them is dropped before encoding. See the masking notes below for why.


Prompting with multiple references:

  - There is no special syntax. Klein taps the Qwen3-VL language model with the visual tower unused, so the text encoder never sees the reference images at all — they enter as VAE latents appended to the transformer's token sequence. Nothing like `<image1>`, `[Image 1]` or `@ref2` is a real token; refer to the images in plain English by their position
  - Position is the socket number: the on-node `image` is reference 1, `ref_image_2` is reference 2, `ref_image_3` is reference 3, and so on. The sockets are numbered to match
  - ComfyUI's own multi-reference Klein template words it as `Figure N`: "Have the man in Figure 1 put on the clothes from Figure 2, wear a hat, and carry a bag. Then, change the background environment to an African savannah while keeping the man in the same posture..."
  - `image 1` / `the first image` is the same kind of plain positional reference and is the wording BFL's own material tends to use. Both forms are just words to the model, so either should work — if a prompt is not binding to the right reference, fix the seed and try the other phrasing
  - Always pair the index with a noun — `the man in Figure 1`, `the clothes from Figure 2`, `the room from image 3`. A bare index has nothing to latch onto; the noun is what actually anchors the reference
  - Say what to preserve as well as what to change: "...preserve her facial identity, hairstyle and proportions from Figure 1". Reference images are context, not constraints — nothing forces the model to keep them

Inputs:

  - `image`: The main image being edited, picked on the node. This is the only image a mask applies to, and it always leads the reference list
  - `target_megapixels`: Resamples the image (and mask) to roughly this many megapixels before encoding, same maths as `ImageScaleToTotalPixels`. Applied independently to every reference image too, so each one lands on the budget whether it has to shrink or grow. `0` keeps every image at its original resolution
  - `crop_2_nearest_16px`: Centre-crops the image (and mask) and every reference image down to the nearest multiple of 16, which Flux.2 prefers. No-op if the dimensions are already aligned
  - `expand_mask`: Dilates the mask outward by this many pixels. Defaults to `16`, one full latent cell — the mask is used at 1/16 of the picture's resolution, so a tightly painted edge loses a sliver of what you painted. Raise to 24-32 for hair, fur and other soft edges, or when the replacement needs more room than what it replaces. `0` disables. Applies to the primary image's mask, from either route in — as do `feather_mask` and `binary_mask` below
  - `feather_mask`: Gaussian-blurs the mask edges by this radius. Defaults to `0` and is best left there — see the masking note below. Above about 32 it acts as a loose, fuzzy edit area rather than a tidy edge. `0` disables
  - `binary_mask`: Hard-thresholds the finished mask to pure black/white at `0.5`, applied last so the result is crisp. On by default: the MaskEditor brush has hardness and opacity settings, so painted masks are often soft-edged whether or not that was intended, and part-strength mask values only buy a muddy blend
  - `external_mask` (optional): A MASK from anywhere else — `LoadImageMask` on a black-and-white mask file, a segmentation node, a threshold — instead of painting one on the node. Applied straight after the image loads, so it goes through resize, crop and `expand_mask`/`feather_mask`/`binary_mask` exactly like a painted mask. It is stretched to the loaded image's dimensions, so match the aspect ratio or the mask will no longer line up with what you masked; the `debug` output warns when the shapes disagree. When connected and non-empty it replaces the painted mask; connected but blank, it falls back to the painted one and says so in `debug` and in the console. The usual cause of a blank one is `LoadImageMask` ("Load Image (as Mask)"), whose `channel` widget starts on **alpha**: a plain black-and-white mask file has no alpha channel, so that node returns an empty 64x64 mask and nothing here gets masked — set its `channel` to **red**. A batch of masks contributes only its first entry. Like a painted mask, it applies to the primary `image` alone — it does not and cannot mask a `ref_image_*`
  - `ref_image_2` … `ref_image_8` (optional): Additional reference images, used as visual context only. One empty socket is shown to start with and a new one appears each time you connect the last, up to eight. These cannot be masked — see the note below
  - `positive` (optional): Positive conditioning to patch. Left as an empty list if not connected
  - `negative` (optional): Negative conditioning, patched the same way as positive

Outputs:

  - `reference_latent`: VAE-encoded latent of the edited image only — the extra references go onto the conditioning, not into this latent
  - `positive`: Conditioning with `reference_latents` (edited image first, then each reference in socket order) and `concat_latent_image` set, plus `concat_mask` when a mask was painted
  - `negative`: Conditioning patched the same way as positive
  - `preview_image`: The edited image after resize/crop, for on-canvas preview
  - `preview_mask`: The mask after expand/feather/binarize. All-zero if no mask was painted
  - `debug`: A plain-text report of what the node actually did — the edited image's size at every stage, its latent size and token cost, the mask state (painted or not, coverage, expand/feather/hard-edge), every connected reference socket with its before/after size and token cost, the totals, and whether `positive`/`negative` were connected. Wire it into any text preview node (Y7 Show Anything will do) when a result is not what you expected. It ends with a `worth knowing` section that only appears when something is worth flagging: a socket carrying a batch (which counts as several references, not one), a reference that got enlarged, conditioning left unconnected, or a token total high enough to threaten your VRAM

Notes:

  - Only the primary image can be masked, and only if you want to — masking is optional. Both routes in behave the same way here: a mask painted in the MaskEditor and a mask wired into `external_mask` both attach to the on-node `image` and to nothing else. The `ref_image_*` sockets take IMAGE only, have no mask input, and any alpha channel on them is dropped before encoding. There is deliberately no way to mask a reference, and no combination of settings that makes one apply to a reference
  - That is not an arbitrary node choice: the mask has to line up pixel-for-pixel with the latent being denoised. The extra references are arbitrary images of arbitrary size, so a mask on one would have nothing to align to
  - What the mask actually does: Klein has no mask input of its own — it is not a fill/inpaint model, and its `in_channels` match its `out_channels`, so ComfyUI drops the `concat_latent_image` / `concat_mask` conditioning this node sets. The mask survives as the latent's `noise_mask`, which the sampler uses to restore everything outside the painted area after each step. That composite happens in latent space, at 1/16 of the picture's resolution
  - Which is why feathering here does so little: a feather under ~16px is erased by that downsample, and a wider one crossfades half-finished latents, decoding as a smeared seam rather than a blend. For a soft join between the edited area and the rest, blend in pixel space after decoding — Y7 Paste Cropped Image Back has a feather amount for exactly this
  - Set the sampler's `denoise` to 1.0. The unmasked area is restored every step regardless, so lowering `denoise` to protect it does nothing but weaken the edit
  - Reference order is `primary image` first, then `ref_image_2`, `ref_image_3`, … in socket order
  - A socket carrying a batch of images is split into one reference latent per image, since the model treats each list entry as a separate reference rather than as a batch
  - Flux.2 Klein is trained around 1.0 MP. Sampling far above that — an 8 MP source at `target_megapixels` 0 — degrades badly at the low step counts the distilled checkpoints use, so leave the default at `1.0` unless you have a reason not to
  - Every reference image adds tokens to the model's context, so `target_megapixels` is the lever for both quality and VRAM: it caps the big images and, just as importantly, brings undersized references up to a resolution that actually contributes detail
