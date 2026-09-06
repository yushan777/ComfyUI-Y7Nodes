# Y7 Flux.2 Klein Edit Multi-Ref

Edit a picture with Flux.2 Klein, with up to seven extra pictures for the model to look at — a character sheet, a style reference, a product shot.

You pick the picture being edited on the node itself, and connect the extras to the `ref_image_*` sockets. To change only part of the picture, paint a mask on it (right-click the node → Open in MaskEditor) or feed one into `external_mask`. Leave it unmasked to edit the whole thing.

**A mask only ever affects the picture being edited.** The extra reference pictures cannot be masked — they are there to be looked at, nothing more.

## Writing the prompt

There is no special syntax — just plain English. Refer to the pictures by their number: the one on the node is 1, `ref_image_2` is 2, `ref_image_3` is 3, and so on.

  - Both `Figure 1` and `image 1` / `the first image` work. If a prompt isn't picking up the right picture, keep the seed and try the other wording.
  - Always pair the number with a noun: *the man in Figure 1*, *the clothes from Figure 2*, *the room from image 3*. A bare number has nothing to latch onto.
  - Say what to keep as well as what to change: *"...keep her face, hair and proportions from Figure 1"*. The reference pictures are suggestions, not rules — nothing forces the model to stick to them.

Example, from ComfyUI's own multi-reference Klein template:

> Have the man in Figure 1 put on the clothes from Figure 2, wear a hat, and carry a bag. Then, change the background environment to an African savannah while keeping the man in the same posture...

## Inputs

  - `vae`: The VAE from your Flux.2 checkpoint or VAE loader.
  - `image`: The picture being edited, picked on the node. This is the only one a mask applies to, and it's always reference 1.
  - `target_megapixels`: How big to make the picture you are editing — `1.0` is about a million pixels, keeping its shape. This also sets how big your finished picture comes out. `0` leaves it at its original size. Klein is built for around `1.0`; going much higher usually makes things worse.
  - `ref_megapixels`: The same, for each reference picture. These don't affect your output size. It applies to each one separately, so they add up — one picture plus six references at `1.0` is seven megapixels of work. If you're slow or running out of memory, turn this down first; try `0.5`.
  - `crop_2_nearest_16px`: Trims a few pixels off the edges so the width and height divide by 16, which Flux.2 prefers. Does nothing if they already do. Default on.
  - `expand_mask`: Grows the area you painted outwards by this many pixels. Default `16` — the mask is used at a much smaller size internally, so a tight edge tends to lose a sliver of what you painted. Try 24–32 for hair, fur or other soft edges.
  - `feather_mask`: Softens the mask edge. Default `8`. It makes almost no difference to what the model generates; it matters when you paste the result back over the original afterwards (`ImageCompositeMasked`, or Y7 Paste Cropped Image Back), where it gives a soft join instead of a hard cut. Above about 32 it becomes a loose, fuzzy edit area rather than a tidy edge.
  - `binary_mask`: Makes the mask edge hard — every part you painted is either fully changed or fully left alone. Off by default, so the softening from `feather_mask` survives. Turn it on if a soft brush has left half-strength areas that come out muddy, or if a later node needs a strictly hard mask. It runs last, so it overrides `feather_mask`.
  - `external_mask` (optional): A mask from somewhere else instead of painting one here. White marks the part to change. It gets stretched to match the picture being edited, so give it the same shape. When connected and not blank it replaces anything painted on the node; if it's blank, the painted mask is used instead and `debug` says so.

    Using **Load Image (as Mask)** with a plain black-and-white file? Set its `channel` to **red**. On **alpha** (where it starts) that node hands back an empty mask and nothing gets masked.
  - `ref_image_2` … `ref_image_8` (optional): Extra pictures for the model to look at. Connect one and another empty socket appears, up to seven extras. These can't be masked.
  - `positive` / `negative` (optional): Your conditioning, passed through with the reference pictures added.

## Outputs

  - `reference_latent`: The edited picture, encoded — connect this to your sampler's latent input.
  - `positive` / `negative`: Your conditioning with all the reference pictures attached.
  - `preview_image`: The picture being edited, after resizing and cropping.
  - `preview_mask`: The mask after expand/feather/binary. All black if there is no mask.
  - `debug`: A plain-text report of what the node actually did, including any problems. Wire it into Y7 Show Anything if something looks wrong.
