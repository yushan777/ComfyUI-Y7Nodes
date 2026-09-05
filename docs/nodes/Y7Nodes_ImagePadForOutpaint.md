# Pad Image for Outpainting

Clone of the built-in Pad Image for Outpainting node, with a `step` input that snaps each padding value to a chosen multiple.

Pads an image on any/all sides ready for outpainting, generating a feathered mask over the new regions.

Inputs:

  - `image`: The input image to pad
  - `left` / `top` / `right` / `bottom`: Pixels to add to each edge
  - `feathering`: Width of the soft gradient at the border between original and padded regions
  - `step`: Each of `left`/`top`/`right`/`bottom` is independently snapped to the nearest multiple of this value, ties rounding up, before padding is applied (default: 8)

Outputs:

  - `IMAGE`: The padded image
  - `MASK`: Mask covering the padded regions, feathered at the border
