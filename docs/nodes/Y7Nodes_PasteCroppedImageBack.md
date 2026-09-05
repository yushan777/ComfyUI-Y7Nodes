# Paste Cropped Image Back

Paste a crop image onto a base image at a region defined by edge-relative coordinates.

Sometimes you may want to change or refine a specific area of an image without affecting the rest too much — for example, fixing a face, hand, or background detail after generation. A typical workflow is to crop the region, run it through img2img or inpainting, then paste the result back using this node.

Works well with the `OLM Drag Crop` custom node, which lets you visually drag-select a crop region and outputs the crop coordinates directly — those coordinates can be wired into this node's `top`, `left`, `right`, and `bottom` inputs.

Unlike the WAS equivalent, `right` and `bottom` are pixel offsets measured inward from the RIGHT and BOTTOM edges of the base image, rather than absolute coordinates from the top-left.

Paste region calculated as:

  - `x1 = left`
  - `y1 = top`
  - `x2 = image_width - right`
  - `y2 = image_height - bottom`

The `image_crop` is always resized to exactly fit the paste region. If it was upscaled for editing (e.g. sent through img2img at a higher resolution), it will be scaled back down during pasting. There is no aspect-ratio preservation — if the aspect ratio of the crop image differs from the paste region, it will be stretched to fit and appear distorted.

Inputs:

  - `image_orig`: Base image to paste onto
  - `image_crop`: Image to paste into the defined region (will be resized to fit)
  - `left`: Pixels from the left edge to the left of the paste region
  - `top`: Pixels from the top edge to the top of the paste region
  - `right`: Pixels inward from the RIGHT edge to the right of the paste region
  - `bottom`: Pixels inward from the BOTTOM edge to the bottom of the paste region
  - `crop_blending`: Feathering amount at the edges of the pasted region (0.0–1.0)
  - `crop_sharpening`: Number of sharpening passes applied to the crop before pasting (0–3)

Outputs:

  - `IMAGE`: The base image with the crop pasted in
  - `MASK`: The blended mask used for the paste operation
