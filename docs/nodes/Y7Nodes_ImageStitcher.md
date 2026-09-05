# Image Stitcher

Stitches 2–8 images side-by-side or top-and-bottom

Use the `image_count` widget to choose how many image inputs are shown (2–8).

All images are resized to match the first image's height (horizontal) or width (vertical) before concatenation.

Inputs:

  - `image_count`: Number of image sockets to display (2–8)
  - `orientation`: `Side-by-Side (Horizontal)` or `Top-and-Bottom (Vertical)`
  - `image1` … `imageN`: Images to stitch together
