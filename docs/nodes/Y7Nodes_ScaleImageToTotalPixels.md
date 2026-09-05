# Scale Image to Total Pixels

Scales an image to a target total pixel count (in megapixels) while preserving aspect ratio.

Computes a uniform scale factor so that `width × height` equals the target megapixel count, then resamples using the chosen method.

The scaled dimensions are displayed directly on the node after execution.

Inputs:

  - `image`: The input image to scale
  - `upscale_method`: Resampling algorithm — `nearest-exact`, `bilinear`, `area`, `bicubic`, or `lanczos`
  - `megapixels`: Target total pixel count in megapixels (default: 1.0, range: 0.01–16.0)
  - `resolution_steps`: Snap output dimensions to the nearest multiple of this value (default: 8). Common values: 8, 16, 64

Outputs:

  - `image`: The scaled image
  - `width`: Output width in pixels (INT)
  - `height`: Output height in pixels (INT)
