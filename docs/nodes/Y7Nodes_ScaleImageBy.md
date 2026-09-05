# Scale Image By

Scales an image by a multiplier while preserving aspect ratio.

Multiplies both width and height by `scale_by`, then resamples using the chosen method.

The scaled dimensions are displayed directly on the node after execution.

Inputs:

  - `image`: The input image to scale
  - `upscale_method`: Resampling algorithm — `nearest-exact`, `bilinear`, `area`, `bicubic`, or `lanczos`
  - `scale_by`: Multiplier applied to both dimensions (default: 1.0, range: 0.01–8.0)
  - `resolution_steps`: Snap output dimensions to the nearest multiple of this value (default: 8). Common values: 8, 16, 64

Outputs:

  - `image`: The scaled image
  - `width`: Output width in pixels (INT)
  - `height`: Output height in pixels (INT)
