# Dual Resolution Selector

Gives a base and a final width/height for one aspect ratio at two megapixel targets.

Wire `base_width` / `base_height` to the empty latent and `final_width` / `final_height` to a Krea 2 Two- or Three-Stage Sampler to run stage 1 small and finish large.

## Inputs

  - `aspect_ratio`: 1:1 to 21:9
  - `base_megapixels`: Size of the base resolution (1.0 = 1024x1024 worth of pixels)
  - `final_megapixels`: Size of the final resolution
  - `multiple`: Both dimensions are rounded to this

## Outputs

  - `base_width` / `base_height`: The base resolution
  - `final_width` / `final_height`: The final resolution


