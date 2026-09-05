# Color Match (Masked)

Color matches the target image to a reference while excluding masked regions from the calculation.

Ideal for correcting color shifts after inpainting. The mask excludes regions (like the inpainted area) from BOTH images during color transfer calculation, preventing the original colors from bleeding into the result.

Example: After changing a red car to blue via inpainting, the background may have a red tint. This node calculates color correction using only the non-masked areas, then applies it without affecting the inpainted region.

Inputs:

  - `image_ref`: Reference image (e.g., original before inpainting)
  - `image_target`: Target image to color match (e.g., result after inpainting)
  - `mask`: Mask where white (1.0) = areas to exclude from color matching
  - `method`: Color transfer algorithm - `mkl` (Monge-Kantorovich), `hm` (histogram), `reinhard`, `mvgd` (Multi-Variate Gaussian)
  - `strength`: Blend between original and color-matched result (0.0 = no change, 1.0 = full correction)
  - `feather`: Blur radius for mask edges to create smooth transitions (0-100 pixels)

Output:

  - `image`: The color-matched result with masked areas preserved unchanged

Requires the `color-matcher` library: `pip install color-matcher`
