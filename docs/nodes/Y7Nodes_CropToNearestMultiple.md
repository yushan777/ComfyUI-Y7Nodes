# Crop to Nearest Multiple

Crops images to ensure dimensions are divisible by a specified multiple value - default 16px. You can always use a calculator if you prefer.

Control over horizontal and vertical cropping, making it ideal for situations where only one dimension needs adjustment.

Inputs:

  - `image`: The input image to check and optionally crop
  - `multiple`: The value that dimensions must be a multiple of (default: 16). Common values are 8, 16, 32
  - `horizontal_crop`: Crop position for width adjustment - `center`, `left`, `right`, or `none`
  - `vertical_crop`: Crop position for height adjustment - `center`, `top`, `bottom`, or `none`

Outputs:

  - `crop_preview`: Original image with a red overlay showing the areas that will be cropped and removed
  - `cropped_image`: The cropped image (or original if no cropping needed)
  - `info`: Status message with dimension information and cropping details

Behavior:

  - If both dimensions are already a multiple of the specified value, no cropping occurs
  - Crops to the nearest multiple down (e.g., 721 → 720 with multiple=16)
  - Setting crop position to `none` disables cropping for that dimension
  - Smart logic only applies crop settings to dimensions that need adjustment
  - The node displays the cropped dimensions (e.g., '1024 x 768') directly on the node after execution

Note on 'center' cropping:

  When using `center` with odd-numbered pixel differences, integer division rounds down, causing a slight bias (max 1px).

    Example: width=721, target=720, diff=1 → removes 1px from right only

    Example: width=723, target=720, diff=3 → removes 1px from left, 2px from right

  This is standard behavior in image processing and the bias is minimal.
