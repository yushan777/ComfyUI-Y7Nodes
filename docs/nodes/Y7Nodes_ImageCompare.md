# Image Compare

Compare two images with a draggable slider and selectable blend modes.

Provides an interactive side-by-side comparison directly on the node. Drag the slider to reveal `image_a` over `image_b`, and switch blend modes to analyse differences.

The preview updates live as the slider is moved or the blend mode is changed, and persists across workflow-tab switches.

Blend modes:

  - `normal`: Slider reveals `image_a` over `image_b`
  - `difference`: Blended comparison for visual analysis of variations

Inputs:

  - `image_a`: First image (required)
  - `image_b`: Second image (optional)

This node has no outputs; it is a preview-only node for on-canvas comparison.

Quality of life:

  - **Copy Image B to Clipboard** button (also on the right-click menu as
    **Copy Image B (clipboard)**): copies the `image_b` preview to your
    clipboard as a PNG, ready to paste into any other app. The button briefly
    says "Copied!" when it works. Your browser only allows clipboard copying on
    a secure page, so if ComfyUI is open over plain `http://` (other than
    `localhost`) the button will say the clipboard is unavailable.
  - The node auto-resizes to match the aspect ratio of the input images
  - Slider position and blend mode are saved with the workflow

Based on `Eses Image Compare` by Eses Nodes.
