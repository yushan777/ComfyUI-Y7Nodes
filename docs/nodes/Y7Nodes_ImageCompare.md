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

  - The node auto-resizes to match the aspect ratio of the input images
  - Slider position and blend mode are saved with the workflow

Based on `Eses Image Compare` by Eses Nodes.
