# Image Size Presets

Node to provide image width and height from a named preset set, with an optional custom dimensions file.

Inputs:

  - `preset`: Selects the active dimension set: `default`, `flux.2`, `qwen-image`, or `custom*`.
  - `dimension`: Dropdown of dimensions for the selected preset set. Updates dynamically when preset changes.
  - `custom_w`: Width to use when 'Custom' is selected from the dimension dropdown.
  - `custom_h`: Height to use when 'Custom' is selected from the dimension dropdown.

Outputs:

  - `Width`: The selected or custom width.
  - `Height`: The selected or custom height.

The `custom*` preset loads from `custom_dimensions.json` in the `nodes` directory; falls back to `default` if missing or invalid.

Examine `custom_dimensions_example.json` for the expected format.
