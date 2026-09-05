# Load Image

Load an image with support for subdirectories — otherwise identical to the native Load Image node.

The native ComfyUI Load Image node only lists files directly in the `input` folder. This node walks the entire `input` directory tree so images organised into subfolders appear in the dropdown.

Outputs:

  - `image`: RGB image tensor
  - `mask`: Alpha channel as a mask (zeros if no alpha channel present)
