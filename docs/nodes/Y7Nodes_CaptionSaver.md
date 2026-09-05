# Caption Saver

Save a caption string as a .txt file next to the source image, using the same filename stem.

Designed to pair with ImageBatchPath and any VLM node that outputs a STRING: connect `IMAGE_PATH` from ImageBatchPath and the caption `STRING` from the VLM node.

Compatible with any node that outputs a STRING. Examples: Florence2, MiniCPM, LLaVA, Qwen-VL, etc.

Example: `cat.jpg` → `cat.txt` saved in the same directory.

Inputs:

  - `string`: The caption text to write (required, must be connected)
  - `image_path`: Full path to the source image (required, must be connected — e.g. from ImageBatchPath)
  - `overwrite`: If true, overwrites any existing .txt file. If false, appends a counter to avoid overwriting (e.g. `cat_01.txt`, `cat_02.txt`)

This node has no outputs — it is a terminal/output node.
