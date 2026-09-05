# Image Batch Path

Load a batch of images from a directory and output them as a list of image tensors with matching file paths.

Supports jpg, jpeg, png, and webp. Images are EXIF-transposed and converted to RGB float32 tensors.

Designed to pair with CaptionSaver: the `IMAGE_PATH` output tells CaptionSaver where to write each .txt file, and `IMAGE` feeds into a VLM node for captioning.

Inputs:

  - `image_dir`: Path to the directory containing images
  - `batch_size`: Number of images to load (0 = all images in the directory)
  - `start_from`: 1-based index of the first image to load — useful for resuming from a specific point
  - `sort_method`: Order to load images — `sequential` (alphabetical), `reverse`, or `random`

Outputs (both are lists):

  - `IMAGE`: List of image tensors (one per image)
  - `IMAGE_PATH`: List of full file paths matching each image tensor

Note: When `sort_method` is `random`, the node re-evaluates on every run.
