import torch
# import numpy as np
from .documentation import descriptions, as_html

# takes a grid of images (like those generated in XY-plots) and processes them into a batch. 
class Y7Nodes_Grid2Batch:

    def __init__(self):
        # Define default values
        pass
    
    # ====================================================
    # Defines what inputs your node accepts
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "The input grid image containing multiple images arranged in a grid pattern"
                }),
                "rows": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Number of rows in the grid"
                }),
                "columns": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Number of columns in the grid"
                }),
                "width": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 2048,
                    "step": 8,
                    "tooltip": "Width of each individual image in the grid (in pixels)"
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 2048,
                    "step": 8,
                    "tooltip": "Height of each individual image in the grid (in pixels)"
                }),
                "x_header": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1000,
                    "step": 1,
                    "tooltip": "If grid has an X header, specify its width in pixels. Set to 0 if there is no header."
                }),
                "y_header": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1000,
                    "step": 1,
                    "tooltip": "If grid has a Y header, specify its height in pixels. Set to 0 if there is no header."
                }),
            }
        }
    
    # Return types - we're returning a batch of images
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_grid"
    CATEGORY = "Y7Nodes/Image/Processing"
    
    def process_grid(self, image, rows, columns, width, height, x_header, y_header):
        # 1. Determine the dimensions of the input image
        input_batch, input_height, input_width, input_channels = image.shape
        
        # 2. Check if dimensions match the expected grid size
        expected_width = columns * width
        expected_height = rows * height
        
        # Check if the input image dimensions match the expected dimensions
        # (after accounting for headers)
        if input_width != expected_width + x_header or input_height != expected_height + y_header:
            raise ValueError(f"With your {rows}x{columns} grid of {width}x{height} images and headers of ({x_header}x{y_header}), "
                            f"I'm expecting an input image of size {expected_width + x_header}x{expected_height + y_header} "
                            f"but your input image is {input_width}x{input_height}. Please check your numbers")
        
        # 3. Handle headers if they exist
        if x_header > 0 or y_header > 0:
            # Crop away headers to get just the grid
            grid_image = image[:, y_header:, x_header:, :]
        else:
            # No headers, use the original image
            grid_image = image
        
        # Create a batch of images from the grid
        batch_size = rows * columns
        batch_images = torch.zeros((batch_size, height, width, input_channels), 
                                  dtype=grid_image.dtype, device=grid_image.device)
        
        # Extract each image from the grid and add it to the batch
        idx = 0
        for r in range(rows):
            for c in range(columns):
                # Calculate the position of this image in the grid
                y_start = r * height
                y_end = (r + 1) * height
                x_start = c * width
                x_end = (c + 1) * width
                
                # Extract the image and add it to the batch
                batch_images[idx] = grid_image[:, y_start:y_end, x_start:x_end, :]
                idx += 1
        
        return (batch_images,)
