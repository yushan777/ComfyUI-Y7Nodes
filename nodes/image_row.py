import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from PIL.PngImagePlugin import PngInfo
import os
import json
from ..utils.colored_print import color, style
from .documentation import descriptions, as_html
import folder_paths


class Y7_ImageRow:
    """
    Combines up to 4 images horizontally into a single row,
    resizing them to a consistent height. Saves the result
    and displays a preview.
    """
    

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                 "captions": ("STRING", {"default": "monday, tuesday, wednesday, thursday", "tooltip": "Comma-separated list of captions for each image"}), 
                 "caption_size": ("INT", {"min": 12, "max": 100, "default": 16, "tooltip": "Font size for the captions"}),                 
                 "save_image": ("BOOLEAN", {"default": False, "label_on": "Save to output", "label_off": "Preview only", "tooltip": "Toggle between saving to output directory or creating a preview only"}),
                 "save_filename": ("STRING", {"default": "image_row", "tooltip": "Prefix for the saved image filename"}),
            },
            "optional": {
                "image1": ("IMAGE", {"tooltip": "First image to include in the row"}),
                "image2": ("IMAGE", {"tooltip": "Second image to include in the row"}),
                "image3": ("IMAGE", {"tooltip": "Third image to include in the row"}),
                "image4": ("IMAGE", {"tooltip": "Fourth image to include in the row"}),
            },
            "hidden": {
                "prompt": "PROMPT", 
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    # we don't define RETURN_TYPES
    # because the output is purely for the UI via the return dictionary.
    RETURN_TYPES = ()
    OUTPUT_NODE = True  # Indicates this node provides a UI output/preview
    FUNCTION = "process_image_row"
    CATEGORY = "Y7Nodes/Image"

    def tensor_to_pil(self, tensor):
        """Converts a torch tensor (B, H, W, C) to a list of PIL Images."""
        if tensor is None:
            return []
        # Remove batch dim, convert to numpy, scale to 0-255, change to uint8
        images = tensor.cpu().numpy() * 255.0
        images = np.clip(images, 0, 255).astype(np.uint8)
        # Convert each image in the batch (if any) to PIL
        pil_images = [Image.fromarray(img) for img in images]
        return pil_images

    def process_image_row(self, captions, caption_size=16, save_image=False, save_filename="image_row", image1=None, image2=None, image3=None, image4=None, prompt=None, extra_pnginfo=None):
        # Parse captions
        caption_list = [cap.strip() for cap in captions.split(',')]
        
        # Check if all captions are empty
        all_captions_empty = all(not caption for caption in caption_list)
        
        # Collect all non-None images
        input_images_pil = []
        for img_tensor in [image1, image2, image3, image4]:
            if img_tensor is not None:
                pil_list = self.tensor_to_pil(img_tensor)
                if pil_list:
                    input_images_pil.append(pil_list[0])  # Add the first image of the batch

        if not input_images_pil:
            # Return empty preview if no images are provided
            return {"ui": {"images": []}}

        # Determine minimum height
        min_height = min(img.height for img in input_images_pil)

        # Resize images to minimum height, maintaining aspect ratio
        resized_images = []
        total_width = 0
        for img in input_images_pil:
            aspect_ratio = img.width / img.height
            new_width = int(min_height * aspect_ratio)
            resized_img = img.resize((new_width, min_height), Image.LANCZOS)
            resized_images.append(resized_img)
            total_width += new_width        

        # Prepare for caption bar
        # Find font to use for captions
        font_size = caption_size
        font = None
        
        # Try to use a TrueType font with a specific size
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
            "/System/Library/Fonts/Helvetica.ttc",  # macOS
            "C:\\Windows\\Fonts\\arial.ttf",  # Windows
            "/usr/share/fonts/TTF/DejaVuSans.ttf",  # Some Linux distros
        ]

        for path in font_paths:
            if os.path.exists(path):
                try:
                    font = ImageFont.truetype(path, font_size)
                    # print(f"TrueType Font Exists", color.ORANGE)
                    break
                except Exception:
                    continue
        
        # If no TrueType font is available, use the default
        if font is None:
            font = ImageFont.load_default()

        # Calculate caption bar height (0 if all captions are empty)
        caption_bar_height = 0 if all_captions_empty else 60  # Fixed height for caption bar
        
        # Create the combined image row with space for caption bar (if needed)
        combined_image = Image.new('RGB', (total_width, min_height + caption_bar_height), (0, 0, 0))
        # Paste images at the top
        current_x = 0
        image_positions = []  # Store positions for caption alignment
        
        for img in resized_images:
            combined_image.paste(img, (current_x, 0))
            # Store the position and width for caption alignment
            image_positions.append((current_x, img.width))
            current_x += img.width
            
        # Draw caption bar only if there are non-empty captions
        draw = ImageDraw.Draw(combined_image)
        
        if not all_captions_empty:
            # Draw caption bar (dark background)
            draw.rectangle(
                [(0, min_height), (total_width, min_height + caption_bar_height)],
                fill=(30, 30, 30)  # Dark background for caption bar
            )
            
            # Add captions in the caption bar
            for i, (start_x, width) in enumerate(image_positions):
                if i < len(caption_list) and caption_list[i].strip():
                    try:
                        # Get the caption text for this image
                        text = caption_list[i].strip()
                        
                        # Calculate text dimensions based on available methods
                        if hasattr(font, 'getbbox'):
                            # Modern Pillow (>=10.0.0)
                            bbox = font.getbbox(text)
                            text_width = bbox[2] - bbox[0]
                            text_height = bbox[3] - bbox[1]
                        elif hasattr(font, 'getsize'):
                            # Older Pillow
                            text_width, text_height = font.getsize(text)
                        
                        # Center text under its corresponding image
                        text_x = start_x + (width - text_width) // 2
                        
                        # Improve vertical centering by accounting for text baseline
                        # For TrueType fonts, we need to adjust the vertical position slightly
                        if hasattr(font, 'getbbox'):
                            # Modern Pillow - adjust for better vertical centering
                            text_y = min_height + (caption_bar_height - text_height) // 2 - 2
                        else:
                            # Older Pillow - use standard centering with a small adjustment
                            text_y = min_height + (caption_bar_height - text_height) // 2 - 2
                        
                        # Draw text
                        draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)
                        
                    except Exception as e:
                        # Fallback if there's an issue with the font or drawing
                        print(f"Error adding caption for image {i+1}: {e}")
                        try:
                            # Very basic fallback with better vertical centering
                            simple_x = start_x + 10
                            simple_y = min_height + (caption_bar_height // 2) - 10
                            draw.text((simple_x, simple_y), text, fill=(255, 255, 255))
                        except Exception:
                            pass

        # If selected to do so, save the image
        if save_image:
            output_dir = folder_paths.get_output_directory()
            file_prefix = save_filename
            preview_type = "output"
            subfolder = "" # Saved to root output

        # oetherwise preview the image (stored in temp)
        else: 
            output_dir = folder_paths.get_temp_directory()
            file_prefix = save_filename + "_preview"
            preview_type = "temp"
            subfolder = "" # Saved to root temp

        # Find the next available filename index
        counter = 1
        while True:
            filename = f"{file_prefix}_{counter:04}.png"
            full_path = os.path.join(output_dir, filename)
            if not os.path.exists(full_path):
                break
            counter += 1

        # Prepare workflow metadata 
        metadata = PngInfo()
        if prompt is not None:
            metadata.add_text("prompt", json.dumps(prompt))
        if extra_pnginfo is not None:
            for key, value in extra_pnginfo.items():
                metadata.add_text(key, json.dumps(value))

        # Save the image with metadata
        combined_image.save(full_path, pnginfo=metadata, compress_level=4) # Save with moderate PNG compression

        # Prepare preview data
        preview_data = [{
            "filename": filename,
            "subfolder": subfolder,
            "type": preview_type
        }]

        # Return the preview data for the UI to display
        return {"ui": {"images": preview_data}}
