import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import os
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
                 "captions": ("STRING", {"default": "image1, image2, image3, image4"}), 
                 "save_image": ("BOOLEAN", {"default": False, "label_on": "Save to output", "label_off": "Preview only"}),
                 "save_filename": ("STRING", {"default": "image_row"}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
            }
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

    def process_image_row(self, captions="image1, image2, image3, image4", save_image=False, save_filename="image_row", image1=None, image2=None, image3=None, image4=None):
        # Parse captions
        caption_list = [cap.strip() for cap in captions.split(',')]
        
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
            
        # print(f"total_width = {total_width}", color.ORANGE)

        # Add captions to images - match captions to available images in order
        for i, img in enumerate(resized_images):
            # Only add caption if one exists for this image index
            if i < len(caption_list) and caption_list[i].strip():
                # Create a draw object
                draw = ImageDraw.Draw(img)
                
                try:
                    # Use a larger font size for better visibility
                    font_size = 20
                    font = None
                    
                    # Try to use a TrueType font with a specific size
                    font_paths = [
                        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
                        "/System/Library/Fonts/Helvetica.ttc",  # macOS
                        "C:\\Windows\\Fonts\\arial.ttf",  # Windows
                        "/usr/share/fonts/TTF/DejaVuSans.ttf",  # Some Linux distros
                    ]
    # Define both font paths
            
                    for path in font_paths:
                        if os.path.exists(path):
                            try:
                                font = ImageFont.truetype(path, font_size)
                                print(f"TrueType Font Exists", color.ORANGE)
                                break
                            except Exception:
                                continue
                    
                    # If no TrueType font is available, use the default
                    if font is None:
                        font = ImageFont.load_default()
                    
                    print(f"font={font}", color.ORANGE)

                    # Get the caption text for this image
                    text = caption_list[i]
                    
                    # Calculate text dimensions based on available methods
                    if hasattr(font, 'getbbox'):
                        # Modern Pillow (>=9.2.0)
                        bbox = font.getbbox(text)
                        text_width = bbox[2] - bbox[0]
                        text_height = bbox[3] - bbox[1]
                        print(f"Modern Pillow", color.ORANGE)
                    elif hasattr(font, 'getsize'):
                        # Older Pillow
                        text_width, text_height = font.getsize(text)
                        print(f"Older Pillow", color.ORANGE)
                    else:
                        # Fallback
                        text_width = len(text) * (font_size // 2)
                        text_height = font_size + 4
                        print(f"Fallback", color.ORANGE)
                    
                    # Calculate position (centered at bottom with padding)
                    padding = 10
                    text_x = (img.width - text_width) // 2
                    text_y = img.height - text_height - padding
                    
                    # Draw semi-transparent background for text
                    draw.rectangle(
                        [(text_x - 5, text_y - 5), (text_x + text_width + 5, text_y + text_height + 5)],
                        fill=(0, 0, 0, 64)  # semi opaque background
                    )
                    
                    # Draw text
                    draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)
                    
                except Exception as e:
                    # Fallback if there's an issue with the font or drawing
                    print(f"Error adding caption for image {i+1}: {e}")
                    try:
                        # Very basic fallback
                        draw.rectangle(
                            [(10, img.height - 30), (img.width - 10, img.height - 5)],
                            fill=(0, 0, 0, 200)
                        )
                        draw.text((15, img.height - 25), text, fill=(255, 255, 255))
                    except Exception:
                        pass

        # Create the combined image row
        combined_image = Image.new('RGB', (total_width, min_height))
        current_x = 0
        for img in resized_images:
            combined_image.paste(img, (current_x, 0))
            current_x += img.width

        # Create the combined image row
        combined_image = Image.new('RGB', (total_width, min_height))
        current_x = 0
        for img in resized_images:
            combined_image.paste(img, (current_x, 0))
            current_x += img.width

        # Create the combined image row
        combined_image = Image.new('RGB', (total_width, min_height))
        current_x = 0
        for img in resized_images:
            combined_image.paste(img, (current_x, 0))
            current_x += img.width

        # --- Saving the image ---
        if save_image:
            output_dir = folder_paths.get_output_directory()
            file_prefix = save_filename
            preview_type = "output"
            subfolder = "" # Saved to root output
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

        # Save the image
        combined_image.save(full_path, compress_level=4) # Save with moderate PNG compression

        # --- Prepare preview data ---
        preview_data = [{
            "filename": filename,
            "subfolder": subfolder,
            "type": preview_type
        }]

        # Return the preview data for the UI
        return {"ui": {"images": preview_data}}
