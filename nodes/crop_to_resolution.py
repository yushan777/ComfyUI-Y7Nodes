import torch


class Y7Nodes_CropToResolution:
    """
    A node to crop images to align with resolution step multiples
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "divisible_by": ("INT", {
                    "default": 16, 
                    "min": 1, 
                    "max": 1024, 
                    "step": 1,
                    "tooltip": "Image dimensions must be divisible by (or be multiples of) this value"
                }),
                "horizontal_crop": (["center", "left", "right", "none"], {
                    "default": "center",
                    "tooltip": "Horizontal crop position: where to keep content when width needs adjustment (none = no cropping)"
                }),
                "vertical_crop": (["center", "top", "bottom", "none"], {
                    "default": "center",
                    "tooltip": "Vertical crop position: where to keep content when height needs adjustment (none = no cropping)"
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("crop_preview", "image", "info")
    
    FUNCTION = "check_dimensions"
    CATEGORY = "Y7Nodes"
    
    def check_dimensions(self, image, divisible_by, horizontal_crop, vertical_crop):
        """
        Check image dimensions and process accordingly
        """
        # Get image dimensions
        # In ComfyUI, images are typically in shape: [batch, height, width, channels]
        height = image.shape[1]
        width = image.shape[2]
        
        # Check if dimensions are even multiples of divisible_by
        width_is_multiple = (width % divisible_by) == 0
        height_is_multiple = (height % divisible_by) == 0
        
        # Start with original image
        output_image = image
        # Create preview with red overlay for crop areas
        crop_preview = image.clone()
        
        if width_is_multiple and height_is_multiple:
            info = f"✓ Dimensions are valid: {width}x{height} (divisible by {divisible_by}). No cropping necessary."
        else:
            # Calculate target dimensions (nearest multiples down)
            target_width = (width // divisible_by) * divisible_by
            target_height = (height // divisible_by) * divisible_by
            
            suggestions = []
            if not width_is_multiple:
                suggestions.append(f"Width {width} → {target_width}")
            if not height_is_multiple:
                suggestions.append(f"Height {height} → {target_height}")
            
            # Determine if cropping should be performed
            should_crop = (horizontal_crop != "none" and not width_is_multiple) or \
                         (vertical_crop != "none" and not height_is_multiple)
            
            if should_crop:
                # Calculate crop offsets based on separate horizontal and vertical settings
                width_diff = width - target_width
                height_diff = height - target_height
                
                # Determine horizontal crop position
                # Note: When using "center" with odd-numbered differences, integer division (//) rounds down
                # Example: width=721, target=720, diff=1 -> left=0 (removes 1px from right only)
                # Example: width=723, target=720, diff=3 -> left=1 (removes 1px left, 2px right)
                # This is standard behavior in image processing - the slight bias is minimal (max 1px difference)
                if not width_is_multiple and horizontal_crop != "none":
                    if horizontal_crop == "center":
                        left = width_diff // 2
                    elif horizontal_crop == "left":
                        left = 0
                    elif horizontal_crop == "right":
                        left = width_diff
                else:
                    left = 0  # No horizontal cropping needed
                
                # Determine vertical crop position
                # Same rounding behavior applies to vertical cropping
                if not height_is_multiple and vertical_crop != "none":
                    if vertical_crop == "center":
                        top = height_diff // 2
                    elif vertical_crop == "top":
                        top = 0
                    elif vertical_crop == "bottom":
                        top = height_diff
                else:
                    top = 0  # No vertical cropping needed
                
                right = left + target_width
                bottom = top + target_height
                
                # Create crop preview with red overlay on areas to be cropped
                # Apply semi-transparent red overlay to cropped areas
                red_overlay = torch.tensor([1.0, 0.0, 0.0], device=image.device, dtype=image.dtype)
                alpha = 0.5  # Semi-transparent
                
                # Apply red to top area
                if top > 0:
                    crop_preview[:, :top, :, :] = crop_preview[:, :top, :, :] * (1 - alpha) + red_overlay * alpha
                
                # Apply red to bottom area
                if bottom < height:
                    crop_preview[:, bottom:, :, :] = crop_preview[:, bottom:, :, :] * (1 - alpha) + red_overlay * alpha
                
                # Apply red to left area (within the vertical crop bounds)
                if left > 0:
                    crop_preview[:, top:bottom, :left, :] = crop_preview[:, top:bottom, :left, :] * (1 - alpha) + red_overlay * alpha
                
                # Apply red to right area (within the vertical crop bounds)
                if right < width:
                    crop_preview[:, top:bottom, right:, :] = crop_preview[:, top:bottom, right:, :] * (1 - alpha) + red_overlay * alpha
                
                # Crop the image
                output_image = image[:, top:bottom, left:right, :]
                
                crop_info = []
                if not width_is_multiple and horizontal_crop != "none":
                    crop_info.append(f"horizontal: {horizontal_crop}")
                if not height_is_multiple and vertical_crop != "none":
                    crop_info.append(f"vertical: {vertical_crop}")
                
                info = f"✓ Image cropped from {width}x{height} to {target_width}x{target_height} ({', '.join(crop_info)})"
            else:
                info = f"✗ Dimensions not divisible by {divisible_by}: {width}x{height}\n" + "\n".join(suggestions) + "\nSet horizontal_crop/vertical_crop to enable automatic cropping"
        
        print(info)
        
        return (crop_preview, output_image, info)
