"""
Y7Nodes Color Match (Masked)

A color matching node that excludes masked regions from both the reference 
and target images during color transfer calculation. This is particularly 
useful after inpainting, where you want to color-match the unaltered areas 
back to the original without the inpainted region affecting the color transfer.

Based on the ColorMatch node from ComfyUI-KJNodes by @kijai
Uses the color-matcher library: https://github.com/hahnec/color-matcher
"""

import torch
import torch.nn.functional as F
import numpy as np


class Y7Nodes_ColorMatchMasked:
    """Color match images while excluding masked regions from the calculation."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_ref": ("IMAGE", {
                    "tooltip": "Reference image for color matching (e.g., original before inpainting)"
                }),
                "image_target": ("IMAGE", {
                    "tooltip": "Target image to apply color correction to (e.g., result after inpainting)"
                }),
                "method": ([
                    'mkl',
                    'hm', 
                    'reinhard', 
                    'mvgd', 
                    'hm-mvgd-hm', 
                    'hm-mkl-hm',
                ], {
                    "default": 'mkl',
                    "tooltip": "Color transfer method: mkl (Monge-Kantorovich), hm (histogram), reinhard, mvgd (Multi-Variate Gaussian)"
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "Mask where white (1.0) = areas to exclude from color matching (e.g., inpainted region). If not provided, color matching is applied to the entire image."
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "tooltip": "Blend strength between original and color-matched result (0=no change, 1=full correction)"
                }),
                "feather": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 100, 
                    "step": 1,
                    "tooltip": "Feather/blur radius for the mask edge transition in pixels"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "color_match_masked"
    CATEGORY = "Y7Nodes/Image"
    
    DESCRIPTION = """
Color Match (Masked) - Color matches the target image to the reference while 
excluding masked regions from BOTH images during the color transfer calculation.

Use case: After inpainting (e.g., changing a red car to blue), the rest of the 
image may have a color shift. This node calculates color correction using only 
the non-masked areas (background), then applies that correction to the non-masked 
areas while keeping the inpainted region (the blue car) unchanged.
"""

    def _gaussian_blur_mask(self, mask: torch.Tensor, radius: int) -> torch.Tensor:
        """Apply Gaussian blur to a mask tensor for feathering edges."""
        if radius <= 0:
            return mask
        
        # Ensure mask is 4D for conv2d: (B, C, H, W)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
        
        # Create Gaussian kernel
        kernel_size = int(radius * 2) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        sigma = radius / 3.0
        x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32)
        kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel_2d = kernel_2d.to(mask.device)
        
        # Expand kernel for conv2d
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)
        
        # Apply convolution with padding
        padding = kernel_size // 2
        blurred = F.conv2d(mask, kernel_2d, padding=padding)
        
        # Return in original shape
        return blurred.squeeze(1) if blurred.shape[1] == 1 else blurred

    def color_match_masked(self, image_ref, image_target, method, mask=None, strength=1.0, feather=0):
        """
        Perform color matching while excluding masked regions.
        
        Args:
            image_ref: Reference image tensor (B, H, W, C)
            image_target: Target image tensor to color match (B, H, W, C)
            method: Color transfer method
            mask: Optional mask tensor (B, H, W) where 1.0 = exclude from color matching. If None, applies color matching to entire image.
            strength: Blend strength (0-1)
            feather: Gaussian blur radius for mask edges
            
        Returns:
            Color matched image tensor
        """
        try:
            from color_matcher import ColorMatcher
        except ImportError:
            raise ImportError(
                "color-matcher library not found. Please install it with: pip install color-matcher"
            )
        
        # Move to CPU for processing
        image_ref = image_ref.cpu()
        image_target = image_target.cpu()
        
        batch_size = image_target.shape[0]
        _, H, W, C = image_target.shape
        
        # If no mask provided, perform standard color matching on entire images
        if mask is None:
            # Ensure reference batch size matches or is 1
            if image_ref.shape[0] > 1 and image_ref.shape[0] != batch_size:
                raise ValueError(
                    "ColorMatchMasked: Use either a single reference image or a matching batch of reference images."
                )
            
            cm = ColorMatcher()
            output_images = []
            
            for i in range(batch_size):
                target_np = image_target[i].numpy()
                ref_np = image_ref[0 if image_ref.shape[0] == 1 else i].numpy()
                
                try:
                    # Perform color matching on full images
                    matched = cm.transfer(src=target_np, ref=ref_np, method=method)
                    
                    # Apply strength blending
                    if strength < 1.0:
                        result = target_np + strength * (matched - target_np)
                    else:
                        result = matched
                    
                    # Clamp to valid range
                    result = np.clip(result, 0, 1)
                    
                except Exception as e:
                    print(f"ColorMatchMasked: Error during color transfer: {e}")
                    result = target_np
                
                output_images.append(torch.from_numpy(result.astype(np.float32)))
            
            output = torch.stack(output_images, dim=0)
            return (output,)
        
        # Mask is provided - process with masking logic
        mask = mask.cpu()
        
        batch_size = image_target.shape[0]
        _, H, W, C = image_target.shape
        
        # Ensure mask has correct batch size
        if mask.shape[0] == 1 and batch_size > 1:
            mask = mask.expand(batch_size, -1, -1)
        
        # Resize mask if dimensions don't match
        if mask.shape[1] != H or mask.shape[2] != W:
            mask = F.interpolate(
                mask.unsqueeze(1), 
                size=(H, W), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(1)
        
        # Apply feathering to mask
        if feather > 0:
            mask = self._gaussian_blur_mask(mask, feather)
            # Ensure values are still in 0-1 range after blur
            mask = torch.clamp(mask.squeeze(1) if mask.dim() == 4 else mask, 0, 1)
        
        # Ensure reference batch size matches or is 1
        if image_ref.shape[0] > 1 and image_ref.shape[0] != batch_size:
            raise ValueError(
                "ColorMatchMasked: Use either a single reference image or a matching batch of reference images."
            )
        
        cm = ColorMatcher()
        output_images = []
        
        for i in range(batch_size):
            # Get current images and mask
            target_np = image_target[i].numpy()
            ref_np = image_ref[0 if image_ref.shape[0] == 1 else i].numpy()
            mask_np = mask[i].numpy()
            
            # Create inverted mask (areas to include in color matching)
            inv_mask_np = 1.0 - mask_np
            
            # Expand mask for RGB channels
            inv_mask_rgb = np.expand_dims(inv_mask_np, axis=-1)
            mask_rgb = np.expand_dims(mask_np, axis=-1)
            
            # For the color transfer calculation, we need to extract just the 
            # unmasked pixels to avoid the masked region affecting the color stats
            
            # Create masked versions for color transfer calculation
            # We'll use the full image but the library will calculate stats from all pixels
            # To properly exclude masked areas, we need to work with pixel arrays
            
            # Get indices of unmasked pixels
            unmasked_indices = inv_mask_np > 0.5
            
            if np.sum(unmasked_indices) < 100:
                # Not enough unmasked pixels, return target unchanged
                output_images.append(torch.from_numpy(target_np))
                continue
            
            # Extract unmasked pixel values for color matching
            ref_pixels = ref_np[unmasked_indices]  # (N, 3)
            target_pixels = target_np[unmasked_indices]  # (N, 3)
            
            # Reshape to fake "image" for color-matcher (it expects image shape)
            # We'll create a 1D "image" of the pixels
            n_pixels = ref_pixels.shape[0]
            ref_fake_img = ref_pixels.reshape(1, n_pixels, 3)
            target_fake_img = target_pixels.reshape(1, n_pixels, 3)
            
            try:
                # Calculate color transfer on unmasked pixels only
                matched_pixels = cm.transfer(src=target_fake_img, ref=ref_fake_img, method=method)
                matched_pixels = matched_pixels.reshape(n_pixels, 3)
                
                # Create output image starting with target
                result = target_np.copy()
                
                # Apply color matched pixels to unmasked areas
                result[unmasked_indices] = matched_pixels
                
                # Blend based on strength
                if strength < 1.0:
                    result = target_np + strength * (result - target_np)
                
                # For feathered areas, blend smoothly between matched and original
                if feather > 0:
                    # Use the feathered mask for smooth blending
                    # inv_mask_rgb goes from 1 (apply correction) to 0 (keep original)
                    result = result * inv_mask_rgb + target_np * mask_rgb
                
                # Clamp to valid range
                result = np.clip(result, 0, 1)
                
            except Exception as e:
                print(f"ColorMatchMasked: Error during color transfer: {e}")
                result = target_np
            
            output_images.append(torch.from_numpy(result.astype(np.float32)))
        
        output = torch.stack(output_images, dim=0)
        return (output,)
