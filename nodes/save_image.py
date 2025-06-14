import os
import sys
import json
import numpy as np
from datetime import datetime
from pathlib import Path
import folder_paths

from PIL import Image
from PIL.PngImagePlugin import PngInfo



# class SaveImageExtended -------------------------------------------------------------------------------
class Y7Nodes_SaveImage:
 
    png_compress_level      = 9
    optimize_image          = True    
    filename_prefix         = 'ComfyUI'
    foldername_prefix       = ''
    counter_digits          = 4
    counter_position        = 'last'
    counter_positions_list  = ['last', 'first']
    image_preview           = True
    output_ext              = '.png'

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.prefix_append = ''
        

    @classmethod
    def INPUT_TYPES(self):
        return {
            'required': {
                'images': ('IMAGE', ),
                'filename_prefix': ('STRING', {'default': self.filename_prefix, 'multiline': False}),
                'foldername_prefix': ('STRING', {'default': self.foldername_prefix, 'multiline': False}),
                'counter_digits': ('INT', {
                    "default": self.counter_digits, 
                    "min": 1, 
                    "max": 8, 
                    "step": 1
                }),
                'counter_position': (self.counter_positions_list, {'default': self.counter_position}),
                'image_preview': ('BOOLEAN', {'default': self.image_preview}),
            },
            'hidden': {'prompt': 'PROMPT', 'extra_pnginfo': 'EXTRA_PNGINFO'},
        }
    
    RETURN_TYPES = ()
    FUNCTION = 'save_images'
    OUTPUT_NODE = True
    CATEGORY = 'image'
    DESCRIPTION = ""
    ### Sample datetime formats: see [man datetime](https://www.man7.org/linux/man-pages/man1/date.1.html)
    # - %F = %Y-%m-%d = 2024-05-22
    # - %H-%M-%S = 09-13-58
    # - %D = 05/22/24 (subfolders)


    def get_subfolder_path(self, image_path, output_path):
        image_path = Path(image_path).resolve()
        output_path = Path(output_path).resolve()
        relative_path = image_path.relative_to(output_path)
        subfolder_path = relative_path.parent
        
        return str(subfolder_path)
    
    
    # Get current counter number from file names
    def get_latest_counter(self, folder_path, filename_prefix, counter_digits=counter_digits, counter_position=counter_position, output_ext=output_ext):
        counter = 1
        if not os.path.exists(folder_path):
            print(f"Error: Folder {folder_path} does not exist, starting counter at 1.")
            return counter
        
        try:
            #  get a list of png files
            files = [file for file in os.listdir(folder_path) if file.endswith(output_ext)]
            extLen = len(output_ext)
            if files:
                if counter_position not in self.counter_positions_list: counter_position = self.counter_position
                if counter_position == 'last':
                    # BUG: this works only if extension is 3 letters like png, this will break with webp and avif:
                    counters = [int(file[-(extLen + counter_digits):-extLen]) if file[-(extLen + counter_digits):-extLen].isdecimal() else 0 for file in files]
                else:
                    counters = [int(file[:counter_digits]) if file[:counter_digits].isdecimal() else 0 for file in files]
                
                if counters:
                    counter = max(counters) + 1
        
        except Exception as e:
            print(f"Error: An error occurred while finding the latest counter: {e}")
        
        return counter
    
    
    def get_metadata_png(self, img, prompt, extra_pnginfo=None):
        metadata = PngInfo()
        if prompt is not None:
            metadata.add_text('prompt', json.dumps(prompt))
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata.add_text(x, json.dumps(extra_pnginfo[x]))
        
        return metadata
    
    
    def save_image(self, image_path, img, prompt, extra_pnginfo=None):
        kwargs = dict()
        

        kwargs["pnginfo"] = self.get_metadata_png(img, prompt, extra_pnginfo)
        kwargs["compress_level"] = self.png_compress_level
        kwargs["optimize"] = self.optimize_image
            
        img.save(image_path, **kwargs)

# class SaveImageExtended -------------------------------------------------------------------------------


    # node will never return None values, except for optional input. Impossible.
    def save_images(self,
            images,
            filename_prefix,
            foldername_prefix,
            counter_digits,
            counter_position,
            image_preview,
            extra_pnginfo=None,
            prompt=None,
        ):
        
        # apply default values: we replicate the default save image box
        if not filename_prefix: filename_prefix=self.filename_prefix
        
        # Convert forward and back slashes to hyphens in filename_prefix
        filename_prefix = filename_prefix.replace('/', '-').replace('\\', '-')
        
        timestamp = datetime.now()
        
        # Format foldername_prefix with datetime if it contains format specifiers
        if foldername_prefix and '%' in foldername_prefix:
            formatted_foldername_prefix = timestamp.strftime(foldername_prefix)
        else:
            formatted_foldername_prefix = foldername_prefix
        
        # Create folders, count images, save images
        try:
            full_output_folder, filename, _, _, custom_filename = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
            output_path = os.path.join(full_output_folder, formatted_foldername_prefix)
            # print(f"debug save_images: full_output_folder={full_output_folder}")
            # print(f"debug save_images: foldername_prefix={foldername_prefix}")
            # print(f"debug save_images: output_path={output_path}")
            os.makedirs(output_path, exist_ok=True)
            counter = self.get_latest_counter(output_path, filename, counter_digits, counter_position, self.output_ext)
            # print(f"debug save_images: counter for {self.output_ext}: {counter}")
        
            results = list()
            for image in images:
                i = 255. * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                
                if counter_position == 'last':
                    file = f'{filename}-{counter:0{counter_digits}}{self.output_ext}'
                else:
                    file = f'{counter:0{counter_digits}}-{filename}{self.output_ext}'
                
                image_path = os.path.join(output_path, file)
                self.save_image(image_path, img, prompt, extra_pnginfo)
                
                subfolder = self.get_subfolder_path(image_path, self.output_dir)
                # results.append({ 'filename': file, 'subfolder': subfolder, 'type': self.type})
                results.append({ 'filename': file, 'subfolder': subfolder})
                counter += 1
        
        except OSError as e:
            print(f"Error: An error occurred while creating the subfolder or saving the image: {e}")
        else:
            if not image_preview:
                results = list()
            return { 'ui': { 'images': results } }
