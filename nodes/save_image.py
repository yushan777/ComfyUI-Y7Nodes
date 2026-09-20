import json
import os
import re
import struct
import zlib
from datetime import datetime

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import folder_paths
from comfy.cli_args import args
from comfy_api.latest import io, ui

COMPRESS_LEVEL = 4  # zlib level, same as the native Save Image

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_COLOR_TYPES = {1: 0, 2: 4, 3: 2, 4: 6}  # channel count -> PNG colour type (grey, grey+alpha, RGB, RGBA)
ROWS_PER_BLOCK = 256  # 16-bit rows are filtered and compressed this many at a time to bound memory

# The counter at the end of a saved name: this node's name_0001.png or the native node's name_00001_.png
COUNTER_PATTERN = re.compile(r"_(\d+)_?\.png$", re.IGNORECASE)

# %date:FORMAT% is normally filled in by the browser before the prompt is sent, so a prompt from a
# script or hand-edited API workflow can arrive with it still in place. These mirror the frontend's
# own parsing so the server fills it in the same way. Formats containing a dot are left alone:
# the frontend reads a dot as %Node.widget%, so those don't work in the browser either.
DATE_TOKEN = re.compile(r"%date:([^%.]*)%")
DATE_CODES = re.compile(r"dd?|MM?|hh?|mm?|ss?|yyy?y?")


class Y7Nodes_SaveImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Y7Nodes_SaveImage",
            display_name="Y7 Save Image",
            category="Y7Nodes/Image",
            search_aliases=["save", "save image", "export image", "output image", "write image", "16-bit"],
            inputs=[
                io.Image.Input("images", tooltip="The images to save."),
                io.Boolean.Input(
                    "save_or_preview",
                    default=True,
                    label_on="save",
                    label_off="preview",
                    tooltip="save: write the images to the output directory. preview: only show them on the node, like Preview Image; they go to the temp directory, which ComfyUI clears on restart, and the settings below are ignored.",
                ),
                io.String.Input(
                    "sub_folder",
                    default="foldername/%date:yyyy-MM-dd%",
                    tooltip="The folder inside the output directory to save to, created if it doesn't exist. Leave empty to save to the output directory itself. Accepts the same formatting as filename_prefix.",
                ),
                io.String.Input(
                    "filename_prefix",
                    default="img_%date:hhmmss%",
                    tooltip="The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes.",
                ),
                io.Int.Input(
                    "counter_digits",
                    default=4,
                    min=1,
                    max=10,
                    tooltip="How many digits the counter at the end of the file name is padded to, e.g. 4 gives img_0001.png.",
                ),
                io.Boolean.Input(
                    "counter_per_folder",
                    default=True,
                    tooltip="On: one counter shared by every image in the folder, whatever its prefix. Off: a separate counter for each prefix, as in the native Save Image.",
                ),
                io.Combo.Input(
                    "bit_depth",
                    options=["8-bit", "16-bit"],
                    default="8-bit",
                    tooltip="8-bit matches the native Save Image. 16-bit keeps the finer tonal steps the image carries, at roughly 4-5x the file size.",
                ),
            ],
            outputs=[
                io.Image.Output("images"),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, images, save_or_preview: bool, sub_folder: str, filename_prefix: str, counter_digits: int,
                counter_per_folder: bool, bit_depth: str) -> io.NodeOutput:
        if not save_or_preview:
            return io.NodeOutput(images, ui=ui.PreviewImage(images, cls=cls))

        now = datetime.now()  # one timestamp, so the folder and file names agree
        sub_folder = _replace_date_tokens(sub_folder, now)
        filename_prefix = _replace_date_tokens(filename_prefix, now)

        # Leading/trailing slashes are dropped so "/renders" means output/renders, not an absolute path
        sub_folder = sub_folder.strip().strip("/\\")
        if sub_folder:
            filename_prefix = f"{sub_folder}/{filename_prefix}"

        output_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, output_dir, images[0].shape[1], images[0].shape[0]
        )
        if counter_per_folder:
            counter = _next_counter_in_folder(full_output_folder)
        elif "%batch_num%" in filename:
            # ComfyUI scans for the literal "%batch_num%" prefix, finds nothing and restarts at 1 every
            # run, overwriting the last run's files. Scan each image's real prefix and start past them all.
            counter = max(
                folder_paths.get_save_image_path(
                    os.path.join(subfolder, filename.replace("%batch_num%", str(batch_number))), output_dir
                )[2]
                for batch_number in range(len(images))
            )
        text_chunks = _metadata_text(cls)

        results = []
        for batch_number, image in enumerate(images):
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            # No trailing underscore: the native node's name_00001_.png only kept it for a parse
            # ComfyUI no longer does, and its counter scan reads both forms
            file = f"{filename_with_batch_num}_{counter:0{counter_digits}}.png"
            path = os.path.join(full_output_folder, file)

            if bit_depth == "16-bit":
                _save_png16(image, path, text_chunks)
            else:
                metadata = PngInfo()
                for key, text in text_chunks:
                    metadata.add_text(key, text)
                img = Image.fromarray(np.clip(255. * image.cpu().numpy(), 0, 255).astype(np.uint8))
                img.save(path, pnginfo=metadata, compress_level=COMPRESS_LEVEL)

            results.append(ui.SavedResult(file, subfolder, io.FolderType.output))
            counter += 1

        return io.NodeOutput(images, ui=ui.SavedImages(results))


def _replace_date_tokens(text, now):
    """Fill in any %date:FORMAT% the browser didn't, using the server's clock"""
    fields = {"d": now.day, "M": now.month, "h": now.hour, "m": now.minute, "s": now.second}

    def format_code(match):
        code = match.group(0)
        if code == "yy":
            return str(now.year)[2:]
        if code == "yyyy":
            return str(now.year)
        if code[0] in fields:
            return str(fields[code[0]]).zfill(len(code))
        return code  # "yyy", which the frontend also leaves as is

    return DATE_TOKEN.sub(lambda match: DATE_CODES.sub(format_code, match.group(1)), text)


def _next_counter_in_folder(folder):
    """One past the highest counter on any PNG in the folder, whatever its prefix"""
    counters = [int(match.group(1)) for match in map(COUNTER_PATTERN.search, os.listdir(folder)) if match]
    return max(counters, default=0) + 1


def _metadata_text(cls):
    """(keyword, text) pairs for the prompt and workflow, the same ones the native node embeds"""
    if args.disable_metadata:
        return []
    text_chunks = []
    if cls.hidden.prompt is not None:
        text_chunks.append(("prompt", json.dumps(cls.hidden.prompt)))
    if cls.hidden.extra_pnginfo is not None:
        for key, value in cls.hidden.extra_pnginfo.items():
            text_chunks.append((key, json.dumps(value)))
    return text_chunks


def _png_chunk(chunk_type, data):
    """length | type | data | CRC32 of type + data"""
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", zlib.crc32(chunk_type + data))


def _paeth_filter(rows, row_above, bpp):
    """Apply PNG filter type 4 (Paeth) to every row, prefixing each with its filter-type byte.

    Pillow can't write 16-bit RGB, so the 16-bit PNG is assembled by hand. Paeth on every row
    compresses as well as libpng-style per-row filter selection on generated images, for less code.
    """
    above = np.vstack([row_above[None], rows[:-1]])
    left = np.zeros_like(rows)
    left[:, bpp:] = rows[:, :-bpp]
    upper_left = np.zeros_like(rows)
    upper_left[:, bpp:] = above[:, :-bpp]

    a, b, c = (x.astype(np.int16) for x in (left, above, upper_left))
    p = a + b - c
    pa, pb, pc = np.abs(p - a), np.abs(p - b), np.abs(p - c)
    predictor = np.where((pa <= pb) & (pa <= pc), a, np.where(pb <= pc, b, c)).astype(np.uint8)

    filtered = rows - predictor  # uint8 wraparound gives the modulo-256 difference PNG expects
    return np.hstack([np.full((len(rows), 1), 4, dtype=np.uint8), filtered])


def _save_png16(image, path, text_chunks):
    pixels = np.clip(65535. * image.cpu().numpy(), 0, 65535).astype(np.uint16)
    if pixels.ndim == 2:
        pixels = pixels[..., None]
    height, width, channels = pixels.shape
    if channels not in PNG_COLOR_TYPES:
        raise ValueError(f"Can't save a {channels}-channel image as PNG: expected 1, 2, 3 or 4 channels.")

    bpp = channels * 2
    rows = np.ascontiguousarray(pixels, dtype=">u2").view(np.uint8).reshape(height, width * bpp)  # PNG is big-endian

    compressor = zlib.compressobj(COMPRESS_LEVEL)
    idat = []
    row_above = np.zeros(width * bpp, dtype=np.uint8)
    for start in range(0, height, ROWS_PER_BLOCK):
        block = rows[start:start + ROWS_PER_BLOCK]
        idat.append(compressor.compress(_paeth_filter(block, row_above, bpp).tobytes()))
        row_above = block[-1]
    idat.append(compressor.flush())

    ihdr = struct.pack(">IIBBBBB", width, height, 16, PNG_COLOR_TYPES[channels], 0, 0, 0)
    with open(path, "wb") as f:
        f.write(PNG_SIGNATURE)
        f.write(_png_chunk(b"IHDR", ihdr))
        for key, text in text_chunks:  # before IDAT, where Pillow puts them for 8-bit
            f.write(_png_chunk(b"tEXt", key.encode("latin-1") + b"\0" + text.encode("latin-1")))
        f.write(_png_chunk(b"IDAT", b"".join(idat)))
        f.write(_png_chunk(b"IEND", b""))
