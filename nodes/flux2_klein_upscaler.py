# All-in-one tiled upscaler/refiner for Flux.2 Klein.
#
# Enlarges a picture and then repaints it a tile at a time, so you can go well past the size
# the model (or the graphics card) could manage in one go. Each tile is treated as a small
# inpaint: the middle of the tile is regenerated while a band of surrounding pixels is held
# frozen as context, so neighbouring tiles agree with each other instead of each inventing its
# own idea of what is there.
#
# Ported from ComfyUI_KleinTiledUpscaler by Gavr728 (MIT licence), commit e46c3b4:
#   https://github.com/Gavr728/ComfyUI_KleinTiledUpscaler
# The frozen-padding anchoring, the separable blend profiles and the shared canvas noise field
# are that project's ideas. What is done differently here, and why:
#
#   * All-in-one. Upstream needs CFGGuider + KSamplerSelect + Flux2Scheduler (and usually
#     SplitSigmasDenoise) wired in front of it. This node takes `model` and builds the guider,
#     the sampler and the sigma schedule itself, the same way Y7Nodes_Flux2Sampler does. Five
#     fewer nodes on the canvas, and the negative conditioning gets the same per-tile treatment
#     as the positive (upstream patches only the positive, which is wrong above cfg 1.0).
#
#   * Per-tile sigma schedule. Flux.2's schedule shift is a function of how many tokens are
#     being sampled, and a tile is not the whole canvas. Upstream builds one schedule from
#     Flux2Scheduler's hand-typed width/height - its own example workflow ships 1536x1534 while
#     sampling 1280px tiles on a 3584x2784 canvas - so the schedule is fitted to a resolution
#     that appears nowhere in the run. Here every tile gets a schedule built from its own latent
#     size, so it is always right.
#
#   * Partition-of-unity blending instead of sequential lerp + a mask_blur widget. Tile weights
#     are complementary raised-cosine ramps centred on each shared tile boundary. Because
#     (0.5 - 0.5cos(pi*t)) + (0.5 + 0.5cos(pi*t)) == 1 exactly, every pixel's weights across all
#     tiles sum to exactly 1.0, so seams are correct by construction and there is nothing to
#     tune. Upstream's blurred masks fall short of 1.0 in the seam band (about 3-5% of the
#     un-refined enlargement bleeds through), and its documented fix - raise mask_blur - makes
#     that worse rather than better.
#
#   * No full-canvas VAE encode. Upstream encodes the entire enlarged canvas in one call to
#     build the reference latent, which is the exact thing tiling exists to avoid: a 4096x4096
#     canvas asks for ~59GB, fails, and silently falls back to a tiled encode after evicting the
#     diffusion model. Here nothing bigger than a single tile is ever encoded or decoded.
#
#   * Tile layout rewritten. Upstream picks a column count, derives a tile width from it, then
#     recomputes the column count from that width - which for 76 of the 241 possible canvas
#     widths adds a phantom extra column (a 32px sliver promoted to a 256px tile: a whole extra
#     sampling pass for a near-duplicate of its neighbour), and for many others silently doubles
#     the tile area through Python's banker's rounding. Here the grid is derived once, the
#     remainder is spread evenly across the tiles, and the result is printed in the report.
#
#   * Per-tile sampler seed. Upstream passes the same seed to every tile, so with any ancestral
#     or SDE sampler each tile receives a byte-identical noise sequence and the same texture
#     repeats across the picture. Inert with plain euler, which is the only sampler it was
#     tested with.
#
#   * Colour matching through the color-matcher library (shared with Y7 Color Match (Masked))
#     rather than a naive per-channel mean/std transfer, and applied once per tile rather than
#     once per tile plus a second global pass that partly undoes the first.

import math

import numpy as np
import torch

import comfy.model_management
import comfy.samplers
import comfy.utils
import latent_preview
import node_helpers
from comfy_api.latest import io

from .color_match_masked import Y7Nodes_ColorMatchMasked
from .flux2_sampler import _flux2_sigma_schedule
from ..utils.logger import logger

# Colour transfer methods offered by the color-matcher library, plus an off switch. Same list as
# Y7 Color Match (Masked); "mkl" is the cheap, well-behaved default.
COLOR_MATCH_METHODS = ["off", "mkl", "hm", "reinhard", "mvgd", "hm-mvgd-hm", "hm-mkl-hm"]

# A tile is allowed to come out this much larger than the requested tile_size before the grid is
# given another row/column. Without a cap, an axis just under 1.5 tiles long would be sampled as a
# single tile half again as large as asked for - the difference between fitting in VRAM and not.
_MAX_TILE_OVERSHOOT = 1.25


# -----------------------------------------------------------------------------------------------
# Tile geometry
# -----------------------------------------------------------------------------------------------

def _plan_axis(length, target, align):
    """Split one axis into near-equal, `align`-aligned tile cores that exactly tile the axis.

    Returns the core boundaries, e.g. [0, 1024, 2048, 3040] for three cores. Work is done in units
    of `align` so every boundary lands on a latent-cell edge, and the leftover units are spread one
    per tile from the left rather than being dumped into a final runt tile. Adjacent cores
    therefore differ by at most `align` pixels, and there is never a sliver.
    """
    units = max(1, length // align)

    # Symmetric rounding, not Python's round(): round(2.5) is 2, which is how upstream turns a
    # 2560px axis into 1280px tiles instead of the 1024px ones it was asked for.
    n = max(1, int(math.floor(length / target + 0.5)))
    n = min(n, units)
    while n < units and (length / n) > target * _MAX_TILE_OVERSHOOT:
        n += 1

    base, rem = divmod(units, n)
    bounds = [0]
    for i in range(n):
        bounds.append(bounds[-1] + (base + (1 if i < rem else 0)) * align)
    return bounds


def _axis_ramp(bounds, overlap):
    """Half-width of the crossfade band, in pixels, shared by every tile on this axis.

    One value for the whole axis, not per tile: two neighbouring ramps only sum to 1.0 if they span
    exactly the same interval, and cores can differ in size by `align`. Capped at half the smallest
    core so a tile's two ramps can never overlap each other.
    """
    cores = [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)]
    return max(0, min(overlap // 2, min(cores) // 2))


def _axis_profile(length, bounds, i, ramp):
    """Blend weight along one axis for tile `i`, as a 1-D array over the whole canvas.

    1.0 across the tile's core, falling to 0.0 across a raised-cosine ramp centred on each shared
    boundary with a neighbour, and held at 1.0 all the way to the canvas edge where there is no
    neighbour. The rising and falling halves are exact complements, so summing this profile over
    every tile on the axis gives exactly 1.0 at every pixel.

    The full 2-D tile weight is the outer product of the x and y profiles. Two useful consequences:
    the 2-D mask never has to be built at canvas size, and average-pooling these same 1-D profiles
    by the VAE's downscale factor gives the latent-space weights - so the pixel and latent
    composites are identical geometry by construction rather than by coincidence.
    """
    n = len(bounds) - 1
    a, b = bounds[i], bounds[i + 1]

    p = np.zeros(length, dtype=np.float32)
    lo = a - ramp if i > 0 else 0
    hi = b + ramp if i < n - 1 else length
    p[lo:hi] = 1.0

    if ramp > 0:
        t = (np.arange(2 * ramp, dtype=np.float32) + 0.5) / (2 * ramp)
        if i > 0:
            p[a - ramp:a + ramp] = 0.5 - 0.5 * np.cos(np.pi * t)
        if i < n - 1:
            p[b - ramp:b + ramp] = 0.5 + 0.5 * np.cos(np.pi * t)
    return p, lo, hi


def _pool_profile(profile, divisor):
    """Average-pool a pixel-space profile down to latent resolution.

    Pooling is a mean, which is linear, so profiles that sum to 1.0 pixel-by-pixel still sum to 1.0
    cell-by-cell after pooling. The canvas is aligned to `divisor`, so nothing is truncated.
    """
    return profile.reshape(-1, divisor).mean(axis=1)


def _tile_crop(bounds, i, overlap, length, align):
    """Pixel range this tile is cropped from: its core plus `overlap` of context on each side.

    Centred on the core, then slid inwards where it would fall off the canvas so tiles keep a
    consistent size (extra context costs nothing and helps the model). Sliding is safe: it never
    moves the crop off the part of the canvas the tile's blend profile actually covers.
    """
    a, b = bounds[i], bounds[i + 1]
    want = min(int(math.ceil(((b - a) + 2 * overlap) / align) * align), length)

    c1 = a - (want - (b - a)) // 2
    c1 = (c1 // align) * align
    c1 = max(0, min(c1, length - want))
    return c1, c1 + want


# -----------------------------------------------------------------------------------------------
# Conditioning
# -----------------------------------------------------------------------------------------------

def _set_reference_latent(cond, ref):
    """Point the conditioning's reference latent at this tile instead of the whole picture.

    Replaces rather than appends: leaving a full-canvas reference alongside the tile-sized one
    gives the model two conflicting ideas of what it is looking at, which shows up as per-tile
    colour and tone drift. Any reference_latents_method already on the conditioning is left alone.
    """
    if not cond:
        return cond
    return node_helpers.conditioning_set_values(cond, {"reference_latents": [ref]})


def _has_reference(cond):
    for c in cond or []:
        if isinstance(c, (list, tuple)) and len(c) == 2 and isinstance(c[1], dict):
            if c[1].get("reference_latents"):
                return True
    return False


class Y7Nodes_Flux2KleinUpscaler(io.ComfyNode):
    """Tile-by-tile Flux.2 Klein upscaler with the whole sampling chain built in."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Y7Nodes_Flux2KleinUpscaler",
            display_name="Y7 Flux.2 Klein Upscaler (Tiled)",
            category="Y7Nodes/Klein",
            description="Enlarges a picture and repaints it one tile at a time with Flux.2 Klein, "
                        "so you can go far past the size the model could handle in one pass. The "
                        "sampler, scheduler and guider are built in - connect the model, the VAE, "
                        "your prompt conditioning and an image.",
            inputs=[
                # The Flux.2 Klein diffusion model. Taken as a MODEL rather than a GUIDER so this
                # node can build a fresh guider per tile: that is what lets it give the negative
                # conditioning the same tile-local reference the positive gets, and it means the
                # node never mutates a guider object belonging to another node.
                io.Model.Input("model"),
                # The picture to enlarge. A batch is processed one image at a time.
                io.Image.Input("image", tooltip="The picture to enlarge and refine."),
                # Used for every tile encode and decode. Nothing larger than one tile is ever put
                # through it, which is what keeps VRAM flat as the canvas grows.
                io.Vae.Input("vae"),
                # Prompt conditioning. Klein is an edit model, so the positive conditioning also
                # carries the reference latent that tells the model what it is looking at - this
                # node swaps that for a crop matching each tile as it goes.
                io.Conditioning.Input(
                    "positive",
                    tooltip="Your prompt. Keep it a short description of the picture as a whole, "
                            "or leave it empty - it is applied to every tile, so anything naming a "
                            "specific object will try to put that object in all of them.",
                ),
                io.Conditioning.Input(
                    "negative",
                    tooltip="What to steer away from. Klein ignores this unless `cfg` is above 1.0.",
                ),
                io.Int.Input(
                    "seed", default=0, min=0, max=0xffffffffffffffff,
                    control_after_generate=io.ControlAfterGenerate.randomize,
                ),
                # Target enlargement. The canvas is always resampled to exactly this multiple of the
                # input (aligned to the VAE grid), whether or not an upscale_model is connected.
                # 1.0 is a supported and useful setting: no enlargement, but every tile is still
                # repainted, which turns the node into a plain sharpen/refine pass.
                io.Float.Input(
                    "scale_factor", default=2.0, min=1.0, max=8.0, step=0.25,
                    tooltip="How much bigger to make the picture. 2.0 doubles the width and height. "
                            "Set it to 1.0 to sharpen or add detail WITHOUT making the picture "
                            "bigger - the size stays the same and every tile is repainted at its "
                            "original scale. With 1.0, leave upscale_model unplugged and use "
                            "refine_strength (0.85-0.92) to control how much changes.",
                ),
                # The size each tile is sampled at. This, not the canvas size, is what determines
                # peak VRAM, so it is the knob to reach for when a run will not fit.
                io.Int.Input(
                    "tile_size", default=1024, min=256, max=2048, step=16,
                    tooltip="Roughly how big each tile is, in pixels. This is what decides how much "
                            "graphics memory the job needs - the finished picture can be any size. "
                            "Turn it down if you run out of memory, up if tiles are inventing "
                            "detail that does not belong. The exact size is adjusted so the tiles "
                            "divide the picture evenly; the report output shows what was used.",
                ),
                # Context band around each tile. Half of it becomes the crossfade with the
                # neighbouring tile; the rest is frozen context the model can see but not change.
                io.Int.Input(
                    "overlap", default=128, min=16, max=512, step=16,
                    tooltip="How far each tile looks past its own edges, in pixels. This is how a "
                            "tile knows what its neighbours contain, so raising it helps things "
                            "that run across a seam (a roofline, a horizon) stay joined up. Costs "
                            "memory and time. 128 suits most pictures.",
                ),
                io.Int.Input(
                    "steps", default=4, min=1, max=4096,
                    tooltip="Sampling steps per tile. Distilled Klein checkpoints usually want 4.",
                ),
                io.Float.Input(
                    "cfg", default=1.0, min=0.0, max=100.0, step=0.1, round=0.01,
                    tooltip="Klein checkpoints are normally guidance-distilled, so leave this at "
                            "1.0 unless you know your checkpoint wants otherwise. Above 1.0 the "
                            "negative prompt starts being used, and every tile costs twice as much.",
                ),
                io.Combo.Input("sampler_name", options=comfy.samplers.SAMPLER_NAMES, default="euler"),
                # Lower values keep more of the enlarged original. Unlike upstream this is a real
                # widget rather than a SplitSigmasDenoise node the user has to remember to connect.
                io.Float.Input(
                    "denoise", default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip="How much each tile is allowed to change. 1.0 repaints it completely, "
                            "which is what you normally want here. Lower values keep more of the "
                            "enlarged original, but they also spend fewer steps, so at 4 steps "
                            "there is very little room between the settings.",
                ),
                io.Combo.Input(
                    "color_match", options=COLOR_MATCH_METHODS, default="mkl",
                    tooltip="Nudges each finished tile back towards the colours of the enlarged "
                            "original, so tiles do not drift apart in tone. `mkl` is a good "
                            "default; `off` leaves the model's own colours alone.",
                ),
                # --- advanced ---
                # Becomes the denoise mask amplitude over the part of the tile that actually reaches
                # the output (its core plus the crossfade ramps). Below 1.0, ComfyUI mixes the
                # un-refined latent back in at every sampler step, so this restrains the model as it
                # works rather than fading the finished tile afterwards - which is what makes it a
                # better detail dial than `denoise` or a downstream blend. Upstream calls this
                # core_anchor, where turning it up meant less anchoring.
                io.Float.Input(
                    "refine_strength", default=1.0, min=0.5, max=1.0, step=0.01, advanced=True,
                    tooltip="How much each tile is allowed to change. 1.0 repaints it completely. "
                            "Lower values hold on to the shapes already there, at every step rather "
                            "than just at the end, which reins in invented detail and calms an "
                            "over-sharp result. 0.92 is a good first try; 0.85 is strongly held "
                            "back.",
                ),
                # Where the per-tile reference latent - the thing that actually tells Klein what the
                # picture contains at cfg 1.0 - is cropped from. "enlarged" matches upstream: always
                # the enlargement, never updated. "progressive" reads back the tile's own context
                # including finished neighbours, which is more consistent but can compound drift
                # across a long run. Progressive is also one VAE encode per tile cheaper.
                io.Combo.Input(
                    "reference_source", options=["enlarged", "progressive"], default="enlarged",
                    advanced=True,
                    tooltip="What each tile is told it is looking at. `enlarged` always shows the "
                            "model the plain enlargement, which is predictable. `progressive` shows "
                            "it the tile's surroundings as they stand, including tiles already "
                            "finished - more consistent between neighbours, but mistakes can build "
                            "up across a big picture. Try `enlarged` first.",
                ),
                # Optional ESRGAN-style upscaler. Whatever it produces is resampled to exactly
                # scale_factor afterwards, so a 4x model at scale_factor 2 is supersampling (good),
                # while a 2x model at scale_factor 4 leaves half the job to plain bicubic.
                io.UpscaleModel.Input(
                    "upscale_model", optional=True,
                    tooltip="Optional. An ordinary upscaler (ESRGAN and friends) to enlarge with "
                            "before the tiles are repainted. Without one, plain bicubic is used. "
                            "Pick a clean, neutral model rather than a sharp one - the enlargement "
                            "becomes the frozen context and the reference every tile works from, so "
                            "halos and invented texture get treated as real detail and copied "
                            "rather than fixed. `4xRealWebPhoto_v4_dat2`, `4xNomosUniDAT_otf` and "
                            "`RealESRGAN_x4plus` are good; anything with \"Sharp\" in the name is "
                            "not. Its result is always resized to exactly `scale_factor`, so a 4x "
                            "model at scale_factor 2 is fine (better, even) but a 2x model at "
                            "scale_factor 4 leaves half the enlargement to bicubic.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                # Same picture in latent form, blended with the identical geometry, so it can be fed
                # onward without a re-encode.
                io.Latent.Output(display_name="latent"),
                # What the node actually did: canvas size, the grid it chose, tile and crop sizes,
                # and a check that the blend weights really do sum to 1.
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, model, image, vae, positive, negative, seed, scale_factor, tile_size,
                overlap, steps, cfg, sampler_name, denoise, color_match, refine_strength,
                reference_source, upscale_model=None) -> io.NodeOutput:

        # The VAE's own downscale factor is the alignment grid everything else is built on: tile
        # boundaries and crop edges land on whole latent cells, so pixel and latent compositing use
        # the same geometry and the shared noise field can be cropped without interpolation.
        align = int(vae.spacial_compression_encode())
        latent_channels = int(vae.latent_channels)

        sampler = comfy.samplers.sampler_object(sampler_name)
        guider = comfy.samplers.CFGGuider(model)
        guider.set_cfg(cfg)

        run_steps = max(1, round(steps * denoise)) if denoise < 1.0 else steps

        if _has_reference(positive):
            logger.info("[Y7 Klein Upscaler] positive conditioning already carries a reference "
                        "latent; it will be replaced per tile with a crop matching that tile.")

        matcher = Y7Nodes_ColorMatchMasked() if color_match != "off" else None
        previewer = latent_preview.get_previewer(model.load_device, model.model.latent_format)

        out_images, out_latents, report_lines = [], [], []

        for b in range(image.shape[0]):
            src = image[b:b + 1]

            # --- enlarge -------------------------------------------------------------------
            if upscale_model is not None:
                import comfy_extras.nodes_upscale_model as upscale_nodes
                enlarged = upscale_nodes.ImageUpscaleWithModel.execute(upscale_model, src)[0]
            else:
                enlarged = src

            canvas_w = max(align, int(math.floor(src.shape[2] * scale_factor / align + 0.5)) * align)
            canvas_h = max(align, int(math.floor(src.shape[1] * scale_factor / align + 0.5)) * align)

            enlarged = enlarged.movedim(-1, 1).float()
            enlarged = torch.nn.functional.interpolate(
                enlarged, size=(canvas_h, canvas_w), mode="bicubic", antialias=True)
            enlarged = enlarged.clamp(0.0, 1.0).movedim(1, -1)
            enlarged = enlarged.detach().to("cpu", dtype=torch.float32)[:, :, :, :3].contiguous()

            # --- grid ----------------------------------------------------------------------
            xb = _plan_axis(canvas_w, tile_size, align)
            yb = _plan_axis(canvas_h, tile_size, align)
            x_ramp, y_ramp = _axis_ramp(xb, overlap), _axis_ramp(yb, overlap)
            cols, rows = len(xb) - 1, len(yb) - 1

            # One profile per row/column, reused by every tile in it. The 2-D weight for a tile is
            # the outer product of its two profiles, taken only over its own crop.
            xp = [_axis_profile(canvas_w, xb, i, x_ramp) for i in range(cols)]
            yp = [_axis_profile(canvas_h, yb, j, y_ramp) for j in range(rows)]
            xl = [_pool_profile(p, align) for p, _, _ in xp]
            yl = [_pool_profile(p, align) for p, _, _ in yp]

            # Self-check: the whole point of the ramp construction is that these sum to exactly 1.
            x_err = float(np.abs(sum(p for p, _, _ in xp) - 1.0).max())
            y_err = float(np.abs(sum(p for p, _, _ in yp) - 1.0).max())

            # Chess order: no tile is sampled next to one still being worked on, so a tile's
            # context is either finished work or untouched enlargement, never something half done.
            tiles = [(i, j) for j in range(rows) for i in range(cols)]
            tiles = [t for t in tiles if (t[0] + t[1]) % 2 == 0] + \
                    [t for t in tiles if (t[0] + t[1]) % 2 == 1]

            # --- accumulators --------------------------------------------------------------
            # Weighted sums rather than a canvas painted over in place. Dividing at the end (or,
            # equivalently here, letting the weights reach 1.0) is what makes the seams exact.
            # Both live on the CPU: only one tile is ever on the GPU, which is the real reason
            # this works on a small card.
            lat_h, lat_w = canvas_h // align, canvas_w // align
            accum = torch.zeros_like(enlarged)
            wsum = torch.zeros((1, canvas_h, canvas_w, 1), dtype=torch.float32)
            lat_accum = torch.zeros((1, latent_channels, lat_h, lat_w), dtype=torch.float32)
            lat_wsum = torch.zeros((1, 1, lat_h, lat_w), dtype=torch.float32)

            # One noise field for the whole canvas; each tile crops its own window out of it, so
            # overlapping tiles start from identical noise where they overlap. Cheap insurance
            # against ghosting, and it costs one latent-sized CPU tensor.
            g = torch.Generator(device="cpu").manual_seed(seed)
            noise_field = torch.randn((1, latent_channels, lat_h, lat_w), generator=g,
                                      dtype=torch.float32)

            logger.info(f"[Y7 Klein Upscaler] image {b + 1}/{image.shape[0]}: "
                        f"{src.shape[2]}x{src.shape[1]} -> {canvas_w}x{canvas_h}, "
                        f"grid {cols}x{rows} = {cols * rows} tiles")

            pbar = comfy.utils.ProgressBar(cols * rows * run_steps)
            crop_note = ""

            for n, (i, j) in enumerate(tiles):
                comfy.model_management.throw_exception_if_processing_interrupted()

                cx1, cx2 = _tile_crop(xb, i, overlap, canvas_w, align)
                cy1, cy2 = _tile_crop(yb, j, overlap, canvas_h, align)
                _, x_lo, x_hi = xp[i]
                _, y_lo, y_hi = yp[j]

                # The tile's context: finished neighbours where there are any, the plain
                # enlargement everywhere else. wsum never exceeds 1, so this is a straight blend.
                deficit = (1.0 - wsum[:, cy1:cy2, cx1:cx2, :]).clamp(0.0, 1.0)
                tile_px = accum[:, cy1:cy2, cx1:cx2, :] + enlarged[:, cy1:cy2, cx1:cx2, :] * deficit

                init_latent = vae.encode(tile_px)

                # What Klein is told the tile depicts. At cfg 1.0 this is the structural signal, so
                # it matters more than the starting latent does.
                if reference_source == "progressive":
                    ref_latent = init_latent
                else:
                    ref_latent = vae.encode(enlarged[:, cy1:cy2, cx1:cx2, :])

                tl_h, tl_w = init_latent.shape[2], init_latent.shape[3]

                # Denoise only what this tile actually contributes: its core plus its crossfade
                # ramps. Everything outside is pinned to the context pixels at every sampler step
                # (ComfyUI blends latent_image back in wherever the mask is 0), which is what makes
                # a tile line up with the neighbour it was cropped against.
                mx1 = max(0, (x_lo - cx1) // align)
                my1 = max(0, (y_lo - cy1) // align)
                mx2 = min(tl_w, int(math.ceil((x_hi - cx1) / align)))
                my2 = min(tl_h, int(math.ceil((y_hi - cy1) / align)))
                noise_mask = torch.zeros((1, 1, tl_h, tl_w), dtype=torch.float32)
                noise_mask[0, 0, my1:my2, mx1:mx2] = refine_strength

                noise = noise_field[:, :, cy1 // align:cy1 // align + tl_h,
                                    cx1 // align:cx1 // align + tl_w].clone()

                # Flux.2's schedule shift depends on the token count of what is being sampled, and
                # this tile is what is being sampled - not the canvas, and not a number typed into
                # a scheduler node somewhere upstream.
                sigmas = _flux2_sigma_schedule(steps, tl_h * tl_w)
                if run_steps < steps:
                    sigmas = sigmas[-(run_steps + 1):]

                base = n * run_steps

                def callback(step, x0, x, total_steps, _base=base):
                    preview = None
                    if previewer:
                        if x0.is_nested:
                            x0 = x0.tensors[0]
                        preview = previewer.decode_latent_to_preview_image("JPEG", x0)
                    pbar.update_absolute(_base + step + 1, cols * rows * run_steps, preview)

                guider.set_conds(_set_reference_latent(positive, ref_latent),
                                 _set_reference_latent(negative, ref_latent))

                # Per-tile seed. With the same seed everywhere, any ancestral or SDE sampler draws
                # the identical noise sequence for every tile and stamps the same texture across
                # the picture.
                samples = guider.sample(noise, init_latent, sampler, sigmas,
                                        denoise_mask=noise_mask, callback=callback,
                                        disable_pbar=True, seed=seed + n)
                samples = samples.to(comfy.model_management.intermediate_device())

                decoded = vae.decode(samples).detach().to("cpu", dtype=torch.float32)
                decoded = decoded[:, :, :, :3].clamp(0.0, 1.0)

                if decoded.shape[1] != (cy2 - cy1) or decoded.shape[2] != (cx2 - cx1):
                    # Should not happen - crops are whole numbers of latent cells - but if a VAE
                    # ever rounds differently, say so rather than blending a misaligned tile.
                    crop_note = (f"  WARNING: tile ({i},{j}) decoded to "
                                 f"{decoded.shape[2]}x{decoded.shape[1]}, expected "
                                 f"{cx2 - cx1}x{cy2 - cy1}\n")
                    logger.warning("[Y7 Klein Upscaler] " + crop_note.strip())
                    ch = min(decoded.shape[1], cy2 - cy1)
                    cw = min(decoded.shape[2], cx2 - cx1)
                    decoded = decoded[:, :ch, :cw, :]
                    cy2, cx2 = cy1 + ch, cx1 + cw

                if matcher is not None:
                    decoded = matcher.color_match_masked(
                        image_ref=enlarged[:, cy1:cy2, cx1:cx2, :],
                        image_target=decoded,
                        method=color_match,
                    )[0].clamp(0.0, 1.0)

                # --- composite -----------------------------------------------------------
                w2d = torch.from_numpy(
                    np.outer(yp[j][0][cy1:cy2], xp[i][0][cx1:cx2])).unsqueeze(0).unsqueeze(-1)
                accum[:, cy1:cy2, cx1:cx2, :] += decoded * w2d
                wsum[:, cy1:cy2, cx1:cx2, :] += w2d

                lx1, ly1 = cx1 // align, cy1 // align
                lw2d = torch.from_numpy(
                    np.outer(yl[j][ly1:ly1 + tl_h], xl[i][lx1:lx1 + tl_w])).unsqueeze(0).unsqueeze(0)
                lat_accum[:, :, ly1:ly1 + tl_h, lx1:lx1 + tl_w] += samples.cpu().float() * lw2d
                lat_wsum[:, :, ly1:ly1 + tl_h, lx1:lx1 + tl_w] += lw2d

            # Weights sum to 1 everywhere, so the deficit term is zero here; it is kept so that an
            # interrupted or unexpectedly short run still returns something sensible.
            final = accum + enlarged * (1.0 - wsum).clamp(0.0, 1.0)
            out_images.append(final.clamp(0.0, 1.0))
            out_latents.append(lat_accum / lat_wsum.clamp(min=1e-6))

            tile_ws = [xb[i + 1] - xb[i] for i in range(cols)]
            tile_hs = [yb[j + 1] - yb[j] for j in range(rows)]
            report_lines.append(
                f"image {b + 1}/{image.shape[0]}\n"
                f"  in:      {src.shape[2]}x{src.shape[1]}\n"
                f"  canvas:  {canvas_w}x{canvas_h}  (scale {scale_factor}, aligned to {align}px)\n"
                f"  grid:    {cols} x {rows} = {cols * rows} tiles, chess order\n"
                f"  cores:   widths {sorted(set(tile_ws))}, heights {sorted(set(tile_hs))}\n"
                f"  crops:   up to {min(max(tile_ws) + 2 * overlap, canvas_w)}"
                f"x{min(max(tile_hs) + 2 * overlap, canvas_h)} (core + {overlap}px context each side)\n"
                f"  blend:   {2 * x_ramp}px wide across vertical seams, {2 * y_ramp}px across horizontal\n"
                f"  weights: sum to 1.0 +/- {max(x_err, y_err):.2e}\n"
                f"  steps:   {run_steps}/{steps} per tile ({sampler_name}, cfg {cfg}, denoise {denoise})\n"
                f"  ref:     {reference_source}, refine_strength {refine_strength}, colour {color_match}\n"
                + crop_note
            )
            comfy.model_management.soft_empty_cache()

        return io.NodeOutput(
            torch.cat(out_images, dim=0),
            {"samples": torch.cat(out_latents, dim=0)},
            "\n".join(report_lines),
        )
