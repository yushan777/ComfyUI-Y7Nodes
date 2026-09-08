# Flux.2 Klein Upscaler (Tiled)

Enlarges a picture and repaints it one tile at a time with Flux.2 Klein, so you can go far past the size the model could handle in a single pass. The sampler, scheduler and guider are built in.

Each tile is treated as a small inpaint. The middle of the tile is regenerated while a band of surrounding pixels is held frozen as context, so a tile can see what its neighbours contain and matches up with them instead of inventing its own version of the scene. Only one tile is ever on the graphics card, so the memory a job needs is set by `tile_size`, not by how big the finished picture is.

Feed it a model, a VAE, your conditioning and an image. Nothing else needs wiring.

Inputs:

  - `model`: The Flux.2 Klein diffusion model
  - `image`: The picture to enlarge and refine. A batch is processed one image at a time
  - `vae`: Used to encode and decode each tile. Nothing bigger than one tile ever passes through it
  - `positive`: Your prompt. Keep it a short description of the picture as a whole, or leave it empty — the same prompt is applied to every tile, so anything naming a specific object will try to put that object in all of them
  - `negative`: What to steer away from. Ignored unless `cfg` is above `1.0`
  - `seed`: Seed for the noise. Each tile derives its own seed from this one
  - `scale_factor`: How much bigger to make the picture. `2.0` doubles the width and height. Set it to `1.0` to sharpen or refine without enlarging at all — see "Sharpening without upscaling" below
  - `tile_size`: Roughly how big each tile is, in pixels. This is what decides how much graphics memory the job needs. Turn it down if you run out of memory, up if tiles are inventing detail that doesn't belong. The exact size is adjusted so the tiles divide the picture evenly — the `report` output shows what was used
  - `overlap`: How far each tile looks past its own edges, in pixels. This is how a tile knows what its neighbours contain, so raising it helps things that run across a seam (a roofline, a horizon) stay joined up. Costs memory and time. `128` suits most pictures
  - `steps`: Sampling steps per tile. Distilled Klein checkpoints usually want `4`
  - `cfg`: Leave at `1.0` for guidance-distilled Klein checkpoints. Above `1.0` the negative prompt starts being used and every tile costs twice as much
  - `sampler_name`: Which sampler algorithm to step with
  - `denoise`: How much each tile is allowed to change. `1.0` repaints it completely, which is normally what you want here
  - `color_match`: Nudges each finished tile back towards the colours of the enlarged original so tiles don't drift apart in tone. `mkl` is a good default; `off` leaves the model's own colours alone
  - `refine_strength` (advanced): How much each tile is allowed to change. `1.0` repaints it completely; lower values hold on to the shapes already there. This is the dial for an over-sharp or over-detailed result - see below. `0.92` is a good first try, `0.85` is strongly held back
  - `reference_source` (advanced): What each tile is told it is looking at. `enlarged` always shows the model the plain enlargement, which is predictable. `progressive` shows it the surroundings as they stand, including tiles already finished — more consistent between neighbours, but mistakes can build up across a big picture. Try `enlarged` first
  - `upscale_model` (optional): An ordinary upscaler (ESRGAN and friends) to enlarge with before the tiles are repainted. Without one, plain bicubic is used

Outputs:

  - `image`: The finished picture
  - `latent`: The same picture in latent form, blended with identical geometry, so it can be fed onward without a re-encode
  - `report`: What the node actually did — canvas size, the grid it chose, tile and crop sizes, blend widths and settings. Wire it into `Y7 Show Anything` when something looks wrong

## Sharpening without upscaling [collapsed]

For a plain sharpen, refine or detail pass — the picture stays exactly the size it already is — set `scale_factor` to `1.0`.

Nothing is enlarged, but every tile is still repainted, so the model puts detail back into a soft, flat or over-denoised picture at its original size. This is the setting to use for cleaning up a picture you are already happy with the size of, or for a second pass over something this node has already upscaled.

What changes when you do this:

  - **Unplug `upscale_model`.** It has nothing to do at `1.0`: whatever it produces is resampled straight back down to the original size, so you pay for it and get its halos in the reference for nothing. Leave the socket empty
  - **`refine_strength` becomes the main dial.** With no enlargement there is no missing resolution to fill in, so at `1.0` the model is free to re-invent detail that was never there. Start at `0.90` and work down towards `0.85` if the result looks synthetic, or up towards `0.95` for a stronger pass
  - **`tile_size` decides how fine the new detail is.** Each tile is repainted at its own scale, so smaller tiles cover less of the picture each and add finer, denser detail. `1024` is a good start; go up to `1536` if the result is too busy
  - **`overlap` and `color_match` work the same as ever.** `128` and `mkl` are still sensible defaults

A `1.0` pass is much cheaper than a `2.0` one — the canvas is a quarter of the area, so roughly a quarter of the tiles.

## Choosing an upscale model [collapsed]

Pick a clean, neutral one - not a sharp one. This is the opposite of how you would normally choose an upscaler, and it is worth understanding why.

The enlargement is not the finished picture. It is three other things:

  - what each tile is told it is looking at (the reference the model works from)
  - the frozen band of context around every tile, which the model is not allowed to change
  - the colour that `color_match` pulls each finished tile towards

So halos, ringing and invented texture are actively harmful here. They get frozen into the context band and encoded into the reference, and the model then treats them as the truth of the scene and matches them rather than fixing them. Softness costs nothing by comparison, because putting the detail back is the whole job of the node.

Good choices, roughly in order:

  - `4xRealWebPhoto_v4_dat2` - trained for real-world photos that have been through JPEG and resizing. Cleans up a degraded source without amplifying what it finds. A good default
  - `4xNomosUniDAT_otf` - broad-subject and natural, with no artificial edge enhancement. The neutral general-purpose pick
  - `RealESRGAN_x4plus` - the safe, boring answer. Its usual criticism is that it smears fine texture into mush, which is exactly the weakness that does not matter here. No halo risk at all. Useful as a control when judging whether another model is actually helping
  - `realesr-general-x4v3` - for a noisy or heavily compressed source. Heavier denoise, a little plastic on its own, which the model then re-textures

Avoid anything built to look sharp on its own: `4x-UltraSharp` and `UltraSharpV2`, `4x_NMKD-Superscale`, `4x_NMKD-Siax`, `4x_foolhardy_Remacri`, and anything else with "Sharp" in the name. They are popular because they look good standing alone. Here their halos become the model's reference.

Subject-specific models (anime, face restoration) are fine on matching material and poor on anything else, same as ever.

One shortcut if the result is too crisp and you want to know why: unplug the upscaler entirely and compare against plain bicubic. That separates sharpening coming from the upscaler from detail the model itself is adding, which is worth knowing before tuning anything else.

## Upscale model and scale_factor [collapsed]

Whatever `upscale_model` produces is always resampled to exactly `scale_factor` afterwards. That makes the pairing matter:

  - A 4x model at `scale_factor` `4.0` runs at its native size - no resampling at all
  - A 4x model at `scale_factor` `2.0` enlarges then shrinks back down. This is supersampling and it looks genuinely good
  - A 2x model at `scale_factor` `4.0` leaves half the enlargement to plain bicubic, so most of the model's benefit is wasted

Undershooting the model's native scale is fine or better. Overshooting wastes it.

At `scale_factor` `1.0` the pairing stops mattering altogether: the enlargement is resampled straight back to the original size, so unplug `upscale_model` entirely. See "Sharpening without upscaling" above.

The shrink in that middle case does more than sharpen-proof the result: averaging four pixels into one removes a good deal of halo and invented texture on its own. So which model you pick matters much less at `scale_factor` `2.0` than at `4.0`. At 4.0, choose carefully.

## Seams and blending [collapsed]

There is no `mask_blur` widget, and that is deliberate.

Tiles are blended with complementary raised-cosine ramps centred on each shared tile boundary. Because a rising ramp and its falling partner add up to exactly `1.0` at every point, the weights of all the tiles covering a pixel always sum to exactly one. Seams are correct by construction, and there is nothing to tune.

The blend band is `overlap / 2` wide on each side of a boundary — the `report` output states it. The rest of `overlap` is context the model can see but is not allowed to change, which is what holds a tile in agreement with the neighbour it was cropped against.

If a seam is still visible, the cause is tile-to-tile drift rather than the blend. Raise `overlap`, lower `refine_strength`, or turn `color_match` on.

## Too much detail, or too sharp [collapsed]

There is no sharpening setting to turn off - nothing here sharpens deliberately. An over-crisp result comes from one of three places, in the order worth checking:

  - The upscale model. See "Choosing an upscale model" above. This is the most common cause, because a sharp upscaler's halos become the frozen context and the reference every tile works from, so the model copies them rather than fixing them. Unplug the upscaler and compare against plain bicubic to find out how much of the effect is coming from there
  - `refine_strength`. Drop it to `0.92`, then `0.88`. Below `1.0` the un-refined tile is mixed back in at every sampler step, so it restrains the model while it works instead of fading the result afterwards. That is why it is a better dial than `denoise` and better than blending the output with the original downstream
  - `tile_size`. Bigger tiles cover more of the picture each, so the content inside a tile is less magnified and the model invents less fine detail per unit area. Going from `1024` to `1536` noticeably calms things down, at the cost of memory

`denoise` is not the answer, despite looking like it should be. Flux.2 shifts its schedule hard toward the high-noise end, so `0.5` still keeps under 10% of the incoming image while also cutting the number of steps. At 4 steps there are only about five reachable settings, most of which do nothing until they suddenly do everything.

## Notes [collapsed]

  - The sigma schedule is built per tile from that tile's own latent size. Flux.2's schedule shift depends on how many tokens are being sampled, so a schedule built for the whole canvas — or for a number typed into a `Flux2Scheduler` — is fitted to a resolution that doesn't appear anywhere in the run
  - Tiles are sampled in a chess-board order, so no tile is worked on next to one that is still in progress
  - Every tile gets its own sampler seed. With a single shared seed, any ancestral or SDE sampler (`euler_ancestral`, `dpmpp_2m_sde`, `dpmpp_sde`) draws the identical noise sequence for every tile and stamps the same texture across the picture
  - All tiles crop their starting noise from one shared canvas-sized noise field, so overlapping tiles begin from identical noise where they overlap
  - The whole canvas is never VAE-encoded. Only tiles are, which is what keeps memory flat as the output grows
  - `color_match` needs the `color-matcher` package, the same one `Y7 Color Match (Masked)` uses. It is in `requirements.txt`
  - Adapted from [ComfyUI_KleinTiledUpscaler](https://github.com/Gavr728/ComfyUI_KleinTiledUpscaler) by Gavr728 (MIT), commit `e46c3b4`
