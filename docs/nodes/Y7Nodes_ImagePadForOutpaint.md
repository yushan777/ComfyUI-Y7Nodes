# Pad Image for Outpainting

Adds empty space around an image so a model can paint into it, and builds the mask that tells it where to paint.

A rework of ComfyUI's built-in `Pad Image for Outpainting`, with a reset button, fields that set several sides at once, a readout of the size you will get, and a choice of what to put in the new area.

Inputs:

  - `image`: The image to pad
  - `step`: Padding is rounded to a multiple of this before anything happens, keeping the final size one the model is happy with. Default 16 is safe for models wanting 8 or 16; set to 1 to turn the rounding off. Also sets how far the `-`/`+` arrows move
  - `fill`: What goes in the new area before the model paints over it — see below
  - `feathering`: Width in pixels of the soft fade where the original meets the new area. 0 gives a hard edge, which often leaves a visible seam; larger values let the model repaint more of the original. Capped to the widest fade the image can fit
  - `left` / `top` / `right` / `bottom`: Pixels to add to each edge. Zero on all four means no change

Outputs:

  - `image (original)`: Passthrough of the orignal, unpadded image.
  - `image (padded)`: The padded image.
  - `mask`: White over the new area, black over the original, faded in between. Feed this to a mask input on your sampler or `SetLatentNoiseMask`
  - `width` / `height`: Size of the new canvas, for anything downstream that needs numbers

Fill modes:

  - `grey`: Flat 50% grey, same as the built-in node
  - `edge replicate`: Stretches the outermost pixels outwards, so sky stays sky and grass stays grass
  - `mirror`: Folds the image back on itself. Good for patterns and textures
  - `blurred edge`: A soft blur of the nearby colours — roughly the right colour with no fake detail. Often the best all-rounder
  - `noise`: Random speckle in the image's own colours

It all gets painted over, but the model can see what it starts with, so it still nudges the result. If an outpaint keeps drifting to grey, try `blurred edge` or `edge replicate`.

On the node:

  - `set all sides`, `set left + right` and `set top + bottom` fill in several edges at once, so you do not type the same number four times
  - `Reset sides` puts all four edges back to zero, leaving everything else alone
  - The blue text shows what you will get, e.g. `1024 x 1024 -> 1280 x 1024`, with the final padding per edge underneath. It stays blank until an image is connected upstream or the workflow has run once
