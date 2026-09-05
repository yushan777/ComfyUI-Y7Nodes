# Pad Image for Outpainting

Adds empty space around an image so a model can paint into it, and builds the mask that tells it where to paint.

This is a rework of ComfyUI's built-in "Pad Image for Outpainting" with the tedious parts smoothed out: a button to reset the padding, fields that set several sides at once, a readout of the size you are going to get, and a choice of what to put in the new area.

Outputs:

  - `image (original)`: The image exactly as it came in, untouched. Saves running a second wire back from the loader when you want to compare before and after, or paste the original back over the result
  - `image (padded)`: The original image sitting on a bigger canvas
  - `mask`: White over the new area, black over the original, with a soft fade in between. Feed this to a mask input on your sampler or `SetLatentNoiseMask`
  - `width` / `height`: The size of the new canvas, handy for anything downstream that needs numbers

## Setting the padding

`left`, `top`, `right` and `bottom` are how many pixels to add to each edge. Zero on all four means no change.

The three fields underneath fill several boxes in at once, so you do not have to type the same number four times:

  - `set all sides`: puts one number into all four
  - `set left + right`: widens the image evenly, leaving the height alone
  - `set top + bottom`: makes it taller evenly, leaving the width alone

Each of them has `-` and `+` arrows that move in whole `step` amounts, so you can nudge the padding up and down rather than retyping.

The `Reset sides` button puts all four edges back to zero and leaves everything else alone.

The blue text at the bottom of the node shows what you are going to get, for example `1024 x 1024 -> 1280 x 1024`, with the final padding for each edge underneath. It updates as you change the numbers. It needs to know how big the incoming image is, so it stays blank until an image is connected upstream or you have run the workflow once.

## step

Models work in blocks of pixels, and sizes that are not a multiple of 8 (or 16, for some models) can produce a slightly soft or shifted result. `step` keeps you out of that trouble automatically: whatever you type into the four edges gets rounded to the nearest multiple of it before anything happens.

The default of 16 keeps you safe on the models that want multiples of 16 as well as the ones that only need 8. 32 and 64 are also common. Set it to 1 to turn the rounding off.

It also sets how far the `-` and `+` arrows move, so at the default one click adds 16 pixels.

## fill

What goes in the new area before the model paints over it. It all gets replaced, but what you start with still nudges the result, because the model can see it.

  - `grey`: flat 50% grey. What the built-in node does, and a safe neutral choice
  - `edge replicate`: stretches the outermost row of pixels outwards. Sky stays sky, grass stays grass
  - `mirror`: folds the image back on itself. Good for patterns and textures
  - `blurred edge`: a very soft blur of the nearby colours, so the new area is roughly the right colour without any fake detail. Often the best all-rounder
  - `noise`: random speckle in the image's own colours

If an outpaint keeps drifting to grey or washing out, try `blurred edge` or `edge replicate`.

## feathering

How wide, in pixels, the soft fade is where the original image meets the new area.

A hard edge (0) often leaves a visible seam. The default of 40 lets the model blend across the join. Larger values fade further into the original image, so it can repaint more of what was already there; smaller values protect the original more closely.

If the image is too small for the value you asked for, the widest fade that fits is used instead.
