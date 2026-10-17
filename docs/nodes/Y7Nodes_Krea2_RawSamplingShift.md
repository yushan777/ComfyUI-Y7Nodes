# Krea 2 Raw Sampling Shift

Sets the sampling shift for Krea 2 Raw: resolution-dependent, or manual.

ComfyUI loads every Krea 2 model with a fixed shift of `1.15`, which suits Turbo but not Raw. Put this on the Raw model (e.g. stage 1 of a Krea 2 Two- or Three-Stage Sampler); the Turbo model needs no patch.

## Inputs

  - `model`: The model to patch
  - `sampling_mode`: `raw_dynamic` follows Krea 2's Raw schedule (shift `0.5` at 256x256 up to `1.15` at 1280x1280). `manual` uses `manual_shift`
  - `width` / `height`: The sampling resolution. Only used by `raw_dynamic`
  - `manual_shift` (advanced): The shift for `manual` mode

## Outputs

  - `MODEL`: The model with the shift applied


