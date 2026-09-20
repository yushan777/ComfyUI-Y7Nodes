# Save Image

Saves images to the ComfyUI output directory, like the native Save Image node, with a separate sub-folder setting, control over the file counter and an optional 16-bit PNG mode.

Files are tagged with the prompt and workflow exactly as the native node does, so they can be dragged back into ComfyUI to reload the workflow.

Inputs:

  - `images`: The images to save
  - `save_or_preview`: `save` (default) writes the images to the output directory. `preview` only shows them on the node, like Preview Image: they go to ComfyUI's temp directory, which is cleared on restart, and the settings below are ignored. Handy for trying things out without filling your output folder
  - `sub_folder`: The folder inside the output directory to save to, e.g. `portraits` or `flux/%date:yyyy-MM-dd%`. Created if it doesn't exist. Leave empty to save to the output directory itself. Default `foldername/%date:yyyy-MM-dd%`, a folder per day
  - `filename_prefix`: The file name prefix, which may include tokens such as `%date:yyyy-MM-dd%` or `%Empty Latent Image.width%`. A path here (`a/b/name`) still works as in the native node, and goes inside `sub_folder`. In `%Node.widget%`, `widget` can also be a name you've renamed a widget to: the real name is tried first, then renamed labels, so `%Float.denoise%` finds a Float node whose `value` you renamed to `denoise`. `Node` can also be a node ID, e.g. `%73.denoise%`, for top-level nodes only; a node titled `73` takes priority. IDs change when a node is copied or pasted into another workflow, so titles are the safer choice for reuse. Add `:N` after the widget to round a number to N decimal places, e.g. `%Float.value:2%` gives `0.80` rather than `0.8`, so names sort and line up. Default `img_%date:hhmmss%`, the time the prompt was queued
  - `counter_digits`: How many digits the counter at the end of the name is padded to (default `4`, giving `img_0001.png`). Unlike the native node there's no underscore after the counter
  - `counter_per_folder`: On (default), one counter runs through every image in the folder, whatever its prefix, so a day's images number `0001`, `0002`, … even when the prefix changes every second. Numbering carries on from any files already there, including the native node's. Off gives each prefix its own counter, as in the native node
  - `bit_depth`: `8-bit` gives the same file as the native node. `16-bit` keeps the finer tonal steps the image carries, which helps smooth gradients survive later grading or editing. Files are roughly 4–5× larger

Outputs:

  - `images`: The input images, passed through unchanged

## Date and time in names [collapsed]

Both `sub_folder` and `filename_prefix` accept date and time tokens, and the two kinds can be mixed.

`%date:FORMAT%` is filled in when the prompt is queued. FORMAT is built from:

  - `yyyy` / `yy`: year (`2026` / `26`)
  - `MM` / `M`: month, zero-padded or not (`09` / `9`)
  - `dd` / `d`: day
  - `hh` / `h`: hour, 24-hour clock
  - `mm` / `m`: minute (lower case, unlike month)
  - `ss` / `s`: second

`%year%`, `%month%`, `%day%`, `%hour%`, `%minute%`, `%second%` are a fixed-format alternative, already zero-padded, filled in when the file is saved.

Examples:

  - `sub_folder` = `%date:yyyy-MM-dd%` saves to `output/2026-09-19/`
  - `sub_folder` = `renders/%date:yyyy-MM%/%date:dd%` saves to `output/renders/2026-09/19/`
  - `filename_prefix` = `%date:yyyy-MM-dd_hhmmss%_flux` gives `2026-09-19_140507_flux_0001.png`
  - `sub_folder` = `%year%-%month%-%day%`, `filename_prefix` = `img_%hour%%minute%%second%`

Things to avoid:

  - Dots inside `%date:...%`. Dots separate `%Node.widget%` references, so `%date:yyyy.MM.dd%` comes out as just the year and `%date:yyyy.MM%` isn't replaced at all. Use `-` or `_`, or put the dot outside the token: `%date:yyyy%.%date:MM%`
  - Colons in the time (`hh:mm`). Linux accepts them but they're invalid in Windows file names
  - `%date:...%` uses the browser's clock when you queue; the `%year%`-style tokens use the server's clock when the file is saved. This only matters if ComfyUI runs on another machine or time zone, or the job waits a long time in the queue
  - API and scripted runs: "Export (API)" goes through the browser, so it saves the date and time of the export into the JSON, and every run of that file uses them. To get the run's own date, put the raw token (e.g. `%date:yyyy-MM-dd%`) back into the exported JSON: the node fills in any `%date:...%` that arrives unreplaced, using the server's clock when the file is saved. `%Node.widget%` tokens are browser-only and have no such fallback
