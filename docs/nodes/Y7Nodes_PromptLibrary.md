# Prompt Library

A visual library of saved prompts, each with an optional example image. Outputs the prompt of the selected card, or an empty string when nothing is selected.

Click a card to select it. Click it again to clear the selection.

Double-click a card to edit it. Right-click for Edit, Copy prompt and Delete.

The `☰` / `▦` button switches between tiles and a list (small thumbnail, name and the start of the prompt). Each node remembers its view.

Changes to the selected entry's prompt are picked up on the next run, so there is no need to reselect it.

Outputs:

  - `prompt`: The selected entry's prompt (STRING), or an empty string.

## Libraries [collapsed]
Pick a library from the dropdown (entry counts in brackets). Use `+` to create one, and `⋯` to rename, export, reload or delete the current library.
Libraries are stored in ComfyUI's user directory under `y7_prompt_lib/`:
  - `libraries/<name>.json`: the entries (name, prompt, image flag).
  - `previews/<id>.webp`: example images, downscaled to 512px.
Entries are tracked by an internal id, so renaming an entry or moving it to another library keeps saved workflows pointing at it.

## Entries [collapsed]
`+ Entry` opens the editor: library, name, prompt and an optional example image.
Add an image by clicking the image box, dropping a file onto it, or pasting from the clipboard. PNG, JPEG and WebP are accepted.
Names must be unique within a library; a duplicate gets a number added (e.g. `Portrait 2`).

## Import and export [collapsed]
`Import` accepts files, a folder, or a .zip, in either of two formats:
  - Paired files: `name.txt` holds the prompt and `name.png` / `.jpg` / `.webp` (same name, same folder) is its image. The entry name is the file name with `_` shown as spaces. A .txt without an image becomes a text-only entry; an image without a .txt is skipped.
  - Library export: a .zip made by `⋯` → `Export library` (`library.json` plus `previews/`). Use this for backups or for moving a library to another machine.
The import dialog previews the entries before anything is written. Entries whose name already exists in the target library are skipped.
