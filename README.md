# Y7 Nodes for ComfyUI
Only one for now.

# Installation
Clone this repo into the `custom_nodes` folder.<br>
```
git clone https://github.com/yushan777/ComfyUI-Y7Nodes.git
```
No dependencies

# Nodes:
**Count Tokens**<br>
Counts tokens in a given text based Supports SD1, SD2, and SDXL.
Will display the total tokens from the text and how they are split into chunks if the number of tokens spill over the limit.

Limits always include the BOS (start of text) and EOS (end of text) tokens.  This is why you might sometimes see the limit being 77 or 75 as some counters will disregard them the two tokens. 

By default will just show the totals. If you wish to see a breakdown of how the text is tokenizes you can turn on `show_token_word_pairs` 

`token_word_pairs_per_line` : changes how many token-word pairs will be displayed per line

`hide_end_of_word_tag` will remove the `</w>` tag and is purely for readability purposes. 

![Alt text](https://github.com/yushan777/ComfyUI-Y7Nodes/blob/main/examples/count-tokens-workflow.png)

**Photo Prompter**<br>

Inspired by dagthomas's Auto Prompter node.  I decided to make my own variation that aligns more with how I tend to prompt.  Also to separate photography and art styles into separate nodes. 

If clothing and footwear are set to `random` then they will the selection will correspond to the gender of the selected subject (if any).  If manually selected, then all items are possible. 

Default data are stored in data/default. If you wish to add your own custom items to the lists then create them in `data/custom` with the same filename and same format as the original file. This is to ensure if that if changes are made to the defaults then your custom items do not get wiped out. 

When set to `random`, some of the attributes will be dependent on others.  For example, if `clothing_upper` is a dress, then `pants` or `trousers` etc will be ignored, but `socks` etc will be allowed.   Or if an `indoor_scene`  is `active` or `random` then outdoor_scene will be ignored.  If `indoor_scene` is disabled then `outdoor_scene` will be considered. However if set to manual selection, then any combination is allowed to allow for flexibility. 

Changes to the lists can potentially make reproduceability difficult since the random choices can be affected if items in lists are added/deleted or shuffled. For this reason I recommend incuding a show text or show any node to display the full prompt so that when saving, the contents of the show text node will be included in the embedded metadata within the image.  In addition using https://github.com/mpiquero1111/ComfyUI-SaveImgPrompt node to save the workflow in a separate txt file next to every image can be useful in the event that you edit the image causing the metadata to be lost. 

| Attribute                  | Description                                                  |
| :------------------------- | ------------------------------------------------------------ |
| `seed`                     | Used to randomize attributes                                 |
| `control_after_generate`   | Seed control                                                 |
| `custom`                   | For custom word(s) to be included in the prompt. Will be placed at the beginning. |
| `style_and_framing`        | Photographic style and framing.                              |
| `subject_class`            | `man`, `woman` etc                                           |
| `role`                     | character types, roles and professions                       |
| `hairstyle`                | hair styles                                                  |
| `body_shape`               | body shapes and types                                        |
| `randomize_clothing_color` | If on then clothing and footwear will be randomized instead of using default. |
| `clothing_upper`           | Upper clothing items. Includes default color per item.  If set to `random`  then the random selection will correspond to the gender of the selected subject (if any). |
| `clothing_lower`           | Lower clothing items. Includes default color per item. If set to `random`  then the random selection will correspond to the gender of the selected subject (if any). |
| `footwear`                 | Footwear. Includes default color per item. If set to `random`  then the random selection will correspond to the gender of the selected subject (if any). |
| `accessories`              | Smaller wearable or carryable items such as hats, necklaces, bags |
| `action`                   | Standing, sitting etc                                        |
| `gaze`                     | Direction of where subject is looking at                     |
| `hands`                    | Positioning of hands                                         |
| `show_detailed_location`   | If on then Location descriptions will be longer and more verbose. |
| `location_interior`        | Interior locations                                           |
| `location_exterior`        | Exterior locations.  Will be active if interior locations is disabled. |
| `lighting`                 | Lighting conditions or styles                                |
| `time_of_day`              | Time of day : `morning`, `midday`, `afternoon`, `evening` etc |
| `weather`                  | Weather conditions. if location is interior then description will describe conditions outside of the interior location. |
| `camera_or_film`           | Snalog film types, digital camera models                     |
| `photographer`             | Photographer names                                           |
| `remove_commas_periods`    | Remove all  commas and periods from the prompt               |

**Art Style Prompter**
