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


