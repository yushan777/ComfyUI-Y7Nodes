# CLIP Token Counter

Counts tokens in a text using the CLIP tokenizer.

Up to the first 77 tokens will be displayed, with any overflow tokens shown below.

The actual limit is 77 tokens, including two special tokens: `&lt;|startoftext|&gt;` (Beginning-of-Sequence) and `&lt;|endoftext|&gt;` (End-of-Sequence).

The final token in the sequence will be shown, along with a brief context of words leading up to it.

The `&lt;/w&gt;` marker indicates a word boundary, typically where a space followed the word in the original text.

Inputs:

  - text_in: Any text (string) input.
  - show_tokens: Displays tokenized version of text (requires re-run).
  - tokens_per_line: Number of token words per line (requires re-run).

Output:

  - text_out: A pass-through output for the input string

Other Widgets:

  - font_size: Change font size used in the text widget. 
  - Copy Text: Copy contents of the text widget.

Note: Longer prompts are supported, but how they are handled depends entirely on the specific implementation of the model and tokenizer. Some implementations may truncate, segment, or otherwise process longer inputs differently.
