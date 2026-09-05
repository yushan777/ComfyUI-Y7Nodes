# T5 V1.1 XXL Token Counter

Counts tokens in a text using the T5 XXL tokenizer.

Up to the first 256 or 512 tokens (default) will be displayed, with any overflow tokens shown below.

The actual limit is 256 or 512 minus one special token reserved for the End-of-Sequence token `&lt;/s&gt;`

Some models such as Flux.1 Schnell impose a 256-token sequence limit.

The final token in the sequence will be shown, along with a brief context of words leading up to it.

Tokens prefixed with an underscore '_' represent a word boundary (New sentence or a space).

Tokens without an underscore '_' are usually subword pieces that continue from the previous token.

Inputs:

  - text_in: Any text (string) input.
  - show_tokens: Displays tokenized version of text (requires re-run).
  - tokens_per_line: Number of token words per line (requires re-run).

Output:

  - text_out: A pass-through output for the input string

Widgets:

  - font_size: Change font size used in the text widget. 
  - Copy Text: Copy contents of the text widget.

Note: Longer prompts are supported, but how they are handled depends entirely on the specific implementation of the model and tokenizer. Some implementations may truncate, segment, or otherwise process longer inputs differently.
