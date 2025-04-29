from ..utils.logger import logger

def title(text):
    """Title text element"""
    return f'<div style="margin-bottom: 10px;">{text}</div>'

def short_desc(desc):
    """Create a short description element with the special ID"""
    return f'<div id="Y7_shortdesc" style="margin-bottom: 15px;">{desc}</div>'

def process_highlights(text):
    """Process text and convert `highlighted` parts to code style that works in both light and dark themes"""
    import re
    pattern = r'`([^`]+)`'
    # Theme-agnostic styling:
    return re.sub(pattern, r'<code style="border: 1px solid #666; border-radius: 3px; padding: 0px 1px; font-family: monospace; display: inline-block;">\1</code>', text)
    
# Then modify normal to use this
def normal(text, indent_level=0, font_size="12px"):
    """Normal text element with optional indentation and font size"""
    indent_px = indent_level * 20  # 20px per indent level
    processed_text = process_highlights(text)
    return f'<div style="margin-bottom: 8px; margin-left: {indent_px}px; font-size: {font_size};">{processed_text}</div>'

descriptions = {
    # "Y7Nodes_Brightness": [
    #     "Image Brightness Adjustment (demo node)",
    #     short_desc("Adjusts the brightness of an image by multiplying pixel values"),
    #     normal("Control the brightness level with the `strength` parameter:"),
    #     normal("- Values > 1.0 increase brightness", 1),
    #     normal("- Values < 1.0 decrease brightness", 1),
    #     normal("- Value of 1.0 leaves the image unchanged", 1),
    #     normal("Optionally provide a `filename_prefix` to use when saving the processed image.")
    # ],

    "Y7Nodes_ImageRow": [
        "Image Row Combiner",
        short_desc("Combines up to 4 images horizontally into a single row with captions"),
        normal("Takes up to 4 images and combines them horizontally into a single row image."),
        normal("All images are resized to a consistent height (the minimum height among all input images) while maintaining their aspect ratios."),
        normal("Captions are displayed in a bar beneath the images, with each caption centered under its corresponding image."),
        normal("Inputs:"),
        normal("- `image1`, `image2`, `image3`, `image4`: Optional image inputs to combine", 1),
        normal("- `captions`: Comma-separated list of captions for each image (e.g., \"image1, image2, image3, image4\")", 1),
        normal("- `save_image`: Toggle between saving to output directory or creating a preview only", 1),
        normal("- `save_filename`: Prefix for the saved image filename", 1),
        normal("The node displays a preview of the combined image in the ComfyUI interface."),
        normal("If no images are provided, an empty preview is returned.")
    ],

    "Y7Nodes_Text":[
        "Basic Text Input With Copy Button", 
        short_desc("Basic Text Input With Copy Button"),
    ],

    "Y7Nodes_Grid2Batch": [
        "Image Grid Splitter", 
        short_desc("Splits a grid of images into a batch of individual images"),
        normal("Takes a grid image (like those generated in XY-plots) and processes it into a batch of separate images."),
        normal("Specify the grid structure with `rows` and `columns` parameters."),
        normal("Set the dimensions of each individual image within the grid with `width` and `height`."),
        normal("If your grid has headers, specify their size with `x_header` and `y_header` parameters."),
        normal("If the dimensions of the image does not match your numbers, it will throw an error."),
        normal("The output is a batch of images that can be processed further in your workflow.")
    ],
    
    "Y7Nodes_ShowAnything": [
        "Show Anything",
        short_desc("Display the content of any input, regardless of its type."),
        normal("A debugging tool that displays information about any input in the ComfyUI interface."),
        normal("For string, integer, float, boolean values: Displays the content directly"),
        normal("For IMAGE and MASK tensors: Shows shape, data type, value range, mean, and std dev."),
        normal("For other tensors: Displays shape, data type, and value range"),                
        normal("For other types: Converts to JSON or string representation"),
        normal("Pass-through for the input.")
    ],
    
    "Y7Nodes_T5_TokenCounter": [
        "T5 V1.1 XXL Token Counter",
        short_desc("Counts tokens in a text using the T5 XXL tokenizer."),
        normal("Up to the first 256 or 512 tokens (default) will be displayed, with any overflow tokens shown below."),
        normal("The actual limit is 256 or 512 minus one special token reserved for the End-of-Sequence token `&lt;/s&gt;`"),
        normal("Some models such as Flux.1 Schnell impose a 256-token sequence limit."),
        normal("The final token in the sequence will be shown, along with a brief context of words leading up to it."),
        normal("Tokens prefixed with an underscore '_' represent a word boundary (New sentence or a space)."),
        normal("Tokens without an underscore '_' are usually subword pieces that continue from the previous token."),        
        normal("Inputs:"),
        normal("- text_in: Any text (string) input.", 1),
        normal("- show_tokens: Displays tokenized version of text (requires re-run).", 1),
        normal("- tokens_per_line: Number of token words per line (requires re-run).", 1),        
        normal("Output:"),
        normal("- text_out: A pass-through output for the input string", 1),
        normal("Widgets:"),
        normal("- font_size: Change font size used in the text widget. ", 1),
        normal("- Copy Text: Copy contents of the text widget.", 1),        
        normal(""),
        normal("Note: Longer prompts are supported, but how they are handled depends entirely on the specific "
        "implementation of the model and tokenizer. Some implementations may truncate, segment, or otherwise process longer inputs differently.")        
    ],

    "Y7Nodes_CLIP_TokenCounter": [
        "CLIP Token Counter",
        short_desc("Counts tokens in a text using the CLIP tokenizer."),
        normal("Up to the first 77 tokens will be displayed, with any overflow tokens shown below."),
        normal("The actual limit is 77 tokens, including two special tokens: `&lt;|startoftext|&gt;` (Beginning-of-Sequence) and `&lt;|endoftext|&gt;` (End-of-Sequence)."),
        normal("The final token in the sequence will be shown, along with a brief context of words leading up to it."),
        normal("The `&lt;/w&gt;` marker indicates a word boundary, typically where a space followed the word in the original text."),
        normal("Inputs:"),
        normal("- text_in: Any text (string) input.", 1),
        normal("- show_tokens: Displays tokenized version of text (requires re-run).", 1),
        normal("- tokens_per_line: Number of token words per line (requires re-run).", 1),        
        normal("Output:"),
        normal("- text_out: A pass-through output for the input string", 1),
        normal("Other Widgets:"),
        normal("- font_size: Change font size used in the text widget. ", 1),
        normal("- Copy Text: Copy contents of the text widget.", 1),
        normal(""),
        normal("Note: Longer prompts are supported, but how they are handled depends entirely on the specific "
            "implementation of the model and tokenizer. Some implementations may truncate, segment, or otherwise process longer inputs differently.")
    ],

    "Y7Nodes_PromptEnhancerFlux":[
        "Prompt Enhancer (Flux)",
        short_desc("Takes a basic prompt and enhances it, and separates response into T5 and CLIP prompts optimized for Flux.1 image generation"),        
        normal("1. T5 Prompt: A detailed natural language description (up to ~512 tokens)", 1),
        normal("2. CLIP Prompt: A concise keyword list (up to ~75 tokens)", 1),
        normal("Inputs:"),
        normal("- `prompt`: Your basic text prompt to enhance. If you have a token (trigger) word(s) then enclose them in square brackets [ohwx man]", 1),
        normal("- `llm_name`: name of the LLM to use", 1),
        normal("- `quantization`: Model precision. Lower precision saves memory and speeds up inference at the cost of quality. So for OpenHermes-2.5-Mistral-7B:", 1),
        normal("-  `none` = 14.3GB, `8bit` = 12.4GB, `4bit` = 4.7GB", 2),
        normal("- `temperature`: Controls randomness (0.1-2.0, default 0.7). Higher values produce more diverse outputs", 1),
        normal("- `top_p`: Nucleus sampling parameter (0.8-1.0, default 0.9). Limits tokens to the most probable ones", 1),
        normal("- `top_k`: Limits token selection (22-100, default 40). Restricts to the k most likely next tokens", 1),
        normal("- `unload_models_before_run`: Frees up memory by unloading all models and clearing cache before running.  Useful for heavy workflows.", 1),
        normal("Outputs:"),
        normal("- `t5_prompt`: Enhanced detailed natural language description", 1),
        normal("- `clip_prompt`: Enhanced concise keyword list", 1),
        normal("LLM Model:"),
        normal("Note: First-time use will download the model if it does not exist.")
    ],

    "Y7Nodes_CatchEditTextNodeDual": [
        "Catch and Edit Text (Dual)",
        short_desc("Catches text from two separate inputs making them editable"),
        normal("This node acts as a receiver and editor for text sent from two sources."),
        normal("action widget:"),
        normal("- use_input: pass the input text as it without modification.",1),
        normal("- use_edit_mute_input: pass the edited text in the widget and mute the source node of the input(s).",1),
        normal(""),
        normal("Based on the original CatchEditTextNode by ImagineerNL"),
        normal("https://github.com/ImagineerNL/ComfyUI-IMGNR-Utils"),
        normal("If you just need one text input then I recommend using his original node."),

    ],

    # Add more node descriptions here
}

def as_html(entry, depth=0):
    """Convert structured documentation into HTML with collapsible sections"""
    if isinstance(entry, dict):
        size = 0.8 if depth < 2 else 1
        html = ''
        for k in entry:
            if k == "collapsed":
                continue
            collapse_single = k.endswith("_collapsed")
            if collapse_single:
                name = k[:-len("_collapsed")]
            else:
                name = k
            if collapse_single:
                name = k[:-len("_collapsed")]
            else:
                name = k
            collapse_flag = ' Y7Nodes_precollapse' if entry.get("collapsed", False) or collapse_single else ''
            html += f'<div Y7Nodes_title=\"{name}\" style=\"display: flex; font-size: {size}em\" class=\"Y7Nodes_collapse{collapse_flag}\"><div style=\"color: #AAA; height: 1.5em;\">[<span style=\"font-family: monospace\">-</span>]</div><div style=\"width: 100%\">{name}: {as_html(entry[k], depth=depth+1)}</div></div>'
        return html
    if isinstance(entry, list):
        if depth == 0:
            depth += 1
            size = .8
        else:
            size = 1
        html = ''
        html += entry[0]
        for i in entry[1:]:
            html += f'<div style=\"font-size: {size}em\">{as_html(i, depth=depth)}</div>'
        return html
    return str(entry)

def format_descriptions(nodes):
    """Applies HTML documentation to node classes"""
    logger.info(f"Formatting descriptions for nodes: {list(nodes.keys())}")
    logger.info(f"Available descriptions: {list(descriptions.keys())}")
    
    for k in descriptions:
        if k in nodes:
            logger.info(f"Setting DESCRIPTION for {k}")
            nodes[k].DESCRIPTION = as_html(descriptions[k])
            # Also set a direct description property for easier access
            nodes[k].description = as_html(descriptions[k])
        else:
            logger.warning(f"Node {k} has a description but is not in the nodes dictionary")
    
    # Optionally, log any undocumented nodes
    undocumented_nodes = []
    for k in nodes:
        if k.startswith("Y7_") and not hasattr(nodes[k], "DESCRIPTION"):
            undocumented_nodes.append(k)
    
    if len(undocumented_nodes) > 0:
        logger.warning(f"Some nodes have not been documented: {undocumented_nodes}")
    
    # Return the number of descriptions applied for confirmation
    return len([k for k in descriptions if k in nodes])
