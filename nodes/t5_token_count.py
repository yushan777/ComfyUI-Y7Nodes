from functools import lru_cache

from transformers import T5Tokenizer
from comfy_api.latest import io

from ..utils.colored_print import color

# This node has 1 backend input and 1 backend output 
# other widgets (frontend) are added in the corresponding javascript file
# 2 hidden input types are used for state persistence.

# =====================================================================================
class Y7Nodes_T5_TokenCounter(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Y7Nodes_T5_TokenCounter",
            display_name="Y7 T5 Token Counter",
            category="Y7Nodes/Utils",
            description="Count T5 tokens for a text input and show the tokenised result.",
            is_output_node=True,
            inputs=[
                io.String.Input(
                    "text_in",
                    optional=True,
                    default="",
                    force_input=True,
                    tooltip="Text input to count tokens for using the T5 tokenizer",
                ),
            ],
            outputs=[
                io.String.Output(display_name="text_out"),
            ],
            # these are used to help restore state
            hidden=[
                io.Hidden.unique_id,
                io.Hidden.extra_pnginfo,
            ],
        )

    # ====================================================================================
    # main function
    @classmethod
    def execute(cls, text_in="") -> io.NodeOutput:
        # unique_id and extra_pnginfo are declared as hidden inputs in the schema
        unique_id = cls.hidden.unique_id
        extra_pnginfo = cls.hidden.extra_pnginfo
        # Get the input string
        string_input = text_in or ""

        # Normalize en/em dashes(U+2013 or U+2014) to standard hyphen (U+002D)
        # cos they might come out looking lik âĢĶ
        string_input = (
            string_input
            .replace('\u2014', '-')  # em dash —
            .replace('\u2013', '-')  # en dash –
            .replace('\u2212', '-')  # minus sign −
        )

        if string_input:
            
            # Initialize tokenizer
            tokenizer = cls.load_tokenizer()
            num_of_special_tokens = 1

            # user-defined token limit or length
            token_sequence_length = cls.get_tokens_limit_val(extra_pnginfo, unique_id)

            # Tokenize the text (both original and truncated versions)
            raw_input_original = tokenizer(string_input, padding=False, truncation=False, max_length=token_sequence_length)
            raw_input_truncated = tokenizer(string_input, padding=False, truncation=True, max_length=token_sequence_length)

            # Extract input IDs
            input_ids_original = raw_input_original['input_ids']
            input_ids_truncated = raw_input_truncated['input_ids']

            # Get token counts
            token_count_original = len(input_ids_original)

            # Convert the truncated token IDs to tokens (one call takes the whole list)
            tokens_truncated = tokenizer.convert_ids_to_tokens(input_ids_truncated)

            # Process limit token information. Truncating to token_sequence_length guarantees
            # the sequence fits, so the only question is whether it reached the limit.
            last_token = ""
            context_limit_token = ""
            index = token_sequence_length - num_of_special_tokens - 1
            if index < len(tokens_truncated):
                last_token = tokens_truncated[index]

                # Get context around the limit token
                selected_token_ids = input_ids_truncated[(token_sequence_length-9):index + 1]
                context_limit_token = tokenizer.decode(selected_token_ids, clean_up_tokenization_spaces=True).strip()

            # Locate this node in the workflow once; used both for reading the widget values
            # the frontend created and for writing the result back
            workflow_node = cls.find_workflow_node(extra_pnginfo, unique_id)

            output_text = ""

            # Build output text
            # output_text = f"Special Token: \n"
            # output_text += f"   1: End-of-Sequence (EOS): '{eos_token}'\n\n"

            # token limit
            # output_text += f"Token Limit: {token_sequence_length}\n"

            # token count
            output_text += f"Token Count: {token_count_original} / {token_sequence_length}"
            over_limit = token_count_original - token_sequence_length
            output_text += f": >>{over_limit} over limit<<\n" if over_limit > 0 else "\n"
            output_text += "\n"

            # Add limit token information if it exists
            if last_token:
                output_text += f'Last Token: "{last_token}" | "...{context_limit_token}"\n\n'

            # Show tokens if the widget is enabled
            show_tokens = cls.get_show_tokens_val(workflow_node)
            if show_tokens:

                # Formatting the token listing is only worth doing if it gets displayed
                max_tokens_per_line = cls.get_tokens_per_line_val(workflow_node)
                tokens_original = tokenizer.convert_ids_to_tokens(input_ids_original)
                tokens_truncated_formatted = cls.format_tokens_T5(tokens_truncated, max_tokens_per_line, start_index=0)
                tokens_overflow_formatted = cls.format_tokens_T5(
                    tokens_original, max_tokens_per_line, start_index=(token_sequence_length - 1)
                )

                # Set appropriate title based on truncation
                title = (f"Tokens (Truncated - Showing first {token_sequence_length})\n==========================================\n"
                        if token_count_original > token_sequence_length
                        else f"Tokens\n==========================================\n")
                
                # Add tokens section
                output_text += title
                output_text += f"{tokens_truncated_formatted}\n"                                                            
            
                if tokens_overflow_formatted:
                    output_text += f"\nOverflow Tokens\n"                    
                    output_text += f"=========================\n"
                    output_text += (
                            f"Note: The final token in the truncated output above is the EOS token.\n"
                            f"However, in the overflow output, the token at the same index position [{token_sequence_length-1}]\n"
                            f"is an actual text token — and would've appeared in the main sequence\n"
                            f"if truncation (and EOS insertion) hadn't occurred.\n")
                    output_text += f"\n{tokens_overflow_formatted}"

            # Update the node's widget values if extra_pnginfo is available
            if not extra_pnginfo:
                print("Error: extra_pnginfo is empty")
            elif (not isinstance(extra_pnginfo, dict) or "workflow" not in extra_pnginfo):
                print("Error: extra_pnginfo is not a dict or missing 'workflow' key")
            elif workflow_node:
                workflow_node["widgets_values"] = [output_text]

        else:            
            output_text = "No input text provided."        
        
        # Pass the token count info (output_text) to the UI but return the original string_input as output
        # This allows the node to be used as a pass-through while still showing token info
        return io.NodeOutput(string_input, ui={"text": output_text})

    # ===================================================================================
    @classmethod
    def format_tokens_T5(cls, tokens, max_tokens_per_line=4, start_index=0):

        # Format tokens  into a string with padding, indices, and number of tokens per line.
        # only include tokens that fit into the tokenizer_max_length, ignoring the rest
        # We include the special token EOS
        # Args: 
        #  - tokens (list): Either a list of first 512 tokens, or could be original list 
        #    to be started at a specific index to list the overflow tokens
        #  - max_tokens_per_line (int): Number of tokens to display per line            
        # Returns: 
        #  - tokens_formatted

        # Early return for empty tokens
        if not tokens:
            return ""
        
        # Calculate the maximum token length for padding
        max_str_length = max(len(token) for token in tokens)
        
        # get token length 
        tokens_length = len(tokens)
        
        # Determine the index formatting based on number of tokens
        index_format = "{:2d}" if tokens_length <= 99 else "{:3d}"

        # Format tokens into a string with padding, indices, and tokens per line
        tokens_formatted = ""
        # for i, token in enumerate(tokens):
        for i, token in enumerate(tokens[start_index:], start=start_index): 
            # Format index with fixed width and pad token
            padded_token = token.ljust(max_str_length)            
            tokens_formatted += f"[{index_format.format(i)}] {padded_token}"

            # Add a new line after every N tokens
            if ((i - start_index + 1) % max_tokens_per_line == 0):
                tokens_formatted += "\n"
            
        return tokens_formatted

    
    # ========================================================================================
    @classmethod
    def find_workflow_node(cls, extra_pnginfo, unique_id):

        # Find this node's entry in the workflow carried by extra_pnginfo, or None.
        # Args:
        #  - extra_pnginfo: The extra PNG info containing workflow data
        #  - unique_id: The unique ID of this node
        # Returns:
        #  - The node dict from the workflow, or None if it can't be found

        if not extra_pnginfo or "workflow" not in extra_pnginfo:
            return None

        # find the node in the WF that matches our unique_id for this node
        # return the first match & convert the id to str
        return next((x for x in extra_pnginfo["workflow"]["nodes"] if str(x["id"]) == unique_id), None)

    # ========================================================================================
    @classmethod
    def get_widget_val(cls, node, property_name, default, cast=None):

        # Get the value of a widget that was created in the frontend javascript, using the
        # index the frontend stored in the node's properties.
        #
        # For widgets created in JavaScript, we can access their vals using the widget
        # index that is stored in the extra_pnginfo dictionary:
        #   extra_pnginfo-> workflow -> nodes -> properties
        #
        # Example:
        # node
        # ├── id
        # ├── type
        # ├── inputs
        # ├── outputs
        # ├── properties
        # │   ├── various properties including "show_tokens_index"
        # └── widgets_values
        #     └── array of widget values
        #
        # {
        #     'workflow': {
        #         'nodes': [
        #             {
        #                 'id': 102,
        #                 'type': 'Y7Nodes_T5_TokenCounter',
        #                 'inputs': [
        #                     {'name': 'text_in', 'shape': 7, 'type': 'MY_STRING', 'link': 152}
        #                 ],
        #                 'outputs': [
        #                     {'name': 'text_out', 'type': 'STRING', 'links': []}
        #                 ],
        #                 'properties': {
        #                     'ver': 'xxxxxxxx',
        #                     'Node name for S&R': 'Y7Nodes_T5_TokenCounter',
        #                     'show_tokens_index': 1,
        #                     'tokens_per_line_index': 2
        #                 },
        #                 'widgets_values': ['hello', False, '10', None, None]
        #             },
        #
        #         ],
        #         .....
        #     }
        # }
        #
        # Args:
        #  - node: The node dict from the workflow (see find_workflow_node)
        #  - property_name: The property holding the widget's index, e.g. "show_tokens_index"
        #  - default: Returned if anything is missing or out of range
        #  - cast: Optional callable applied to the value before returning it
        # Returns:
        #  - The widget's value, or default

        # check node is truthy and it has widgets_values key
        if not node or "widgets_values" not in node:
            return default

        # now check if the node has properties key and contains the custom property
        # we are after (as defined by property_name)
        index = node.get("properties", {}).get(property_name)

        # make sure the index is within bounds for the widgets_values array
        if not isinstance(index, int) or index >= len(node["widgets_values"]):
            return default

        value = node["widgets_values"][index]
        return cast(value) if cast else value

    # ========================================================================================
    @classmethod
    def get_show_tokens_val(cls, node):
        # The bool from the show_tokens widget, or False if not found
        return cls.get_widget_val(node, "show_tokens_index", False)

    # ===================================================================================
    @classmethod
    def get_tokens_per_line_val(cls, node):
        # The number of tokens to print per line, or 4 if not found
        return cls.get_widget_val(node, "tokens_per_line_index", 4, cast=int)

    # ===================================================================================
    @classmethod
    def get_tokens_limit_val(cls, extra_pnginfo, unique_id):
        # similar in functionality to get_show_tokens_val() above
        # gets value from widget named token_limit

        property_name = "token_limit_index"
        # print(f"→ [get_tokens_limit_val] Called for node ID: {unique_id}", color.BRIGHT_CYAN)

        # Check if extra_pnginfo exists and it contains workflow info
        if not extra_pnginfo:
            print("⚠️ extra_pnginfo is None or empty", color.RED)
            return 512
        if "workflow" not in extra_pnginfo:
            print("⚠️ 'workflow' key not found in extra_pnginfo", color.RED)
            return 512

        workflow = extra_pnginfo["workflow"]
        # print(f"→ Workflow contains {len(workflow.get('nodes', []))} nodes", color.BRIGHT_CYAN)

        # Find the node in the workflow matching this node's unique_id
        node = next((x for x in workflow["nodes"] if str(x["id"]) == unique_id), None)

        if not node:
            print(f"⚠️ No node found with ID: {unique_id}", color.RED)
            return 512        

        if "widgets_values" not in node:
            print(f"⚠️ Node is missing 'widgets_values'", color.RED)
            return 512

        if "properties" not in node:
            print(f"⚠️ Node is missing 'properties'", color.RED)
            return 512

        if property_name not in node["properties"]:
            print(f"⚠️ Property '{property_name}' not found in node['properties']", color.RED)
            return 512

        index = node["properties"][property_name]
        # print(f"token_limit_index = {index}", color.YELLOW)

        widgets = node["widgets_values"]
        # print(f"→ Node has {len(widgets)} widget(s)", color.BRIGHT_CYAN)

        if index >= len(widgets):
            print(f"⚠️ token_limit_index {index} out of bounds for widgets_values (len={len(widgets)})", color.RED)
            return 512

        value = widgets[index]
        # print(f"token limit value = {value}", color.GREEN)

        try:
            return int(value)
        except ValueError:
            print(f"⚠️ Value at index {index} is not an int: {value}", color.RED)
            return 512

 
            
    # ===================================================================================
    @classmethod
    def load_tokenizer(cls):
        # Returns the T5 tokenizer, loading it on first use.
        # ComfyUI hands execute() a fresh clone of this class on every run, so the cache
        # lives at module level rather than on the class.
        return load_t5_tokenizer()


# =====================================================================================
@lru_cache(maxsize=1)
def load_t5_tokenizer():

    # Load the T5 tokenizer from local dir
    # Returns the loaded T5 tokenizer

    # FALLBACK_HF_MODEL = "t5-large"
    FALLBACK_HF_MODEL = "google/t5-v1_1-xxl"

    try:
        # Use Path to get the absolute path to the tokenizer directory
        # This matches the approach used in __init__.py for web routes
        from pathlib import Path
        current_dir = Path(__file__).parent.parent.absolute()
        local_tokenizer_path = (current_dir / "text_encoders" / "t5_tokenizer").as_posix()
        # print(f"Loading tokenizer from local path: {local_tokenizer_path}")
        return T5Tokenizer.from_pretrained(local_tokenizer_path)

    except Exception as e:
        print(f"Unable loading local tokenizer: files may be missing or corrupt.", color.ORANGE)
        print(f"Downloading from {FALLBACK_HF_MODEL} tokenizer on HuggingFace", color.ORANGE)

        try:

            # Download the tokenizer
            tokenizer = T5Tokenizer.from_pretrained(FALLBACK_HF_MODEL)

            # Create the local directory if it doesn't exist
            from pathlib import Path
            current_dir = Path(__file__).parent.parent.absolute()
            local_tokenizer_path = (current_dir / "text_encoders" / "t5_tokenizer")
            local_tokenizer_path.mkdir(parents=True, exist_ok=True)

            # Save the tokenizer to the local directory
            print(f"Saving tokenizer to local path: {local_tokenizer_path.as_posix()}", color.BRIGHT_GREEN)
            tokenizer.save_pretrained(local_tokenizer_path.as_posix())
            print(f"Tokenizer saved successfully!", color.BRIGHT_GREEN)

            return tokenizer

        except Exception as save_error:
            print(f"Error saving tokenizer locally: {save_error}\nUsing cached copy.", color.ORANGE)

            # If saving fails, still return the downloaded tokenizer
            return T5Tokenizer.from_pretrained(FALLBACK_HF_MODEL)
