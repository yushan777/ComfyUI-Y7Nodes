import argparse
import os
import json
from transformers import CLIPTokenizer
from ..utils.logger import logger
from ..utils.colored_print import color, style
from .documentation import descriptions, as_html

# This node has 1 backend input and 1 backend output 
# other widgets (frontend) are added in the corresponding javascript file
# 2 hidden input types are used for state persistence.

# =====================================================================================
class Y7Nodes_CLIP_TokenCounter:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {}, 
            "optional": {
                "text_in": ("STRING", {"default": "", "forceInput": True}),                   
            },
            "hidden": {
                # these are used to help restore state
                "unique_id": "UNIQUE_ID", 
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ('text_out',)
    OUTPUT_NODE = True
    FUNCTION = "count_tokens_CLIP"
    CATEGORY = "Y7Nodes/Utils"

    # ====================================================================================
    # main function
    def count_tokens_CLIP(self, **kwargs):
        # Extract parameters from kwargs
        unique_id = kwargs.get('unique_id')
        extra_pnginfo = kwargs.get('extra_pnginfo')
        string_input = kwargs.get('text_in', '')        

        # Normalize en/em dashes(U+2013 or U+2014) to standard hyphen (U+002D)
        # cos they might come out looking lik âĢĶ
        string_input = (
            string_input
            .replace('\u2014', '-')  # em dash —
            .replace('\u2013', '-')  # en dash –
            .replace('\u2212', '-')  # minus sign −
        )

        # If no input is provided, return early
        if not string_input:            
            output_text = "No input text provided."
            return {"ui": {"text": output_text}, "result": (string_input,)}

        # Initialize tokenizer 
        tokenizer = self.load_tokenizer()
        token_sequence_length = tokenizer.model_max_length
        num_of_special_tokens = 2

        # Print tokenizer information
        # print(f"tokenizer_max_length = {tokenizer_max_length}", color.YELLOW)
        # print(f"CLIP has both a BOS and an EOS special token:", color.YELLOW)
        
        # Extract special tokens information
        bos_input_id = tokenizer.encode(f'{tokenizer.bos_token}', add_special_tokens=False)[0]
        eos_input_id = tokenizer.encode(f'{tokenizer.eos_token}', add_special_tokens=False)[0]
        bos_token = tokenizer.decode(bos_input_id)
        eos_token = tokenizer.decode(eos_input_id)

        # Tokenize the text (both original and truncated versions)
        raw_input_original = tokenizer(string_input, padding=False, truncation=False)
        raw_input_truncated = tokenizer(string_input, padding=False, truncation=True, max_length=token_sequence_length)

        # Extract input IDs
        input_ids_original = raw_input_original['input_ids']
        input_ids_truncated = raw_input_truncated['input_ids']

        # Get token counts
        token_count_original = len(input_ids_original)
        token_count_truncated = len(input_ids_truncated)
        
        # Convert all token IDs to tokens
        tokens_original = [tokenizer.convert_ids_to_tokens(token_id) for token_id in input_ids_original]
        # Convert first (or up to) 77 token IDs to tokens
        tokens_truncated = [tokenizer.convert_ids_to_tokens(token_id) for token_id in input_ids_truncated]

        # Process limit token information
        last_token = ""
        context_limit_token = ""
        
        # Check if token count is within max length
        if token_count_truncated <= token_sequence_length:
            index = token_sequence_length - num_of_special_tokens
            last_token = tokens_truncated[index] if index < len(tokens_truncated) else ""
            
            # Get context around the limit token
            selected_token_ids = input_ids_truncated[(token_sequence_length-9):index + 1]
            context_limit_token = tokenizer.decode(selected_token_ids, clean_up_tokenization_spaces=True).strip()

        
        # Get formatting settings
        max_tokens_per_line = self.get_tokens_per_line_val(extra_pnginfo, unique_id) 
        tokens_truncated_formatted = self.format_tokens_CLIP(tokens_truncated, max_tokens_per_line, start_index=0)
        tokens_overflow_formatted = self.format_tokens_CLIP(tokens_original,  max_tokens_per_line, start_index=76)      

        output_text = ""
        # Build output text
        # output_text = f"Special Token(s): \n"
        # output_text += f"   1: Beginning-of-Sequence (BOS): '{bos_token}'\n"
        # output_text += f"   2: End-of-Sequence (EOS): '{eos_token}'\n\n"
        # output_text += f"Token Limit: {tokenizer_max_length}\n"
        
        # Add token count information
        # token count
        output_text += f"Token Count: {token_count_original} / {token_sequence_length}"
        over_limit = token_count_original - token_sequence_length
        output_text += f": >>{over_limit} over limit<<\n" if over_limit > 0 else "\n"
        output_text += "\n" if last_token else "\n"

        # Add limit token information if it exists
        if last_token:
            output_text += f'Last Token: "{last_token} | "...{context_limit_token}"\n\n'

        # Show tokens if the widget is enabled
        show_tokens = self.get_show_tokens_val(extra_pnginfo, unique_id)        
        if show_tokens:
            
            # Set appropriate title based on truncation
            title = ""
            if token_count_original > token_sequence_length:
                title = f"Tokens (Truncated - Showing first 77)\n==========================================\n"         
            else: 
                title = f"Tokens\n==========================================\n"
            
            # Add tokens section
            output_text += title
            output_text += f"{tokens_truncated_formatted}\n"

            if tokens_overflow_formatted:
                output_text += f"\nOverflow Tokens\n"                    
                output_text += f"=========================\n"
                output_text += (
                        f"Note: The final token in the truncated output above is the EOS token.\n"
                        f"However, in the overflow output, the token at the same index position [76]\n"
                        f"is an actual text token — and would've appeared in the main sequence\n"
                        f"if truncation (and EOS insertion) hadn't occurred.\n")
                output_text += f"\n{tokens_overflow_formatted}"
        
        # Update node widget values if extra_pnginfo is available
        if not extra_pnginfo:
            print("Error: extra_pnginfo is empty")
        elif (not isinstance(extra_pnginfo, dict) or "workflow" not in extra_pnginfo):
            print("Error: extra_pnginfo is not a dict or missing 'workflow' key")
        else:
            workflow = extra_pnginfo["workflow"]
            node = next((x for x in workflow["nodes"] if str(x["id"]) == unique_id), None)
            if node:
                node["widgets_values"] = [output_text]
        
        # Return both UI info (output_text) and the original input string as output
        # This allows the node to be used as a pass-through while still showing token info
        return {"ui": {"text": output_text}, "result": (string_input,)}


    # ===================================================================================
    def format_tokens_CLIP(self, tokens, max_tokens_per_line=4, start_index=0):

        # Format tokens  into a string with padding, indices, and number of tokens per line.
        # only include tokens that fit into the tokenizer_max_length, ignoring the rest
        # We include the two special tokens BOS and EOS
        # Args: 
        #  - tokens (list): Either a list of first 77 tokens, or could be original list 
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
        
        for i, token in enumerate(tokens[start_index:], start=start_index):
            # Format index with fixed width and pad token
            padded_token = token.ljust(max_str_length)
            tokens_formatted += f"[{index_format.format(i)}] {padded_token}"
            
            # Add a new line after every N tokens
            if ((i - start_index + 1) % max_tokens_per_line == 0):
                tokens_formatted += "\n"
        
        return tokens_formatted

    # ===================================================================================
    def get_show_tokens_val(self, extra_pnginfo, unique_id):
        
        # Get the value of show_tokens widget using the index stored in node properties.
        # that was set in the frontend javascripts.  We find the correct node in the workflow using
        # unique_id        
        # Args:
        #  - extra_pnginfo: The extra PNG info containing workflow data
        #  - unique_id: The unique ID of this node                
        # Returns: 
        #  - The bool from the show_tokens widget, or False if not found
        

        # For widgets created in JavaScript, we can access their vals using the widget 
        # index that is stored in the extra_pnginfo dictionary:
        #   extra_pnginfo-> workflow -> nodes -> properties

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

        #         ],
        #         .....
        #     }
        # }        
        
        property_name = "show_tokens_index"

        # Check if extra_pnginfo exists and it contains workflow info
        if extra_pnginfo and "workflow" in extra_pnginfo:
            # Extract the workflow object from extra_pnginfo
            workflow = extra_pnginfo["workflow"]
            
            # find the node in the WF that matches our unique_id for this node
            # return the first match & convert the id to str
            node = next((x for x in workflow["nodes"] if str(x["id"]) == unique_id), None)
            
            # check node is truthy and it has widgets_values key
            if node and "widgets_values" in node:

                # now check if the node has properties key and contains the show_tokens_index custom property
                if "properties" in node and property_name in node["properties"]:
                    # if so get the value 
                    index = node["properties"][property_name]
                                        
                    # make sure the index is within bounds for the widgets_values array
                    if index < len(node["widgets_values"]):
                        # return the value at that index (should be a boolean in this case)
                        return node["widgets_values"][index]

        # If anything fails (missing data, index out of range, etc.), return False as default
        return False

    # ===================================================================================        
    def get_tokens_per_line_val(self, extra_pnginfo, unique_id):
        # similar in functionality to get_show_tokens_val() above

        property_name = "tokens_per_line_index"

        # Check if extra_pnginfo exists and it contains workflow info
        if extra_pnginfo and "workflow" in extra_pnginfo:
            # Extract the workflow object from extra_pnginfo
            workflow = extra_pnginfo["workflow"]
            
            # find the node in the WF that matches our unique_id for this node
            # return the first match & convert the id to str
            node = next((x for x in workflow["nodes"] if str(x["id"]) == unique_id), None)
            
            # check node is truthy and it has widgets_values key
            if node and "widgets_values" in node:

                # now check if the node has properties key and contains the custom property 
                # we are after (as defined by property_name)
                if "properties" in node and property_name in node["properties"]:
                    # if so get the value 
                    index = node["properties"][property_name]


                    # make sure the index is within bounds for the widgets_values array
                    if index < len(node["widgets_values"]):
                        # return the value at that index. (converted to int) 
                        return int(node["widgets_values"][index])

        # If anything fails (missing data, index out of range, etc.), return 4 as default
        return 4

        
    # ===================================================================================
    def load_tokenizer(self):

        # Load the CLIP tokenizer from local dir    
        # Returns the loaded CLIP tokenizer

        FALLBACK_HF_MODEL = "openai/clip-vit-large-patch14"

        try:
            # Use Path to get the absolute path to the tokenizer directory
            # This matches the approach used in __init__.py for web routes
            from pathlib import Path

            current_dir = Path(__file__).parent.parent.absolute()
            local_tokenizer_path = (current_dir / "text_encoders" / "clip_tokenizer").as_posix()
            # print(f"Loading tokenizer from local path: {local_tokenizer_path}")
            tokenizer = CLIPTokenizer.from_pretrained(local_tokenizer_path)
            return tokenizer

        except Exception as e:
            print(f"Unable loading local tokenizer: files may be missing or corrupt.", color.ORANGE)
            print(f"Downloading from openai/clip-vit-large-patch14 tokenizer on HuggingFace", color.ORANGE)
            
            try:

                # Download the tokenizer
                tokenizer = CLIPTokenizer.from_pretrained(FALLBACK_HF_MODEL)
                
                # Create the local directory if it doesn't exist
                from pathlib import Path
                current_dir = Path(__file__).parent.parent.absolute()
                local_tokenizer_path = (current_dir / "text_encoders" / "clip_tokenizer")
                local_tokenizer_path.mkdir(parents=True, exist_ok=True)
                
                # Save the tokenizer to the local directory
                print(f"Saving tokenizer to local path: {local_tokenizer_path.as_posix()}", color.BRIGHT_GREEN)
                tokenizer.save_pretrained(local_tokenizer_path.as_posix())
                print(f"Tokenizer saved successfully!", color.BRIGHT_GREEN)
                
                return tokenizer
                
            except Exception as save_error:
                print(f"Error saving tokenizer locally: {save_error}\nUsing cached copy.", color.ORANGE)
                
                # If saving fails, still return the downloaded tokenizer
                return CLIPTokenizer.from_pretrained(FALLBACK_HF_MODEL)
                    
        # except Exception as e:
        #     print(f"Unable loading local tokenizer: files may be missing or corrupt.", color.BRIGHT_ORANGE)
        #     print(f"Downloading from openai/clip-vit-large-patch14 tokenizer on HuggingFace",color.ORANGE)            
        #     # This fallback will still try to download, so it won't work offline
        #     return CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
