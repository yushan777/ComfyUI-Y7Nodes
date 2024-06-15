import os 

# tokenization process splits the tokens into multiple lists when the number of tokens exceeds a certain threshold (77).

# 49406 = BOS (beginning of sentence)
# 49407 = EOS (end of sentence)


# FOR SD1
# tokens = {
#     'l': [
#         [(49406, 1.0), (25602, 1.0), (1125, 1.0), ....]
#     ]
# }

# FOR SD2
# tokens = {
#     'h': [
#         [(49406, 1.0), (25602, 1.0), (1125, 1.0), ....]
#     ]
# }


# FOR SDXL
# for some reason g pads with 0, and l pads with 49407

# tokens = {
#     'g': [
#         [(49406, 1.0), (25602, 1.0), (1125, 1.0), ...],
#         [(49406, 1.0), (12953, 1.0), (267, 1.0), ...]
#     ],
#     'l': [
#         [(49406, 1.0), (25602, 1.0), (1125, 1.0), ...],
#         [(49406, 1.0), (12953, 1.0), (267, 1.0), ...]
#     ]
# }

import logging

class CountTokens:
    logging.basicConfig(level=logging.DEBUG)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"forceInput": True}),
                "show_token_word_pairs": ("BOOLEAN", {"default": False}),
                "token_word_pairs_per_line": ("INT", {"default": 4, "min": 1, "max": 10, "step": 1}),
                "hide_end_of_word_tag": ("BOOLEAN", {"default": True}),
                "text": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ('STRING',)
    FUNCTION = "count_tokens"
    CATEGORY = "Y7/Tokens"

    def count_tokens(self, clip, text, show_token_word_pairs, token_word_pairs_per_line, hide_end_of_word_tag):
        try:
            tokens = clip.tokenize(text)

            print(tokens)
            model_type = self.detect_model(tokens)
            logging.info("Counting Tokens for text: %s", text)

            summary_text, tokenid_word_pairs_string = '', ''
            
            summary_text = self.count_tokens_for_model(tokens, clip, model_type, hide_end_of_word_tag)
            
            if show_token_word_pairs == True:
                tokenid_word_pairs_string = self.process_token_word_pairs(tokens, clip, model_type, token_word_pairs_per_line, hide_end_of_word_tag)

            # logging.info("Total Token Count: %d", token_count)
            merged_text = summary_text + "\n" + tokenid_word_pairs_string

            return merged_text, # comma needed as we are returning a single-element tuple

        except Exception as e:
            logging.error("Error in count_tokens: %s", str(e))
            return 0, str(e), ""

    # ===================================================================
    def detect_model(self, tokens):
        if 'g' in tokens and 'l' in tokens:
            logging.info('sdxl model detected')
            return 'sdxl'
        elif 'g' not in tokens and 'l' in tokens:
            logging.info('sd1 model detected')
            return 'sd1'
        elif 'g' not in tokens and 'l' not in tokens and 'h' in tokens:
            logging.info('sd2 model detected')
            return 'sd2'
        
        return 'none'

    
    # ===================================================================
    def count_tokens_for_model(self, tokens, clip, model_type, hide_end_of_word_tag):
        text_output = f'{model_type.upper()} CLIP MODEL\n'
        
        total_token_count = 0

        if model_type == 'sdxl':
            token_key = 'g'
        elif model_type == 'sd1':
            token_key = 'l'
        elif model_type == 'sd2':
            token_key = 'h'
         

        # number of lists of tokens, (broken up into sets of 77 if more than than that number)
        num_of_lists = len(tokens[token_key])
            
        token_sublists = []
        for index, token_list in enumerate(tokens[token_key]):
        
            real_tokens = self.filter_padding_tokens(token_list, model_type)

            total_token_count += len(real_tokens)
            
            token_sublists.append(f'Chunk[{index}] = {len(real_tokens)} tokens\n')
        

        if num_of_lists == 1:
            text_output += f'Total Token Count: {total_token_count}, in {num_of_lists} chunk\n'
        elif num_of_lists > 1:
            text_output += f'Total Token Count: {total_token_count}, split across {num_of_lists} chunks\n'
        
        

        for tlist in token_sublists:
            text_output += tlist

        return text_output

    # ===================================================================
    def process_token_word_pairs(self, tokens, clip, model_type, token_word_pairs_per_line, hide_end_of_word_tag):
        tokenid_word_pairs_string = ''

        if model_type == 'sdxl':
            token_key = 'g'
        elif model_type == 'sd1':
            token_key = 'l'
        elif model_type == 'sd2':
            token_key = 'h'

        for index, token_list in enumerate(tokens[token_key]):
            real_tokens = self.filter_padding_tokens(token_list, model_type)
            word_token_pairs = clip.tokenizer.untokenize(real_tokens)

            # token values max out at 5 digits
            formatted_pairs = [
                f"{str(token[0]).zfill(5)}: {self.remove_end_of_word_tag(word) if hide_end_of_word_tag else word}"
                for token, word in word_token_pairs
            ]

            
            tokenid_word_pairs_string += f'=========== Chunk[{index}] : {len(real_tokens)} tokens ===========\n'
            
            longest_word = self.find_longest_word(word_token_pairs, hide_end_of_word_tag)
            tokenid_word_pairs_string += self.format_token_pairs(formatted_pairs, len(longest_word), token_word_pairs_per_line)
            tokenid_word_pairs_string += "\n"

        return  tokenid_word_pairs_string
    # ===================================================================
    # FILTER OUT PADDING TOKENS tokens
    def filter_padding_tokens(self, token_list, model_type):
        # SD1 CLIP USES EOS TOKEN FOR PADDING
        # SDXL CLIP USES 0 FOR PADDING
        real_tokens = []
        encountered_eos = False
        for token in token_list:

            if model_type == 'sdxl':
                if token[0] != 0:
                    real_tokens.append(token)

            elif model_type == 'sd1':
                # if EOS token encountered, add the first instance only
                if token[0] == 49407:
                    if not encountered_eos:
                        encountered_eos = True
                        real_tokens.append(token)
                else:
                    real_tokens.append(token)

            elif model_type == 'sd2':
                if token[0] != 0:
                    real_tokens.append(token)
        return real_tokens
    
    # ===================================================================
    def find_longest_word(self, word_token_pairs, hide_end_of_word_tag):
        longest_word = ""
        for token, word in word_token_pairs:
            # ignore BOS and EOS
            if token[0] != 49406 and token[0] != 49407:
                if hide_end_of_word_tag:
                    word = self.remove_end_of_word_tag(word)
                if len(word) > len(longest_word):
                    longest_word = word
        
        print(f"longuest word = {longest_word}")
        return longest_word

    # ===================================================================
    def remove_end_of_word_tag(self, input_string):
        eow_tag = '</w>'
        if input_string.endswith(eow_tag):
            return input_string[:-len(eow_tag)]
        return input_string

    # ===================================================================
    def format_token_pairs(self, formatted_pairs, longest_word_length, token_word_pairs_per_line):
        num_of_token_pairs_per_line = token_word_pairs_per_line
        formatted_string = ""
        counter = 0  # Counter to track the number of token pairs in the current line

        for pair in formatted_pairs:
            token_id, word = pair.split(": ")  # Split the token and the word
            padding = longest_word_length - len(word)  # Calculate the padding needed to align the word

            # Check if the token ID is either 49406 or 49407, which should be on their own lines
            if token_id == "49406" or token_id == "49407":
                if counter != 0:  # If there are existing tokens on the current line, add a newline before
                    formatted_string += "\n"
                    counter = 0  # Reset the counter
                formatted_string += f"{token_id}: {word}\n"  # Add to its own line
            else:
                # Add the token and word with padding and additional 3 spaces
                formatted_string += f"{token_id}: {word}{' ' * padding}   "
                counter += 1  # Increment the counter

                # If there are {num_of_token_pairs_per_line} token pairs in the line, add a newline and reset the counter
                if counter >= num_of_token_pairs_per_line:
                    formatted_string += "\n"
                    counter = 0

        # Ensure the formatted string ends with a newline
        if not formatted_string.endswith("\n"):
            formatted_string += "\n"
        
        return formatted_string


    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

        return float("NaN")


NODE_CLASS_MAPPINGS = {
    "Count_Tokens_(Y7)": CountTokens,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Count_Tokens_(Y7)": "Count Tokens (Y7)",
}
