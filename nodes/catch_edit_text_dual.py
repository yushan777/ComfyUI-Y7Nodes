# Node Name: CatchEditTextNodeDual
# Based on the original CatchEditTextNode by ImagineerNL
# https://github.com/ImagineerNL/ComfyUI-IMGNR-Utils
# Modified to use dual inputs and ouputs
# Version: 1.0.

class Y7Nodes_CatchEditTextNodeDual:
    """
    Catches up to two text inputs, displays them in editable widgets.
    'Use Input': Outputs the original inputs, attempts to update widget displays.
    'Use Edit & Mute Input': Outputs widget texts, mutes upstream nodes providing inputs.
    Inputs are optional to allow upstream muting without validation errors.
    """

    @classmethod
    def INPUT_TYPES(cls):
        widget_default_text_1 = (
            "Widget 1: Catches and shows text from 'input_text_1'.\n"
            "Enables editing the text for subsequent runs.\n"
            "Output is controlled by 'action' below.\n"
            "- Use Input: Outputs the connected text, updates this view.\n"
            "- Use Edit & Mute Input: Outputs the (edited) text from this widget, mutes input node 1."
        )
        widget_default_text_2 = (
            "Widget 2: Catches and shows text from 'input_text_2'.\n"
            "Enables editing the text for subsequent runs.\n"
            "Output is controlled by 'action' below.\n"
            "- Use Input: Outputs the connected text, updates this view.\n"
            "- Use Edit & Mute Input: Outputs the (edited) text from this widget, mutes input node 2."
        )
        return {
            "required": {
                "editable_text_widget_1": ("STRING", {
                    "multiline": True,
                    "default": widget_default_text_1
                }),
                 "editable_text_widget_2": ("STRING", {
                    "multiline": True,
                    "default": widget_default_text_2
                }),
                "action": (
                    ["use_input", "use_edit_mute_input"],
                    {"default": "use_input"}
                )
            },
            "optional": {
                 "input_text_1": ("STRING", {"forceInput": True, "default": ""}),
                 "input_text_2": ("STRING", {"forceInput": True, "default": ""}) 
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("output_text_1", "output_text_2",)
    FUNCTION = "process_text"
    CATEGORY = "Y7Nodes/utils"
    OUTPUT_NODE = True

    # --- Helper functions ---
    def find_node_by_id(self, unique_id, workflow_info):
        if not workflow_info or "nodes" not in workflow_info: print(f"[{self.__class__.__name__}] Helper Error: Invalid workflow_info."); return None 
        target_id = str(unique_id[0]) if isinstance(unique_id, list) else str(unique_id)
        for node_data in workflow_info["nodes"]:
            if str(node_data.get("id")) == target_id: return node_data
        print(f"[{self.__class__.__name__}] Helper Error: Node ID {target_id} not found in workflow."); return None 

    def find_widget_index(self, node_data, widget_name):
        req_keys = list(self.INPUT_TYPES().get("required", {}).keys())
        opt_keys = list(self.INPUT_TYPES().get("optional", {}).keys())
        all_keys = req_keys + opt_keys
        try:
            idx = all_keys.index(widget_name)
            # print(f"[{self.__class__.__name__}] Found widget '{widget_name}' at combined index {idx}.") # Optional log
            return idx
        except ValueError:
            print(f"[{self.__class__.__name__}] Helper Error: Widget '{widget_name}' not found in INPUT_TYPES keys: {all_keys}") 
            return None

    # --- Main Processing Function ---
    def process_text(self, editable_text_widget_1: str, editable_text_widget_2: str, action: str, unique_id=None, extra_pnginfo=None, input_text_1: str = None, input_text_2: str = None):
        output_text_1 = ""
        output_text_2 = ""
        text_for_widget_update_1 = None
        text_for_widget_update_2 = None
        class_name_log = self.__class__.__name__ # For logging

        # Use defaults if inputs are None
        effective_input_text_1 = input_text_1 if input_text_1 is not None else self.INPUT_TYPES()['optional']['input_text_1'][1].get('default', '')
        effective_input_text_2 = input_text_2 if input_text_2 is not None else self.INPUT_TYPES()['optional']['input_text_2'][1].get('default', '')

        print(f"[{class_name_log}] Action: '{action}', Node ID: {unique_id}")

        if action == "use_input":
            output_text_1 = effective_input_text_1
            output_text_2 = effective_input_text_2
            text_for_widget_update_1 = output_text_1
            text_for_widget_update_2 = output_text_2
            print(f"[{class_name_log}] Chose 'use_input'. Outputting received/default inputs. Attempting UI widget updates.")
            if input_text_1 is None: print(f"[{class_name_log}] Info: Input 1 using default (disconnected?).")
            if input_text_2 is None: print(f"[{class_name_log}] Info: Input 2 using default (disconnected?).")

        elif action == "use_edit_mute_input":
            output_text_1 = editable_text_widget_1
            output_text_2 = editable_text_widget_2
            print(f"[{class_name_log}] Chose 'use_edit_mute_input'. Outputting widget texts.")
        else:
             print(f"[{class_name_log}] Warning: Unknown action '{action}'. Defaulting to outputting widget texts.")
             output_text_1 = editable_text_widget_1
             output_text_2 = editable_text_widget_2

        # --- Attempt to update the UI widgets ---
        node_data_updated = False
        if (text_for_widget_update_1 is not None or text_for_widget_update_2 is not None) and unique_id and extra_pnginfo:
            print(f"[{class_name_log}] Attempting UI widget update(s) for node {unique_id[0]}...")
            current_workflow_info = extra_pnginfo[0] if isinstance(extra_pnginfo, list) and extra_pnginfo else extra_pnginfo
            if current_workflow_info and isinstance(current_workflow_info, dict) and "workflow" in current_workflow_info:
                node_data = self.find_node_by_id(unique_id, current_workflow_info["workflow"])
                if node_data:
                    # Update Widget 1 if needed
                    if text_for_widget_update_1 is not None:
                        widget_index_1 = self.find_widget_index(node_data, "editable_text_widget_1")
                        if widget_index_1 is not None:
                            if "widgets_values" not in node_data or not isinstance(node_data["widgets_values"], list):
                                req_widgets = len(self.INPUT_TYPES().get("required", {}))
                                opt_widgets = len(self.INPUT_TYPES().get("optional", {}))
                                num_widgets = req_widgets + opt_widgets
                                node_data["widgets_values"] = ["" for _ in range(num_widgets)]; print(f"[{class_name_log}] Initialized/Reset widgets_values.")
                            while len(node_data["widgets_values"]) <= widget_index_1: node_data["widgets_values"].append(""); print(f"[{class_name_log}] Padded widgets_values for index {widget_index_1}.")
                            current_widget_val_1 = node_data["widgets_values"][widget_index_1]
                            if current_widget_val_1 != text_for_widget_update_1:
                                node_data["widgets_values"][widget_index_1] = text_for_widget_update_1; print(f"[{class_name_log}] ---> Set widgets_values[{widget_index_1}] (Widget 1)."); node_data_updated = True
                            else: print(f"[{class_name_log}] Widget 1 value already matches target.")
                    # Update Widget 2 if needed
                    if text_for_widget_update_2 is not None:
                        widget_index_2 = self.find_widget_index(node_data, "editable_text_widget_2")
                        if widget_index_2 is not None:
                            if "widgets_values" not in node_data or not isinstance(node_data["widgets_values"], list): # Check again in case it wasn't initialized above
                                req_widgets = len(self.INPUT_TYPES().get("required", {}))
                                opt_widgets = len(self.INPUT_TYPES().get("optional", {}))
                                num_widgets = req_widgets + opt_widgets
                                node_data["widgets_values"] = ["" for _ in range(num_widgets)]; print(f"[{class_name_log}] Initialized/Reset widgets_values.")
                            while len(node_data["widgets_values"]) <= widget_index_2: node_data["widgets_values"].append(""); print(f"[{class_name_log}] Padded widgets_values for index {widget_index_2}.")
                            current_widget_val_2 = node_data["widgets_values"][widget_index_2]
                            if current_widget_val_2 != text_for_widget_update_2:
                                node_data["widgets_values"][widget_index_2] = text_for_widget_update_2; print(f"[{class_name_log}] ---> Set widgets_values[{widget_index_2}] (Widget 2)."); node_data_updated = True
                            else: print(f"[{class_name_log}] Widget 2 value already matches target.")
            elif text_for_widget_update_1 is not None or text_for_widget_update_2 is not None: print(f"[{class_name_log}] Cannot attempt UI update - missing unique_id or extra_pnginfo.")

        # Determine text to show in UI (used by JS)
        text_to_show_in_ui_1 = text_for_widget_update_1 if text_for_widget_update_1 is not None else editable_text_widget_1
        text_to_show_in_ui_2 = text_for_widget_update_2 if text_for_widget_update_2 is not None else editable_text_widget_2

        print(f"[{class_name_log}] Final Output 1: '{str(output_text_1)[:60]}...'")
        print(f"[{class_name_log}] Final Output 2: '{str(output_text_2)[:60]}...'")

        # Return structure: UI update info and actual node outputs
        return_dict = {
            "ui": {"text": [str(text_to_show_in_ui_1), str(text_to_show_in_ui_2)]}, # JS expects an array of strings
            "result": (str(output_text_1), str(output_text_2),) # Tuple for multiple outputs
        }
        return return_dict

