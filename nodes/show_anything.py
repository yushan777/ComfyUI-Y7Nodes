# Y7 Show Anything node
import json
import torch

from comfy_api.latest import io

from ..utils.colored_print import color


# Code based on :
# - "Show Any" from yolain's ComfyUI-Easy-Use custom nodes 
# - "Show Any To JSON" from crystian's ComfyUI-Crystools custom nodes.
# Additional formatting controls added as well as a Copy Text button. 


# This node has 1 input and 1 output 
# and 2 hidden input types used for state persistence.
# The front-end widgets (Textbox, boolean switch and button) are handled in javascript 
# =====================================================================================
class Y7Nodes_ShowAnything(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Y7Nodes_ShowAnything",
            display_name="Y7 Show Anything",
            category="Y7Nodes/Utils",
            description="Displays any input value for inspection or debugging.",
            # io.AnyType is the "*" wildcard, replacing the hand-rolled AlwaysEqualProxy
            inputs=[
                io.AnyType.Input(
                    "anything",
                    optional=True,
                    tooltip="Any input type that you want to inspect or debug",
                ),
            ],
            outputs=[
                io.AnyType.Output(display_name="output"),
            ],
            is_input_list=True,
            is_output_node=True,
            hidden=[
                io.Hidden.unique_id,
                io.Hidden.extra_pnginfo,
            ],
        )

    # ====================================================================================
    # Helper function to detect IMAGE tensors
    @classmethod
    def is_image_tensor(cls, tensor):
        """Check if a tensor matches the IMAGE datatype format"""
        return (
            isinstance(tensor, torch.Tensor) and 
            len(tensor.shape) == 4 and
            # The last dimension should be the channels (typically 3 for RGB)
            tensor.shape[-1] <= 4
        )
    
    # ====================================================================================
    # Helper function to detect MASK tensors
    @classmethod
    def is_mask_tensor(cls, tensor):
        """Check if a tensor matches the MASK datatype format"""
        return (
            isinstance(tensor, torch.Tensor) and 
            (
                # [H,W] format
                len(tensor.shape) == 2 or
                # [1,H,W] format (single-channel)
                (len(tensor.shape) == 3 and tensor.shape[0] == 1) or
                # [B,C,H,W] format with C=1
                (len(tensor.shape) == 4 and tensor.shape[1] == 1)
            )
        )
    
    # ====================================================================================
    # Helper function to format IMAGE tensor information
    @classmethod
    def format_image_info(cls, tensor):
        """Format useful information about an IMAGE tensor"""
        b, h, w, c = tensor.shape
        
        info = [
            f"IMAGE Tensor:",
            f"  Shape: {tensor.shape}",
            f"   - [Batch={b}, Height={h}, Width={w}, Channels={c}]",
            f"  Data Type: {tensor.dtype}",
            f"  Value Range: [{tensor.min().item():.4f}, {tensor.max().item():.4f}]",
            f"  Mean: {tensor.mean().item():.4f}",
            f"  Std Dev: {tensor.std().item():.4f},\n\n"
        ]

        # print(info)
        return "\n".join(info)
    
    # ====================================================================================
    # Helper function to format MASK tensor information
    @classmethod
    def format_mask_info(cls, tensor):
        """Format useful information about a MASK tensor"""
        if len(tensor.shape) == 2:
            h, w = tensor.shape
            info = [
                f"MASK Tensor:",
                f"  Shape: {tensor.shape} [Height={h}, Width={w}]",
                f"  Data Type: {tensor.dtype}",
                f"  Value Range: [{tensor.min().item():.4f}, {tensor.max().item():.4f}]",
                f"  Mean: {tensor.mean().item():.4f}",
                f"  Std Dev: {tensor.std().item():.4f},\n\n"
            ]
        elif len(tensor.shape) == 3 and tensor.shape[0] == 1:  # [1,H,W] format
            c, h, w = tensor.shape
            info = [
                f"MASK Tensor:",
                f"  Shape: {tensor.shape}",
                f"   - [Channel={c}, Height={h}, Width={w}]",
                f"  Data Type: {tensor.dtype}",
                f"  Value Range: [{tensor.min().item():.4f}, {tensor.max().item():.4f}]",
                f"  Mean: {tensor.mean().item():.4f}",
                f"  Std Dev: {tensor.std().item():.4f},\n\n"
            ]
        else:  # [B,C,H,W] format
            b, c, h, w = tensor.shape
            info = [
                f"MASK Tensor:",
                f"  Shape: {tensor.shape} [Batch={b}, Channel={c}, Height={h}, Width={w}]",
                f"  Data Type: {tensor.dtype}",
                f"  Value Range: [{tensor.min().item():.4f}, {tensor.max().item():.4f}]",
                f"  Mean: {tensor.mean().item():.4f}",
                f"  Std Dev: {tensor.std().item():.4f},\n\n"
            ]
        
        return "\n".join(info)

    # ====================================================================================
    @classmethod
    def format_latent_info(cls, tensor):
        """Format useful information about a latent tensor"""
        b, c, h, w = tensor.shape
        
        info = [
            f"Latent Tensor Dictionary:",
            f"  Shape: {tensor.shape}",
            f"   - [Batch={b}, Channels={c}, Height={h}, Width={w}]",
            f"  Data Type: {tensor.dtype}",
            f"  Value Range: [{tensor.min().item():.4f}, {tensor.max().item():.4f}]",
            f"  Mean: {tensor.mean().item():.4f}",
            f"  Std Dev: {tensor.std().item():.4f},\n\n"
        ]
        
        return "\n".join(info)    
    
    # ====================================================================================
    # main function
    @classmethod
    def execute(cls, anything=None) -> io.NodeOutput:
        # unique_id and extra_pnginfo are declared as hidden inputs in the schema.
        # is_input_list makes `anything` a list, but hidden values arrive unwrapped.
        unique_id = cls.hidden.unique_id
        extra_pnginfo = cls.hidden.extra_pnginfo

        # values list to be returned
        values = []

        if anything is not None:
            for val in anything:
                try:
                    
                    if isinstance(val, (str, int, float, bool)):
                        # print(f"val is {type(val)}", color.RED)
                        # append val for primitive data types
                        values.append(val)                                                            
                    # elif isinstance(val, list):
                    #     print(f"val is a {type(val)}", color.YELLOW)
                    #     # assign whole val for list types (since values is already a list)
                    #     values = val
                    elif isinstance(val, list):
                        # print(f"val is a {type(val)}", color.RED)
                        # Process each item in the list to ensure everything is serializable
                        processed_list = []
                        for item in val:
                            if isinstance(item, torch.Tensor):
                                # Convert tensor to a string representation
                                if cls.is_image_tensor(item):
                                    processed_list.append(cls.format_image_info(item))
                                elif cls.is_mask_tensor(item):
                                    processed_list.append(cls.format_mask_info(item))
                                else:
                                    try:
                                        tensor_info = f"Tensor: Shape={item.shape}, Type={item.dtype}, " \
                                                    f"Range=[{item.min().item():.4f}, {item.max().item():.4f}]"
                                        processed_list.append(tensor_info)
                                    except:
                                        tensor_info = f"Tensor: Shape={item.shape}, Type={item.dtype}"
                                        processed_list.append(tensor_info)
                            elif isinstance(item, (str, int, float, bool)):
                                processed_list.append(item)
                            else:
                                try:
                                    # Try to serialize the item
                                    json_item = json.dumps(item)
                                    processed_list.append(json_item)
                                except:
                                    # If serialization fails, convert to string
                                    processed_list.append(f"Non-serializable object of type: {type(item).__name__}")
                        values.append(processed_list)

                    elif isinstance(val, torch.Tensor):
                        # if tensor has shape of image tensor
                        if cls.is_image_tensor(val):
                            # print(f"val is a {type(val)} (IMAGE)", color.YELLOW)
                            
                            # Format image information
                            values.append(cls.format_image_info(val))
                            val = json.dumps(val)
                            values.append(str(val))
                        elif cls.is_mask_tensor(val):
                            # print(f"val is a {type(val)} (MASK)", color.YELLOW)
                            values.append(cls.format_mask_info(val))
                            val = json.dumps(val)
                            values.append(str(val))
                        else:                        
                            # print(f"val is a {type(val)}", color.YELLOW)
                            # assign whole val for list types (since values is already a list)
                            val = json.dumps(val)
                            values.append(str(val))

                    elif isinstance(val, dict):
                        # print(f"val is a {type(val)}", color.YELLOW)
                        
                        # Check if it's a latent tensor dictionary (has 'samples' key with tensor value)
                        if 'samples' in val and isinstance(val['samples'], torch.Tensor):
                            tensor = val['samples']

                            # Check tensor dimensions - latent tensors are typically 4D with shape [B, C, H, W]
                            if len(tensor.shape) == 4:
                                values.append(cls.format_latent_info(tensor))
                                val = json.dumps(tensor)
                                values.append(str(val))
                            else:
                                # Handle other tensor shapes in the dictionary
                                values.append(f"Dictionary with tensor of shape {tensor.shape}")
                        else:
                            # For other dictionaries, try to convert to JSON
                            try:
                                processed_dict = {}
                                # Process each item in the dictionary
                                for key, item in val.items():
                                    if isinstance(item, torch.Tensor):
                                        processed_dict[key] = f"Tensor with shape {item.shape}"
                                    else:
                                        processed_dict[key] = item
                                
                                # Convert processed dictionary to JSON
                                json_val = json.dumps(processed_dict)
                                values.append(json_val)
                            except:
                                values.append(f"Dictionary with non-serializable contents")
                
                    else:
                        print(f"val is a something else: {type(val)}", color.YELLOW)
                        val = json.dumps(val)
                        values.append(str(val))
                except Exception:
                    values.append(str(val))
                    pass
        else:
            values.append("No input provided.")
            
        print(values)

        if not extra_pnginfo:
            print("Error: extra_pnginfo is empty")
        elif (not isinstance(extra_pnginfo, dict) or "workflow" not in extra_pnginfo):
            print("Error: extra_pnginfo is not a dict or missing 'workflow' key")
        else:
            workflow = extra_pnginfo["workflow"]
            node = next((x for x in workflow["nodes"] if str(x["id"]) == unique_id), None)
            if node:
                node["widgets_values"] = [values]
        if isinstance(values, list) and len(values) == 1:
            return io.NodeOutput(values[0], ui={"text": values})
        else:
            return io.NodeOutput(values, ui={"text": values})
            
    # def log_input(self, unique_id=None, extra_pnginfo=None, **kwargs):
    #     # Initialize an empty list to store our processed values
    #     values = []
        
    #     # Check if "anything" key exists in the input kwargs
    #     if "anything" in kwargs:
    #         # Iterate through each value passed to the "anything" parameter
    #         for val in kwargs['anything']:
    #             try:
    #                 # Check if the value is an IMAGE tensor
    #                 if isinstance(val, torch.Tensor) and cls.is_image_tensor(val):
    #                     # Format image information
    #                     values.append(cls.format_image_info(val))
    #                 # Check if the value is a MASK tensor
    #                 elif isinstance(val, torch.Tensor) and cls.is_mask_tensor(val):
    #                     # Format mask information
    #                     values.append(cls.format_mask_info(val))
    #                 # If the value is a list, just output a simple message
    #                 elif isinstance(val, list): # for conditioning
    #                     values.append("The input is a list, but could not be serialized.")
    #                 # If the value is a string, add it directly to our values list
    #                 elif type(val) is str:
    #                     values.append(val)
    #                 # Handle any other tensor types that aren't IMAGE or MASK
    #                 elif isinstance(val, torch.Tensor):
    #                     try:
    #                         tensor_info = f"Tensor: Shape={val.shape}, Type={val.dtype}, " \
    #                                      f"Range=[{val.min().item():.4f}, {val.max().item():.4f}]"
    #                         values.append(tensor_info)
    #                     except:
    #                         # Some tensors might not support min/max operations
    #                         tensor_info = f"Tensor: Shape={val.shape}, Type={val.dtype}"
    #                         values.append(tensor_info)
    #                 # For other types (dicts, etc.), convert to JSON string
    #                 else:
    #                     try:
    #                         val = json.dumps(val)
    #                         values.append(str(val))
    #                     except TypeError:
    #                         # Handle non-JSON serializable objects
    #                         values.append(f"Non-serializable object of type: {type(val).__name__}")
    #             except Exception as e:
    #                 # If processing fails, convert to string directly and continue
    #                 values.append(f"Error processing value: {str(e)}")
    #                 pass

    #     # ============================================================
    #     # Error handling for workflow information
    #     if not extra_pnginfo:
    #         print("Error: extra_pnginfo is empty")
    #     elif (not isinstance(extra_pnginfo[0], dict) or "workflow" not in extra_pnginfo[0]):
    #         print("Error: extra_pnginfo[0] is not a dict or missing 'workflow' key")
    #     else:
    #         # Get the workflow from extra_pnginfo
    #         workflow = extra_pnginfo[0]["workflow"]
    #         # Find the current node in the workflow by matching its unique_id
    #         node = next((x for x in workflow["nodes"] if str(x["id"]) == unique_id[0]), None)
    #         if node:
    #             # Store the processed values in the node's widgets_values for persistence
    #             node["widgets_values"] = [values]
        
    #     # Return structure depends on the number of values
    #     if isinstance(values, list) and len(values) == 1:
    #         # For a single value, return it directly as the output
    #         # Make sure we're not returning a raw tensor
    #         result = values[0]
    #         # If the result is a tensor, convert it to a string representation
    #         if isinstance(result, torch.Tensor):
    #             result = f"Tensor with shape {result.shape}"
    #         return {"ui": {"text": values}, "result": (result,), }
    #     else:
    #         # For multiple values, return the entire list
    #         # Make sure we're not returning raw tensors
    #         processed_values = []
    #         for val in values:
    #             if isinstance(val, torch.Tensor):
    #                 processed_values.append(f"Tensor with shape {val.shape}")
    #             else:
    #                 processed_values.append(val)
    #         return {"ui": {"text": values}, "result": (processed_values,), }
