# Show Anything

Display the content of any input, regardless of its type.

A debugging tool that displays information about any input in the ComfyUI interface.

For string, integer, float, boolean values: Displays the content directly

For IMAGE and MASK tensors: Shows shape, data type, value range, mean, and std dev.

For other tensors: Displays shape, data type, and value range

For other types: Converts to JSON or string representation

Pass-through for the input.
