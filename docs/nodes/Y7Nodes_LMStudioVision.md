# LM Studio (Vision)

Send an image to a vision-capable LLM in LM Studio for analysis and description.

Connects to an LM Studio server and sends an image along with an instruction to a vision-language (VL) model. The model must be vision-enabled or an error will be raised.

Inputs:

  - `image`: The image to analyze (required)
  - `model_identifier`: The VL model name/identifier loaded in LM Studio (connect a Select LMS Model node or type manually)
  - `system_message`: The instruction sent alongside the image (default describes the image in detail)
  - `reasoning_tag`: Tag name used to extract reasoning blocks (e.g., `think` for `<think>...</think>`)
  - `ip` / `port`: LM Studio server address (default: localhost:1234)
  - `temperature`: Controls randomness (0.01-1.0, default 0.7)
  - `max_tokens`: Maximum tokens to generate (-1 for unlimited)
  - `unload_llm`: Unload the LLM from LM Studio after generation
  - `unload_comfy_models`: Free VRAM by unloading ComfyUI models before running the LLM

Outputs:

  - `Response`: The model's analysis/description with reasoning blocks removed
  - `Reasoning`: The extracted reasoning content (if present)

Requires the `lmstudio` Python package: `pip install lmstudio`

Note: The loaded model must support vision. Non-vision models will raise an error.
