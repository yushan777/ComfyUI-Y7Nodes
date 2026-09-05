# LM Studio (Text)

Send text prompts to a local LM Studio server for text generation and prompt enhancement.

Connects to an LM Studio server to generate or enhance text using a specified LLM. Uses a system message to guide the model's behavior and a user prompt as input.

Inputs:

  - `prompt`: The text prompt to send to the LLM
  - `model_identifier`: The model name/identifier loaded in LM Studio (connect a Select LMS Model node or type manually)
  - `draft_model`: Optional speculative decoding draft model name (leave empty to disable)
  - `system_message`: System prompt that guides the LLM's behavior (default is optimized for image prompt enhancement)
  - `reasoning_tag`: Tag name used to extract reasoning blocks (e.g., `think` for `<think>...</think>`)
  - `ip` / `port`: LM Studio server address (default: localhost:1234)
  - `temperature`: Controls randomness (0.01-1.0, default 0.7)
  - `max_tokens`: Maximum tokens to generate (-1 for unlimited)
  - `unload_llm`: Unload the LLM from LM Studio after generation
  - `unload_comfy_models`: Free VRAM by unloading ComfyUI models before running the LLM

Outputs:

  - `Extended Prompt`: The generated text with reasoning blocks removed
  - `Reasoning`: The extracted reasoning content (if present)

Requires the `lmstudio` Python package: `pip install lmstudio`
