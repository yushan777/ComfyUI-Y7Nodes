# Prompt Enhancer (Native)

Takes a basic prompt and enhances it using any generation-capable text encoder already loaded by ComfyUI

Unlike the other Y7 prompt enhancers, this node downloads and loads nothing itself. It takes a `CLIP` input from a standard `CLIPLoader` and lets ComfyUI handle all model loading and VRAM management. Nothing stays resident that ComfyUI is not already managing.

Supported models:

  - `Gemma 4` - E2B, E4B, 31B and the 12B unified model. Also handles image, video and audio, though this node only sends text.
  - `Gemma 3` - the 12B model, including the LTX-2 text encoder built on it.
  - `Qwen3` - 0.6B, 2B, 4B and 8B.
  - `Qwen3.5` - 0.8B, 2B, 4B, 9B and 27B.
  - `Qwen3-VL` - 4B and 8B.

Not supported - these encoders have no text generation path and the node will say so rather than failing obscurely:

  - `T5` (all sizes), `UMT5`, `CLIP-L`, `CLIP-G`, `Gemma 2`, `LLaMA-3.1`

Weights must be safetensors placed in `models/text_encoders/`. GGUF will not work: ComfyUI core cannot load GGUF at all, and for Gemma 4 the tokenizer is embedded inside the safetensors file itself. The `CLIPLoader` `type` dropdown is ignored for Gemma 4 - the model is detected from the weights, so any value works.

Inputs:

  - `clip`: A generation-capable text encoder from `CLIPLoader` - see the supported models above.
  - `text`: Your basic prompt to enhance. Accepts a connection from any string node.
  - `instruction`: The instruction placed before your text. Edit this to change the style of enhancement.
  - `max_length`: Maximum NEW tokens to generate (64-32768, default 2048). This is not the context window. Reasoning is spent from the same budget, and the KV cache reserves ~84KB of VRAM per token up front.
  - `temperature`: Controls randomness (0.0-2.0, default 1.0, Google's recommended value for Gemma). 0 switches to greedy decoding and ignores top_k/top_p.
  - `top_k`: Limits selection to the k most likely tokens (0-1000, default 64, Google's recommended value). 0 disables.
  - `top_p`: Nucleus sampling (0.0-1.0, default 0.95, Google's recommended value). 1.0 disables.
  - `seed`: Random seed. Change it to re-roll - identical inputs return a cached result.
  - `thinking`: Let the model reason before answering. Its reasoning is always separated out, never mixed into the prompt.

Outputs:

  - `thinking_output`: The model's reasoning, if it produced any
  - `enhanced_prompt`: The enhanced prompt, with all reasoning removed

On reasoning:

Gemma 4 emits reasoning inside a thought channel, and ComfyUI's decoder deliberately keeps that text rather than discarding it. Setting `thinking` to False only primes the model to skip it - Gemma 4 frequently reasons anyway, which is why the core `Generate Text` node can return a wall of planning notes ahead of the actual prompt.

This node always splits the two apart, including the awkward case where the model reasons past the primed channel and closes it with an orphan tag. If `enhanced_prompt` ever comes back empty, the model spent the whole `max_length` budget reasoning - raise it, or lower `temperature`.
