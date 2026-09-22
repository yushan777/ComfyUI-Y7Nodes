# Switch (model)

Pick one of several connected models by number. Only the selected model is loaded.

Inputs:

  - `select`: Which `model_N` socket to pass through. `1` picks `model_1`, and so on. If that socket isn't connected, the nearest connected one below it is used
  - `model_1` ... `model_10`: Models to choose from. The node starts with one socket, and another empty one appears each time the last is connected

Outputs:

  - `model`: The selected model
  - `selected_index`: The socket number actually used, after any clamping

## Notes [collapsed]

The loaders feeding the sockets that aren't selected never run, so switching between several large diffusion models only costs the load time and memory of the one in use.

The `select` widget follows the connected sockets: its maximum is the highest connected socket, and connecting, disconnecting or loading a workflow moves an out-of-range value down to one that's connected, so the number on the node is the one that runs.

When `select` comes from a link instead, its value isn't known until the workflow runs, so the widget can't adjust it. If it points at a socket that isn't connected, the node falls back to the highest connected socket below it - so setting `select` past the last model just picks the last one, and a gap left by disconnecting a middle socket picks the one before it. If there is no connected socket below it, the lowest connected one is used. The node only stops with an error when no model is connected at all.

Based on Impact Pack's `Switch (Any)`, but limited to `MODEL` sockets so only model outputs can be wired in.
