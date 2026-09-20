# Float

A float value, like the native Float node, with the option to step itself up or down by a set amount each time a prompt is queued.

Inputs:

  - `value`: The number sent to the output. Shown to 2 decimal places
  - `control_after_generate`: `fixed` (default) leaves the value where it is. `increment` adds `step` to it, `decrement` subtracts `step`, each time a prompt is queued
  - `step`: How much `increment` / `decrement` moves the value by. Default `0.1`. Ignored when `control_after_generate` is `fixed`

Outputs:

  - `value`: The float value

## Notes [collapsed]

The stepping happens in the browser, the same way the native `control_after_generate` widget moves a seed. The difference is the amount: the native widget always steps by the widget's own step size, this one steps by whatever `step` is set to.

Whether the value moves before or after the prompt is sent follows ComfyUI's own `Widget Control Mode` setting. On `after` (the default) the number on the node is the one that will be used next; on `before` it's the one that was just used.

When several prompts are queued at once with a batch count, the value moves once per prompt, so each one gets its own number.

`value` steps by `0.01` when dragged or typed into, which is where the 2 decimal places come from. A `step` finer than that still works, but the node will round the displayed number.

Converting `value` to an input and connecting something to it turns the stepping off - the incoming link supplies the number instead.
