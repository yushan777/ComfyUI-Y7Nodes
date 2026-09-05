# Sampler Select (Name)

Select a sampler by name and output it as a linkable string.

ComfyUI's built-in KSampler nodes define `sampler_name` as a COMBO widget with no input socket, so it can't receive a node connection. This node exposes the selection as a typed output that can be wired into any node that accepts a sampler name.

Output:

  - `sampler_name`: The selected sampler name
