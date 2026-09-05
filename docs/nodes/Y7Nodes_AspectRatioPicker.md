# Aspect Ratio Picker

Interactive 2D canvas for picking image width and height by dragging.

Click or drag anywhere in the canvas to set width (X axis, left→right) and height (Y axis, bottom→top).

A filled rectangle shows the selected proportions; the dot marks the current position.

The simplified aspect ratio (e.g. 16:9, 4:3, 1:1) is displayed at the bottom of the canvas.

Current width and height values are shown in the right panel. Double-click either value to type a number directly.

Click the Swap button (below the height value) to swap width and height (portrait ↔ landscape).

Snapping to step increments is on by default. Hold Shift while dragging to temporarily disable snapping.

Node properties (right-click → Properties):

  - `valueX` / `valueY`: Current width and height.
  - `minX` / `maxX` / `minY` / `maxY`: Range for each axis.
  - `stepX` / `stepY`: Snap increment for each axis.
  - `snap`: Whether dragging snaps to step increments by default.
  - `dots`: Show grid dots at each step position.

Outputs:

  - `width`: Selected width in pixels (INT).
  - `height`: Selected height in pixels (INT).

The right panel also shows the total megapixel count (e.g. `1.0MP` at 1024×1024).
