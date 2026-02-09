import { app } from "../../../scripts/app.js";

// Default node size
const DEFAULT_NODE_WIDTH = 230;
const DEFAULT_NODE_HEIGHT = 150;


app.registerExtension({
    name: "Y7Nodes.CropToNearestMultiple",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Check if the node being registered is our target node
        // This must match the class mapping name in __init__.py
        if (nodeData.name === "Y7Nodes_CropToNearestMultiple") {
            // Store the original methods
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            const onDrawForeground = nodeType.prototype.onDrawForeground;

            // Override onDrawForeground to display cropped dimensions on the node
            nodeType.prototype.onDrawForeground = function(ctx) {
                const r = onDrawForeground?.apply?.(this, arguments);

                // Get the UI output data for this node
                const v = app.nodeOutputs?.[this.id + ""];
                if (!this.flags.collapsed && v?.text) {
                    const text = v.text[0] + "";
                    ctx.save();
                    ctx.font = "bold 12px sans-serif";
                    ctx.fillStyle = "dodgerblue";
                    const text_size = ctx.measureText(text);
                    // Draw at fixed position near top right (below title, above widgets)
                    const x_pos = 20 //this.size[0] - text_size.width - 5
                    const y_pos = LiteGraph.NODE_SLOT_HEIGHT * 3
                    ctx.fillText(text, x_pos, y_pos);
                    ctx.restore();
                }

                return r;
            };

            // When a new instance of this node type is created...
            nodeType.prototype.onNodeCreated = function() {
                // Call the original method first
                onNodeCreated?.apply(this, arguments);

                console.log("===> Node Size = ", this.size);

                // Set a default initial size for the node
                // [width, height]
                // ==========================================================
                // SET INITIAL NODE SIZE
                // ==========================================================                
                // Set the initial size of the node
                let resized = false;

                if (this.size[0] < DEFAULT_NODE_WIDTH) {
                    this.size[0] = DEFAULT_NODE_WIDTH;
                    resized = true;
                }

                if (this.size[1] < DEFAULT_NODE_HEIGHT) {
                    this.size[1] = DEFAULT_NODE_HEIGHT;
                    resized = true;
                }

                if (resized) {
                    if (this.onResize) {
                        this.onResize(this.size);
                    }

                    app.graph.setDirtyCanvas(true, false);
                }

            };
        }
    },
});
