import { app } from "../../../scripts/app.js";

// Default node width (wide enough for the full title)
const DEFAULT_NODE_WIDTH = 300;


app.registerExtension({
    name: "Y7Nodes.KleinQwenEditAlign",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Check if the node being registered is our target node
        // This must match the node_id in the node's schema
        if (nodeData.name === "Y7Nodes_KleinQwenEditAlign") {
            // Store the original method
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            // When a new instance of this node type is created...
            nodeType.prototype.onNodeCreated = function() {
                // Call the original method first
                onNodeCreated?.apply(this, arguments);

                // Widen the node so the title isn't cropped
                if (this.size[0] < DEFAULT_NODE_WIDTH) {
                    this.size[0] = DEFAULT_NODE_WIDTH;

                    if (this.onResize) {
                        this.onResize(this.size);
                    }

                    app.graph.setDirtyCanvas(true, false);
                }
            };
        }
    },
});
