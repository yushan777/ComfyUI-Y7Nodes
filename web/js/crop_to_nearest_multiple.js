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
            // Store the original onNodeCreated method
            const onNodeCreated = nodeType.prototype.onNodeCreated;

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
