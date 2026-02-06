import { app } from "../../../scripts/app.js";

// Default node size
const DEFAULT_NODE_WIDTH = 250;
const DEFAULT_NODE_HEIGHT = 150;


app.registerExtension({
    name: "Y7Nodes.CropToResolution",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Check if the node being registered is our target node
        // This must match the class mapping name in __init__.py
        if (nodeData.name === "CropToResolution") {
            // Store the original onNodeCreated method
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            // When a new instance of this node type is created...
            nodeType.prototype.onNodeCreated = function() {
                // Call the original method first
                onNodeCreated?.apply(this, arguments);

                // Set a default initial size for the node
                // [width, height]
                this.size = [DEFAULT_NODE_WIDTH, DEFAULT_NODE_HEIGHT]; 
            };
        }
    },
});
