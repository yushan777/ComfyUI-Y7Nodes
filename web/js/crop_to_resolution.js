import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

app.registerExtension({
    name: "Y7Nodes.CropToResolution",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        
        // Check if the node being registered is our target node
        // This must match the class mapping name in __init__.py
        if (nodeData.name === "Y7Nodes_CropToResolution") {

            // Store the original onNodeCreated method
            const onNodeCreated = nodeType.prototype.onNodeCreated;
        
            // When a new instance of this node type is created
            nodeType.prototype.onNodeCreated = function() {
                
                // Call the original onNodeCreated method if it exists
                onNodeCreated?.apply(this, arguments);
                
                // Custom initialization code can go here if needed
                console.log("Y7Nodes_CropToResolution node created");
            };

            // When the node is executed
            nodeType.prototype.onExecuted = function (message) {
                // The message parameter contains the output from the backend
                // message.text will contain our dimension check message
                if (message && message.text) {
                    console.log("Dimension Check Result:", message.text);
                    
                    // You can add visual feedback here if needed
                    // For example, changing node color based on validation result
                    if (message.text[0].includes("✓")) {
                        // Dimensions are valid
                        this.boxcolor = "#00ff00"; // Green
                    } else if (message.text[0].includes("✗")) {
                        // Dimensions need adjustment
                        this.boxcolor = "#ff9800"; // Orange
                    }
                    
                    app.graph.setDirtyCanvas(true, false);
                }
            };

            // When loading a saved workflow
            nodeType.prototype.onConfigure = function () {
                // Reset box color when loading
                this.boxcolor = null;
            };
        }
    },
});
