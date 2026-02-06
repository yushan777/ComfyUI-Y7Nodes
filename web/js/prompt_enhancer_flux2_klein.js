import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";


// Default node size
const DEFAULT_NODE_WIDTH = 300;
const DEFAULT_NODE_HEIGHT = 250;


app.registerExtension({
    // name: a unique identifier used by CUI to register and track this extension
    // It can be any string but it's good practice to namespace it with your prefix (Y7)    
    name: "Y7Nodes.PromptEnhancerFlux2",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {

        // check if the node being registered is our target node
        // this must match the class mapping name in __init__.py
        if (nodeData.name === "Y7Nodes_PromptEnhancerFlux2") {

            // store the original onNodeCreated method
            const onNodeCreated = nodeType.prototype.onNodeCreated;
        
            // When a new instance of this node type is created....
            nodeType.prototype.onNodeCreated = function() {
                                
                // override it with our own implementation, but still 
                // preserve the ability to call the original method with
                onNodeCreated?.apply(this, arguments);
                
                // ==========================================================
                // SET INITIAL NODE SIZE
                // ==========================================================
                // Set the initial size of the node
                // this.size[0] = width
                // this.size[1] = height
                if (this.size[0] < DEFAULT_NODE_WIDTH || this.size[1] < DEFAULT_NODE_HEIGHT) {
                    this.size[0] = DEFAULT_NODE_WIDTH;
                    this.size[1] = DEFAULT_NODE_HEIGHT;
                    
                    // Apply the resize
                    if (this.onResize) {
                        this.onResize(this.size);
                    }
                    
                    // Tell ComfyUI to redraw
                    app.graph.setDirtyCanvas(true, false);
                }
                
            };

            // ===========================================================================================
            // When the node is executed
            nodeType.prototype.onExecuted = function (message) {

            };

            // ===========================================================================================
            // When loading a saved workflow
            nodeType.prototype.onConfigure = function () {

            };

    

        }
    },
});
