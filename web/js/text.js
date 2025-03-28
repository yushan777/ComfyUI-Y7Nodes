import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

// Widget indexes
// widget 0  = text (created in the backend)
const TEXT_WIDGET = 0;
const COPYBUTTON_WIDGET = 1;

app.registerExtension({
    // name: a unique identifier used by CUI to register and track this extension
    // It can be any string but it's good practice to namespace it with your prefix (Y7)    
    name: "Y7Nodes.Text",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {

        // check if the node being registered is our target node
        // this must match the class mapping name in __init__.py
        if (nodeData.name === "Y7Nodes_Text") {

            // store the original onNodeCreated method
            const onNodeCreated = nodeType.prototype.onNodeCreated;
        
            // When a new instance of this node type is created....
            nodeType.prototype.onNodeCreated = function() {
                
                // override it with our own implementation, but still 
                // preserve the ability to call the original method with
                const result = onNodeCreated?.apply(this, arguments);
                                
                // ==========================================================
                // COPY BUTTON
                // ==========================================================
                // Add button - Copy button
                // serialize: false tells ComfyUI not to save the button's state when 
                // saving the workflow since buttons here are transient
                this.addWidget("button", "📋 Copy Text", null, () => {
                    

                    // get the text from the text widget
                    const textToCopy = this.widgets[TEXT_WIDGET].value;

                    console.log(textToCopy)
                    
                    // Fallback for empty text
                    if (!textToCopy && textToCopy !== 0) {
                        textToCopy = "";
                    }

                    // Copy to clipboard 
                    navigator.clipboard.writeText(textToCopy)
                        .then(() => {
                            // Temporarily change button text to show success
                            const button = this.widgets[COPYBUTTON_WIDGET];  // The button is the 4th widget in array
                            const originalText = button.name;
                            button.name = "✅ Text copied.";
                            
                            // Reset button text after 1 second
                            setTimeout(() => {
                                button.name = originalText;
                                // Redraw the canvas
                                app.graph.setDirtyCanvas(true, false);  
                            }, 1000);
                            
                            // Redraw the canvas
                            app.graph.setDirtyCanvas(true, false);  
                        })
                        .catch(err => {
                            console.error('Failed to copy text:', err);
                            // Show error state
                            const button = this.widgets[COPYBUTTON_WIDGET];
                            const originalText = button.name;
                            button.name = "❌ Failed to copy text.";
                            
                            setTimeout(() => {
                                button.name = originalText;
                                app.graph.setDirtyCanvas(true, false);
                            }, 2000);
                            
                            app.graph.setDirtyCanvas(true, false);
                        });                    

                 
                }, { serialize: false });

                return result;
            };

            // ===========================================================================================
            // When the node is executed
            nodeType.prototype.onExecuted = function (message) {
            };

            // ===========================================================================================
            // When loading a saved workflow
            nodeType.prototype.onConfigure = function () {
            };

            // ================================================================================================

        }
    },
});
