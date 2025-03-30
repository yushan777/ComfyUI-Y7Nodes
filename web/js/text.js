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
                    // Get the text from the text widget
                    let textToCopy = this.widgets[TEXT_WIDGET].value;
                    
                    console.log("textToCopy =====> \n " + textToCopy);
                    
                    // Fallback for empty text
                    if (!textToCopy && textToCopy !== 0) {
                        textToCopy = "";
                    }

                    // Function to show success message
                    const showSuccess = () => {
                        const button = this.widgets[COPYBUTTON_WIDGET];
                        const originalText = button.name;
                        button.name = "✅ Text copied.";
                        
                        // Reset button text after 1 second
                        setTimeout(() => {
                            button.name = originalText;
                            app.graph.setDirtyCanvas(true, false);
                        }, 1000);
                        
                        app.graph.setDirtyCanvas(true, false);
                    };
                    
                    // Function to show error message
                    const showError = (err) => {
                        console.error('Failed to copy text:', err);
                        const button = this.widgets[COPYBUTTON_WIDGET];
                        const originalText = button.name;
                        button.name = "❌ Failed to copy text.";
                        
                        setTimeout(() => {
                            button.name = originalText;
                            app.graph.setDirtyCanvas(true, false);
                        }, 2000);
                        
                        app.graph.setDirtyCanvas(true, false);
                    };
                    
                    // Use only the fallback method which is more reliable in this environment
                    try {
                        // Create a temporary textarea element
                        const textarea = document.createElement('textarea');
                        textarea.value = textToCopy;
                        // Make the textarea out of viewport
                        textarea.style.position = 'fixed';
                        textarea.style.left = '-999999px';
                        textarea.style.top = '-999999px';
                        document.body.appendChild(textarea);
                        textarea.focus();
                        textarea.select();
                        
                        // Execute the copy command
                        const successful = document.execCommand('copy');
                        document.body.removeChild(textarea);
                        
                        if (successful) {
                            showSuccess();
                        } else {
                            showError(new Error("execCommand('copy') failed"));
                        }
                    } catch (err) {
                        showError(err);
                    }
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
