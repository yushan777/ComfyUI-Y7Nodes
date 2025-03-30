import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

// Font family options
const monospaceFonts = 'Consolas, Menlo, Monaco, "Courier New", "Lucida Console", monospace';
const sansSerifFonts = 'Arial, Helvetica, sans-serif';

// Font size options
const fontSizes = ["8", "9", "10", "11", "12", "14", "16", "18", "20"];

// Widget indexes
const TEXT_WIDGET = 0;
const MONOSPACE_WIDGET = 1;
const FONTSIZE_WIDGET = 2;
const COPYBUTTON_WIDGET = 3;

// Single global style for the text area
const textAreaStyles = {
    readOnly: true,
    opacity: 0.9,
    padding: '4px',
    paddingLeft: '7px',
    border: '1px solid #808080',
    borderRadius: '5px',
    backgroundColor: '#222',
    color: 'rgb(168,153,249)',  // Single color for all text
    fontFamily: monospaceFonts, // Default to monospace
    fontSize: '11px',
    lineHeight: '1.2',
    resize: 'none',
    overflowY: 'auto',
};

app.registerExtension({
    // name: a unique identifier used by CUI to register and track this extension
    // It can be any string but it's good practice to namespace it with your prefix (Y7)    
    name: "Y7Nodes.ShowAnything",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {

        // check if the node being registered is our target node
        // this must match the class mapping name in __init__.py
        if (nodeData.name === "Y7Nodes_ShowAnything") {

            // store the original onNodeCreated method
            const onNodeCreated = nodeType.prototype.onNodeCreated;
        
            // When a new instance of this node type is created....
            nodeType.prototype.onNodeCreated = function() {
                
                // override it with our own implementation, but still 
                // preserve the ability to call the original method with
                const result = onNodeCreated?.apply(this, arguments);
                
                // Create widgets array (if not exist)
                if (!this.widgets) {
                    // console.log("===> Creating widgets array")
                    this.widgets = [];
                }
                
                // ==========================================================
                //  1. TEXT WIDGET
                // ==========================================================
                // If number of widgets in the array is empty (it should be)
                if (!this.widgets.length) {
                    // create text widget - this will be the first widget on the node. 

                    // const w = ComfyWidgets["STRING"](this, "text", ["STRING", { multiline: true }], app).widget;
                    // above line is done using named variables for clarity

                    // The widget type
                    const widgetType = "STRING";
                    // The name/key of the widget
                    const widgetName = "text"; 
                    // array containing widget type, multiline option
                    const widgetConfig = [widgetType, { multiline: true }];      
                    // create the widget, last arg app is the ComfyUI application instance
                    const widgetResult = ComfyWidgets["STRING"](this, widgetName, widgetConfig, app);
                    // we just need the widget instance
                    const w = widgetResult.widget;                         
                    // apply styles and make readonly               
                    Object.assign(w.inputEl.style, textAreaStyles);
                    w.inputEl.readOnly = true;
                }

                // ==========================================================
                // 2. MONOSPACE FONT TOGGLE
                // ==========================================================
                // Add boolean widget for monospace font toggle
                // Default to true (monospace font)
                const monospaceFontWidget = this.addWidget("toggle", "monospace_font", true, (value) => {
                    // Update font family based on toggle value
                    if (this.widgets[TEXT_WIDGET] && this.widgets[TEXT_WIDGET].inputEl) {
                        this.widgets[TEXT_WIDGET].inputEl.style.fontFamily = value ? monospaceFonts : sansSerifFonts;
                    }
                }, { serialize: true });

                // ==========================================================
                // 3. FONT SIZE
                // ==========================================================
                // Create a select element for font size
                const fontSizeWidget = this.addWidget("combo", "font_size", "11", (value) => {
                    // Update font size when selection changes
                    if (this.widgets[TEXT_WIDGET] && this.widgets[TEXT_WIDGET].inputEl) {
                        this.widgets[TEXT_WIDGET].inputEl.style.fontSize = value + "px";
                    }
                }, { 
                    values: fontSizes,
                    serialize: true
                });
                
                // Apply default font size on creation
                if (this.widgets[TEXT_WIDGET] && this.widgets[TEXT_WIDGET].inputEl) {
                    this.widgets[TEXT_WIDGET].inputEl.style.fontSize = "11px";
                }
                
                // ==========================================================
                // 4. BUTTON
                // ==========================================================
                // Add button - Copy button
                // serialize: false tells ComfyUI not to save the button's state when 
                // saving the workflow since buttons here are transient
                // Add button - Copy button
                this.addWidget("button", "📋 Copy Text", null, () => {
                    // Get the text from the text widget
                    const textToCopy = this.widgets[TEXT_WIDGET].value;
                    
                    console.log("textToCopy =====> \n " + textToCopy);

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
                populate.call(this, message.text);
            };

            // ===========================================================================================
            // When loading a saved workflow
            nodeType.prototype.onConfigure = function () {
                if (this.widgets_values?.length) {
                    populate.call(this, this.widgets_values[TEXT_WIDGET]);
                    
                    // Apply saved monospace font setting if available
                    // The monospace font toggle is the second widget (index 1)
                    // But in widgets_values, it's the second value (index 1) because the button is not serialized
                    if (this.widgets_values.length > 1 && this.widgets[MONOSPACE_WIDGET]) {
                        const useMonospace = this.widgets_values[MONOSPACE_WIDGET];
                        this.widgets[MONOSPACE_WIDGET].value = useMonospace;
                        
                        // Update font family based on saved value
                        if (this.widgets[TEXT_WIDGET] && this.widgets[TEXT_WIDGET].inputEl) {
                            this.widgets[TEXT_WIDGET].inputEl.style.fontFamily = useMonospace ? monospaceFonts : sansSerifFonts;
                        }
                    }
                    
                    // Apply saved font size setting if available
                    // The font size widget is the third widget (index 2)
                    // In widgets_values, it's the third value (index 2)
                    if (this.widgets_values.length > 2 && this.widgets[FONTSIZE_WIDGET]) {
                        const fontSize = this.widgets_values[FONTSIZE_WIDGET];
                        this.widgets[FONTSIZE_WIDGET].value = fontSize;
                        
                        // Update font size based on saved value
                        if (this.widgets[TEXT_WIDGET] && this.widgets[TEXT_WIDGET].inputEl) {
                            this.widgets[TEXT_WIDGET].inputEl.style.fontSize = fontSize + "px";
                        }
                    }
                }
            };

            // ================================================================================================
            function populate(text) {
                // Convert to string if it's not already
                const displayText = text.toString();
                                
                // Update the text
                this.widgets[TEXT_WIDGET].value = displayText;

            
                // ===========================================================================================
                // Resize node (if needed)
                requestAnimationFrame(() => {

                    console.log("=====>> requestAnimationFrame() called!!!")
                    // this function doesn't seem to do anything

                    // 1. Get required node dimensions based on content
                    //sz[0] = width, sz[1] = height
                    //let sz = this.computeSize();

                    // 2. Maintain current size : (ensure the node never shrinks smaller than its current size)
                    // allow it to grow but never shrink (can get annoying)
                    // If the computed width (sz[0]) <  the current width (this.size[0]), keep the current width
                    // If the computed height (sz[1]) <  the current height (this.size[1]), keep the current width
                    //if (sz[0] < this.size[0]) sz[0] = this.size[0];  // width
                    //if (sz[1] < this.size[1]) sz[1] = this.size[1];  // height

                    // 3. Add padding to the height (e.g., 10 pixels)
                    // sz[1] -= 50;  // Add your desired padding amount here

                    // 4. Apply the resize
                    //this.onResize?.(sz);

                    // 5. Tell ComfyUI to redraw
                    //app.graph.setDirtyCanvas(true, false);
                });
            }

        }
    },
});
