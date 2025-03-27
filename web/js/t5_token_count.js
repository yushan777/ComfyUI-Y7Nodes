import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

// Font family options
const monospaceFonts = 'Consolas, Menlo, Monaco, "Courier New", "Lucida Console", monospace';
// const sansSerifFonts = 'Arial, Helvetica, sans-serif';

// Font size options
const fontSizes = ["8", "9", "10", "11", "12", "14", "16", "18", "20"];

const tokensPerLine = ["2", "3", "4", "5", "6", "7", "8", "9", "10"];

// Default node size
const DEFAULT_NODE_WIDTH = 500;
const DEFAULT_NODE_HEIGHT = 300;

// Widget indexes
// if a widget of basic data type (like "STRING") is used in the backend with forceInput=True or a COMFYUI widget of type like "MODEL" 
// then its widget index is counted. Only time it does not is when a widget with custom data type is used
// in the backend with forceInput=True then it is not counted in the widget array (don't know why)
const TEXT_WIDGET = 1;
const TOKEN_LIMIT_WIDGET = 2;
const SHOW_TOKENS_WIDGET = 3;
const TOKENS_PER_LINE_WIDGET = 4;
const FONTSIZE_WIDGET = 5;
const COPYBUTTON_WIDGET = 6;

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
    fontSize: '10px',
    lineHeight: '1.2',
    resize: 'none',
    overflowY: 'auto',
};

app.registerExtension({
    // name: a unique identifier used by CUI to register and track this extension
    // It can be any string but it's good practice to namespace it with your prefix (Y7)    
    name: "Y7Nodes.T5_TokenCounter",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {

        // check if the node being registered is our target node
        // this must match the class mapping name in __init__.py
        if (nodeData.name === "Y7Nodes_T5_TokenCounter") {

            // store the original onNodeCreated method
            const onNodeCreated = nodeType.prototype.onNodeCreated;
        
            // When a new instance of this node type is created....
            nodeType.prototype.onNodeCreated = function() {
                
                
                // override it with our own implementation, but still 
                // preserve the ability to call the original method with
                const result = onNodeCreated?.apply(this, arguments);
                
                // Create widgets array (if not exist)
                if (!this.widgets) {
                    console.log("===> Creating widgets array")
                    this.widgets = [];
                }
                
                // NOTE THAT WIDGET INDEXES ARE DIFFERENT FROM SHOW_ANYTHING NODE

                // Debug existing widgets
                // check the length of the widgets array (if any, they will be backend widgets added in python)
                console.log("===> widgets array length = " + this.widgets.length);
                // if array length > 0 
                if (this.widgets.length > 0) {
                    // list the widget names
                    for (let i = 0; i < this.widgets.length; i++) {
                        console.log("===> Existing widget:", this.widgets[i]);
                    }
                }

                
                // ==========================================================
                // ADD TEXT WIDGET
                // ==========================================================
                console.log("creating text widget")
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

                // ==========================================================
                // TOKEN LIMIT WIDGET
                // ==========================================================
                // Create select element for token-limit
                const tokenLimitWidget = this.addWidget("combo", "token_limit", "512", (value) => {
                    // no frontend functionality but we create the widget her to it neatly is placed under the text widget
                    // access to this widget's value can be done in backend side.                     
                }, { 
                    values: ["256", "512"],
                    serialize: true
                });                
                // Store the widget index in properties so Python can access it
                this.properties = this.properties || {};
                this.properties.token_limit_index = TOKEN_LIMIT_WIDGET;                

                // ==========================================================
                // SHOW TOKENS WIDGET (BOOLEAN)
                // ==========================================================
                // Add boolean widget for SHOW TOKENStoggle
                // Default to true 
                const showTokensWidget = this.addWidget("toggle", "show_tokens", true, (value) => {
                    // no frontend functionality but we create the widget her to it neatly is placed under the text widget
                    // access to this widget's value can be done in backend side. 
                }, {serialize: true});                     
                // Store the widget index in properties so Python can access it
                this.properties = this.properties || {};
                this.properties.show_tokens_index = SHOW_TOKENS_WIDGET;

                // ==========================================================
                // TOKENS PER LINE (IF SHOW TOKENS IS TRUE)
                // ==========================================================
                // Create select element for number of tokens to show per line
                const tokensPerLineWidget = this.addWidget("combo", "tokens_per_line", "4", (value) => {
                    // no frontend functionality but we create the widget her to it neatly is placed under the text widget
                    // access to this widget's value can be done in backend side.                     
                }, { 
                    values: tokensPerLine,
                    serialize: true
                });                
                // Store the widget index in properties so Python can access it
                this.properties = this.properties || {};
                this.properties.tokens_per_line_index = TOKENS_PER_LINE_WIDGET;                
                
                // ==========================================================
                //  FONT SIZE
                // ==========================================================
                // Create a select element for font size
                const fontSizeWidget = this.addWidget("combo", "font_size", "10", (value) => {
                    // Update font size when selection changes
                    // Update text widget using constant
                    if (this.widgets[TEXT_WIDGET] && this.widgets[TEXT_WIDGET].inputEl) {
                        this.widgets[TEXT_WIDGET].inputEl.style.fontSize = value + "px";
                    }
                }, { 
                    values: fontSizes,
                    serialize: true
                });
                
                // Apply default font size on creation
                // Apply default font size using constant
                if (this.widgets[TEXT_WIDGET] && this.widgets[TEXT_WIDGET].inputEl) {
                    this.widgets[TEXT_WIDGET].inputEl.style.fontSize = "11px";
                }
                
                // ==========================================================
                //  BUTTON
                // ==========================================================
                // Add button - Copy button
                // serialize: false tells ComfyUI not to save the button's state when 
                // saving the workflow since buttons here are transient
                this.addWidget("button", "📋 Copy Text", null, () => {
                    
                    // Get the text from the text widget using constant
                    const textToCopy = this.widgets[TEXT_WIDGET].value;
                    
                    // Copy to clipboard 
                    navigator.clipboard.writeText(textToCopy)
                        .then(() => {
                            // Temporarily change button text to show success
                            const button = this.widgets[COPYBUTTON_WIDGET];
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

                // ==========================================================
                // SET INITIAL NODE SIZE
                // ==========================================================
                // Set the initial size of the node
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


                return result;
            };

            // ===========================================================================================
            // When the node is executed
            nodeType.prototype.onExecuted = function (message) {
                // Store the original full text when it comes from the backend
                const nodeId = this.id;
                populate.call(this, message.text, true);
            };

            // ===========================================================================================
            // When loading a saved workflow
            nodeType.prototype.onConfigure = function () {
                if (this.widgets_values?.length) {
                    populate.call(this, this.widgets_values[TEXT_WIDGET]);
                                        
                    // Apply saved font size setting if available
                    if (this.widgets_values.length > FONTSIZE_WIDGET && this.widgets[FONTSIZE_WIDGET]) {
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
            function populate(text, isNewText) {
                const nodeId = this.id;
                let displayText;
                
                // Check if text is undefined or null
                if (text === undefined || text === null) {
                    displayText = "";
                }
                // Handle char arrays by joining elements
                else if (Array.isArray(text)) {
                    displayText = text.join('');
                } else {
                    displayText = text.toString();
                }
                                

                this.widgets[TEXT_WIDGET].value = displayText;

            
                // ===========================================================================================
                // Resize node (if needed)
                requestAnimationFrame(() => {

                    // console.log("=====>> requestAnimationFrame() called!!!")
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
