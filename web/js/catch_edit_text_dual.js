// ComfyUI/custom_nodes/ComfyUI-IMGNR-Utils/js/catch_edit_text.js
// VERSION: 1.0

import { app } from "../../../scripts/app.js";

// Default node size
const DEFAULT_NODE_WIDTH = 350;
const DEFAULT_NODE_HEIGHT = 350;

// Widget names

const TEXT_WIDGET_1_NAME = "editable_text_widget_1";
const TEXT_WIDGET_2_NAME = "editable_text_widget_2";
const ACTION_WIDGET_NAME = "action";
const COPY_BUTTON_1_NAME = "copy_text_1";
const COPY_BUTTON_2_NAME = "copy_text_2";

const LOG_PREFIX = "[Y7Nodes_CatchEditTextNodeDual]";

// --- Style Definitions ---
const baseTextAreaStyles = {
    padding: '4px',
    paddingLeft: '7px',
    border: '1px solid #808080',
    borderRadius: '5px',
    backgroundColor: '#222',
    fontFamily: 'Consolas, Menlo, Monaco, "Courier New", "Lucida Console", monospace',
    fontSize: '11px',
    lineHeight: '1.2',
    resize: 'none',
    overflowY: 'auto',
};

const textAreaStyle1 = {
    ...baseTextAreaStyles,
    color: '#b5cda8',
};

const textAreaStyle2 = {
    ...baseTextAreaStyles,
    color: '#cd9178',
};
// --- End Style Definitions ---

app.registerExtension({
    name: "Y7Nodes.CatchEditTextNodeDual", // Changed name

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "Y7Nodes_CatchEditTextNodeDual") {

            // Store original methods
            const onExecuted = nodeType.prototype.onExecuted;
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            const onConfigure = nodeType.prototype.onConfigure; // Store original onConfigure

            // ===========================================================================================
            // Helper function to update text widgets
            // ===========================================================================================
            function populateDualTextWidgets(textArray) {
                if (!textArray || !Array.isArray(textArray)) {
                    console.warn(`${LOG_PREFIX} populateDualTextWidgets: Input is not a valid array.`);
                    return;
                }

                // Update first widget if text exists
                if (textArray.length > 0) {
                    const newText1 = textArray[0];
                    const targetWidget1 = this.widgets.find(w => w.name === TEXT_WIDGET_1_NAME);
                    if (targetWidget1) {
                        if (targetWidget1.value !== newText1) {
                            targetWidget1.value = newText1;
                            console.log(`${LOG_PREFIX} Updated widget '${targetWidget1.name}' value.`);
                        }
                    } else {
                        console.warn(`${LOG_PREFIX} Could not find widget named '${TEXT_WIDGET_1_NAME}' to update.`);
                    }
                }

                // Update second widget if text exists
                if (textArray.length > 1) {
                    const newText2 = textArray[1];
                    const targetWidget2 = this.widgets.find(w => w.name === TEXT_WIDGET_2_NAME);
                    if (targetWidget2) {
                        if (targetWidget2.value !== newText2) {
                            targetWidget2.value = newText2;
                            console.log(`${LOG_PREFIX} Updated widget '${targetWidget2.name}' value.`);
                        }
                    } else {
                        console.warn(`${LOG_PREFIX} Could not find widget named '${TEXT_WIDGET_2_NAME}' to update.`);
                    }
                }
            }

            // ===========================================================================================
            // When the node is executed
            // ===========================================================================================
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                // Expect message.text to be an array with potentially two strings
                if (message?.text) {
                    populateDualTextWidgets.call(this, message.text);
                }
            };

            // ===========================================================================================
            // When a new instance of this node type is created
            // ===========================================================================================
            nodeType.prototype.onNodeCreated = function () {                
                onNodeCreated?.apply(this, arguments);

                // === Apply Text Widget Styles ===
                const textWidget1 = this.widgets.find(w => w.name === TEXT_WIDGET_1_NAME);
                const textWidget2 = this.widgets.find(w => w.name === TEXT_WIDGET_2_NAME);

                if (textWidget1 && textWidget1.inputEl) {
                    Object.assign(textWidget1.inputEl.style, textAreaStyle1);
                    console.log(`${LOG_PREFIX} Applied style 1 to ${TEXT_WIDGET_1_NAME}`);
                } else {
                    console.warn(`${LOG_PREFIX} Could not find inputEl for ${TEXT_WIDGET_1_NAME} to apply style.`);
                }

                if (textWidget2 && textWidget2.inputEl) {
                    Object.assign(textWidget2.inputEl.style, textAreaStyle2);
                    console.log(`${LOG_PREFIX} Applied style 2 to ${TEXT_WIDGET_2_NAME}`);
                } else {
                    console.warn(`${LOG_PREFIX} Could not find inputEl for ${TEXT_WIDGET_2_NAME} to apply style.`);
                }

                // === Handle Action Widget ===
                const actionWidget = this.widgets.find(w => w.name === ACTION_WIDGET_NAME);
                if (actionWidget) {
                    const originalCallback = actionWidget.callback;
                    actionWidget.callback = (value) => {
                        originalCallback?.call(this, value);
                        console.log(`${LOG_PREFIX} Action changed to: ${value}`);
                        const shouldMuteUpstream = (value === "use_edit_mute_input");
                        // Mute/unmute both input 0 and input 1
                        this.setInputMuted(0, shouldMuteUpstream);
                        this.setInputMuted(1, shouldMuteUpstream);
                    };
                } else {
                    console.warn(`${LOG_PREFIX} Could not find '${ACTION_WIDGET_NAME}' widget to attach callback.`);
                }

                // === Add Copy Buttons ===
                // Widgets already found above for styling, no need to find again
                // const textWidget1 = this.widgets.find(w => w.name === TEXT_WIDGET_1_NAME);
                // const textWidget2 = this.widgets.find(w => w.name === TEXT_WIDGET_2_NAME);

                if (textWidget1) {
                    const copyButton1 = this.addWidget("button", "📋 Copy Text 1", COPY_BUTTON_1_NAME, () => {
                        const textToCopy = textWidget1.value;
                        const button = copyButton1; // Reference the button widget itself
                        const originalText = button.name;

                        const showSuccess = () => {
                            button.name = "✅ Text copied.";
                            setTimeout(() => {
                                button.name = originalText;
                                app.graph.setDirtyCanvas(true, false);
                            }, 1000);
                            app.graph.setDirtyCanvas(true, false);
                        };

                        const showError = (err) => {
                            console.error(`${LOG_PREFIX} Failed to copy text 1:`, err);
                            button.name = "❌ Failed to copy.";
                            setTimeout(() => {
                                button.name = originalText;
                                app.graph.setDirtyCanvas(true, false);
                            }, 2000);
                            app.graph.setDirtyCanvas(true, false);
                        };

                        try {
                            const textarea = document.createElement('textarea');
                            textarea.value = textToCopy;
                            textarea.style.position = 'fixed';
                            textarea.style.left = '-999999px';
                            textarea.style.top = '-999999px';
                            document.body.appendChild(textarea);
                            textarea.focus();
                            textarea.select();
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
                } else {
                    console.warn(`${LOG_PREFIX} Could not find '${TEXT_WIDGET_1_NAME}' to add copy button.`);
                }

                if (textWidget2) {
                    const copyButton2 = this.addWidget("button", "📋 Copy Text 2", COPY_BUTTON_2_NAME, () => {
                        const textToCopy = textWidget2.value;
                        const button = copyButton2; // Reference the button widget itself
                        const originalText = button.name;

                        const showSuccess = () => {
                            button.name = "✅ Text copied.";
                            setTimeout(() => {
                                button.name = originalText;
                                app.graph.setDirtyCanvas(true, false);
                            }, 1000);
                            app.graph.setDirtyCanvas(true, false);
                        };

                        const showError = (err) => {
                            console.error(`${LOG_PREFIX} Failed to copy text 2:`, err);
                            button.name = "❌ Failed to copy.";
                            setTimeout(() => {
                                button.name = originalText;
                                app.graph.setDirtyCanvas(true, false);
                            }, 2000);
                            app.graph.setDirtyCanvas(true, false);
                        };

                        try {
                            const textarea = document.createElement('textarea');
                            textarea.value = textToCopy;
                            textarea.style.position = 'fixed';
                            textarea.style.left = '-999999px';
                            textarea.style.top = '-999999px';
                            document.body.appendChild(textarea);
                            textarea.focus();
                            textarea.select();
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
                } else {
                    console.warn(`${LOG_PREFIX} Could not find '${TEXT_WIDGET_2_NAME}' to add copy button.`);
                }

                // ==========================================================
                // SET INITIAL NODE SIZE
                // ==========================================================
                // Set the initial size of the node when created
                // this.size[0]: width, this.size[1]: height
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
            // When loading a saved workflow
            // ===========================================================================================
            nodeType.prototype.onConfigure = function () {
                onConfigure?.apply(this, arguments);
                // Currently no specific configuration needed here as text is populated via onExecuted
                // Add logic here if needed in the future to restore state from widgets_values
            };

            // ===========================================================================================
            // Method to mute/unmute upstream nodes connected to specific inputs
            // ===========================================================================================
            nodeType.prototype.setInputMuted = function(inputIndex, shouldMute) {
                if (!this.inputs || inputIndex >= this.inputs.length) { console.warn(`${LOG_PREFIX} setInputMuted: Invalid input index ${inputIndex}`); return; }
                const linkId = this.inputs[inputIndex].link;
                if (linkId === null || linkId === undefined) { console.log(`${LOG_PREFIX} setInputMuted: Input ${inputIndex} is not connected.`); return; }
                const linkInfo = this.graph.links[linkId];
                if (!linkInfo) { console.warn(`${LOG_PREFIX} setInputMuted: Could not find link info for link ID ${linkId}`); return; }
                const originNodeId = linkInfo.origin_id;
                const upstreamNode = this.graph.getNodeById(originNodeId);
                if (upstreamNode) {
                    const targetMode = shouldMute ? 2 : 0; // 2 = Muted, 0 = Always
                    if (upstreamNode.mode !== targetMode) {
                        upstreamNode.mode = targetMode;
                        console.log(`${LOG_PREFIX} setInputMuted: Set upstream node ${upstreamNode.id} mode to ${targetMode} (Muted: ${shouldMute})`);
                    } else {
                         console.log(`${LOG_PREFIX} setInputMuted: Upstream node ${upstreamNode.id} mode already ${targetMode}.`);
                    }
                } else {
                    console.warn(`${LOG_PREFIX} setInputMuted: Could not find upstream node with ID ${originNodeId}`);
                }
            };

        } // end if (nodeData.name === "Y7Nodes_CatchEditTextNodeDual")
    }, // end beforeRegisterNodeDef
}); // end registerExtension
