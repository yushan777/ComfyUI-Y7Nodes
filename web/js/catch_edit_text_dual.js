// ComfyUI/custom_nodes/ComfyUI-IMGNR-Utils/js/catch_edit_text.js
// VERSION: 1.0

import { app } from "../../../scripts/app.js";

// Default node size
const DEFAULT_NODE_WIDTH = 350;
const DEFAULT_NODE_HEIGHT = 350;

app.registerExtension({
    name: "Comfy.Y7Nodes_CatchEditTextNodeDual.JS",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "Y7Nodes_CatchEditTextNodeDual") {

            const onExecuted = nodeType.prototype.onExecuted;
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                 // Expect message.text to be an array with potentially two strings
                 if (message?.text && Array.isArray(message.text)) {
                    // Update first widget if text exists
                    if (message.text.length > 0) {
                        const newText1 = message.text[0];
                        const targetWidget1 = this.widgets.find(w => w.name === "editable_text_widget_1");
                        if (targetWidget1) {
                            if (targetWidget1.value !== newText1) {
                                targetWidget1.value = newText1;
                                console.log(`[Y7Nodes_CatchEditTextNodeDual.JS] Updated widget '${targetWidget1.name}' value via onExecuted.`);
                            }
                        } else {
                             console.warn("[Y7Nodes_CatchEditTextNodeDual.JS] Could not find widget named 'editable_text_widget_1' to update.");
                        }
                    }
                    // Update second widget if text exists
                    if (message.text.length > 1) {
                        const newText2 = message.text[1];
                        const targetWidget2 = this.widgets.find(w => w.name === "editable_text_widget_2");
                         if (targetWidget2) {
                            if (targetWidget2.value !== newText2) {
                                targetWidget2.value = newText2;
                                console.log(`[Y7Nodes_CatchEditTextNodeDual.JS] Updated widget '${targetWidget2.name}' value via onExecuted.`);
                            }
                        } else {
                             console.warn("[Y7Nodes_CatchEditTextNodeDual.JS] Could not find widget named 'editable_text_widget_2' to update.");
                        }
                    }
                 }
             };

             nodeType.prototype.onNodeCreated = function () {
                 onNodeCreated?.apply(this, arguments);

                 // Set initial node size if it's the default computed size
                 // This aims to only resize newly created nodes, not ones loaded from a workflow
                 const computedSize = this.computeSize();
                 // Check if size exists and if it matches the initial computed size
                 if (!this.size || (computedSize && this.size[0] === computedSize[0] && this.size[1] === computedSize[1])) {
                    this.size = [DEFAULT_NODE_WIDTH, DEFAULT_NODE_HEIGHT];
                    // Apply the resize explicitly if needed
                    if (this.onResize) {
                        this.onResize(this.size);
                    }
                    app.graph.setDirtyCanvas(true, false); // Redraw needed after size change
                    console.log(`[Y7Nodes_CatchEditTextNodeDual.JS] Set initial size for node ${this.id} to ${this.size}`);
                 }


                 // --- Existing action widget logic ---
                 const actionWidget = this.widgets.find(w => w.name === "action");

                if (actionWidget) {
                    const originalCallback = actionWidget.callback;
                    actionWidget.callback = (value) => {
                        originalCallback?.call(this, value);
                        console.log(`[Y7Nodes_CatchEditTextNodeDual.JS] Action changed to: ${value}`);
                        const shouldMuteUpstream = (value === "use_edit_mute_input");
                        // Mute/unmute both input 0 and input 1
                        this.setInputMuted(0, shouldMuteUpstream);
                        this.setInputMuted(1, shouldMuteUpstream);
                    };
                } else {
                     console.warn("[Y7Nodes_CatchEditTextNodeDual.JS] Could not find 'action' widget to attach callback.");
                 }
                 // --- End of existing action widget logic ---

                 // --- Add Copy Buttons ---
                 const textWidget1 = this.widgets.find(w => w.name === "editable_text_widget_1");
                 const textWidget2 = this.widgets.find(w => w.name === "editable_text_widget_2");

                 if (textWidget1) {
                     const copyButton1 = this.addWidget("button", "📋 Copy Text 1", "copy_text_1", () => {
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
                             console.error('[Y7Nodes_CatchEditTextNodeDual.JS] Failed to copy text 1:', err);
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
                     }, { serialize: false }); // Added serialize: false like in clip_token_count
                 } else {
                     console.warn("[Y7Nodes_CatchEditTextNodeDual.JS] Could not find 'editable_text_widget_1' to add copy button.");
                 }

                 if (textWidget2) {
                     const copyButton2 = this.addWidget("button", "📋 Copy Text 2", "copy_text_2", () => {
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
                             console.error('[Y7Nodes_CatchEditTextNodeDual.JS] Failed to copy text 2:', err);
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
                     }, { serialize: false }); // Added serialize: false like in clip_token_count
                 } else {
                     console.warn("[Y7Nodes_CatchEditTextNodeDual.JS] Could not find 'editable_text_widget_2' to add copy button.");
                 }
                 // --- End of Add Copy Buttons ---
             };

             nodeType.prototype.setInputMuted = function(inputIndex, shouldMute) {
                if (!this.inputs || inputIndex >= this.inputs.length) { console.warn(`[Y7Nodes_CatchEditTextNodeDual.JS] setInputMuted: Invalid input index ${inputIndex}`); return; }
                const linkId = this.inputs[inputIndex].link;
                if (linkId === null || linkId === undefined) { console.log(`[Y7Nodes_CatchEditTextNodeDual.JS] setInputMuted: Input ${inputIndex} is not connected.`); return; }
                const linkInfo = this.graph.links[linkId];
                if (!linkInfo) { console.warn(`[Y7Nodes_CatchEditTextNodeDual.JS] setInputMuted: Could not find link info for link ID ${linkId}`); return; }
                const originNodeId = linkInfo.origin_id;
                const upstreamNode = this.graph.getNodeById(originNodeId);
                if (upstreamNode) {
                    const targetMode = shouldMute ? 2 : 0;
                    if (upstreamNode.mode !== targetMode) {
                        upstreamNode.mode = targetMode;
                        console.log(`[Y7Nodes_CatchEditTextNodeDual.JS] setInputMuted: Set upstream node ${upstreamNode.id} mode to ${targetMode} (Muted: ${shouldMute})`);
                    } else {
                         console.log(`[Y7Nodes_CatchEditTextNodeDual.JS] setInputMuted: Upstream node ${upstreamNode.id} mode already ${targetMode}.`);
                    }
                } else {
                    console.warn(`[Y7Nodes_CatchEditTextNodeDual.JS] setInputMuted: Could not find upstream node with ID ${originNodeId}`);
                }
             };

        } // end if (nodeData.name === "Y7Nodes_CatchEditTextNodeDual")
    }, // end beforeRegisterNodeDef
}); // end registerExtension
