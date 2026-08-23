import { app } from "../../../scripts/app.js";

const DEFAULT_NODE_WIDTH = 300;
const EXTRA_HEIGHT_FOR_TEXT = 20;

app.registerExtension({
    name: "Y7Nodes.ResolutionSelector",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "Y7Nodes_ResolutionSelector") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            const onDrawForeground = nodeType.prototype.onDrawForeground;

            nodeType.prototype.onNodeCreated = function() {
                onNodeCreated?.apply(this, arguments);

                let resized = false;

                if (this.size[0] < DEFAULT_NODE_WIDTH) {
                    this.size[0] = DEFAULT_NODE_WIDTH;
                    resized = true;
                }

                const minHeight = this.computeSize()[1] + EXTRA_HEIGHT_FOR_TEXT;
                if (this.size[1] < minHeight) {
                    this.size[1] = minHeight;
                    resized = true;
                }

                if (resized) {
                    if (this.onResize) {
                        this.onResize(this.size);
                    }
                    app.graph.setDirtyCanvas(true, false);
                }
            };

            nodeType.prototype.onDrawForeground = function(ctx) {
                const r = onDrawForeground?.apply?.(this, arguments);

                const v = app.nodeOutputs?.[this.id + ""];
                if (!this.flags.collapsed && v?.text) {
                    const text = v.text[0] + "";
                    ctx.save();
                    ctx.font = "12px sans-serif";
                    ctx.fillStyle = "dodgerblue";
                    ctx.textAlign = "center";
                    ctx.fillText(text, this.size[0] / 2, this.size[1] - 3);
                    ctx.restore();
                }

                return r;
            };
        }
    },
});
