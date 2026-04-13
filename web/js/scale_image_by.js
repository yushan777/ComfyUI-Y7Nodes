import { app } from "../../../scripts/app.js";

const DEFAULT_NODE_WIDTH = 300;
const DEFAULT_NODE_HEIGHT = 150;

app.registerExtension({
    name: "Y7Nodes.ScaleImageBy",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "Y7Nodes_ScaleImageBy") {
            const onDrawForeground = nodeType.prototype.onDrawForeground;
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onDrawForeground = function(ctx) {
                const r = onDrawForeground?.apply?.(this, arguments);

                const v = app.nodeOutputs?.[this.id + ""];
                if (!this.flags.collapsed && v?.text) {
                    const text = v.text[0] + "";
                    ctx.save();
                    ctx.font = "bold 12px sans-serif";
                    ctx.fillStyle = "dodgerblue";
                    ctx.fillText(text, 20, LiteGraph.NODE_SLOT_HEIGHT * 3);
                    ctx.restore();
                }

                return r;
            };

            nodeType.prototype.onNodeCreated = function() {
                onNodeCreated?.apply(this, arguments);

                let resized = false;

                if (this.size[0] < DEFAULT_NODE_WIDTH) {
                    this.size[0] = DEFAULT_NODE_WIDTH;
                    resized = true;
                }

                if (this.size[1] < DEFAULT_NODE_HEIGHT) {
                    this.size[1] = DEFAULT_NODE_HEIGHT;
                    resized = true;
                }

                if (resized) {
                    if (this.onResize) {
                        this.onResize(this.size);
                    }
                    app.graph.setDirtyCanvas(true, false);
                }
            };
        }
    },
});
