import { app } from "../../../scripts/app.js";

const DEFAULT_NODE_WIDTH = 340;
const DEFAULT_NODE_HEIGHT = 280;

app.registerExtension({
    name: "Y7Nodes.Flux2KleinEdit1",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "Y7Nodes_Flux2KleinEdit_Ref1") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;

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
