import { app } from "../../../scripts/app.js";


const DEFAULT_NODE_WIDTH = 400;
const DEFAULT_NODE_HEIGHT = 500;


app.registerExtension({
    name: "Y7Nodes.QwenVL",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {

        if (nodeData.name === "Y7Nodes_QwenVL") {

            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function() {
                onNodeCreated?.apply(this, arguments);

                if (this.size[0] < DEFAULT_NODE_WIDTH || this.size[1] < DEFAULT_NODE_HEIGHT) {
                    this.size[0] = DEFAULT_NODE_WIDTH;
                    this.size[1] = DEFAULT_NODE_HEIGHT;

                    if (this.onResize) {
                        this.onResize(this.size);
                    }

                    app.graph.setDirtyCanvas(true, false);
                }
            };

            nodeType.prototype.onExecuted = function (message) {

            };

            nodeType.prototype.onConfigure = function () {

            };

        }
    },
});
