import { app } from "../../../scripts/app.js";


const TEXT_NODE_WIDTH = 400;
const TEXT_NODE_HEIGHT = 500;

const VISION_NODE_WIDTH = 400;
const VISION_NODE_HEIGHT = 480;

const SELECT_NODE_WIDTH = 300;
const SELECT_NODE_HEIGHT = 80;

function forceSize(node, width, height) {
    // Deferred so it runs after ComfyUI's auto-size pass
    setTimeout(() => {
        node.size[0] = width;
        node.size[1] = height;
        node.onResize?.(node.size);
        app.graph.setDirtyCanvas(true, false);
    }, 0);
}

app.registerExtension({
    name: "Y7Nodes.LMStudio",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {

        if (nodeData.name === "Y7Nodes_LMStudioText") {

            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function() {
                onNodeCreated?.apply(this, arguments);
                forceSize(this, TEXT_NODE_WIDTH, TEXT_NODE_HEIGHT);
            };

            nodeType.prototype.onExecuted = function (message) {

            };

            nodeType.prototype.onConfigure = function () {

            };

        }

        if (nodeData.name === "Y7Nodes_LMStudioVision") {

            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function() {
                onNodeCreated?.apply(this, arguments);
                forceSize(this, VISION_NODE_WIDTH, VISION_NODE_HEIGHT);
            };

            nodeType.prototype.onExecuted = function (message) {

            };

            nodeType.prototype.onConfigure = function () {

            };

        }

        if (nodeData.name === "Y7Nodes_SelectLMSModel") {

            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function() {
                onNodeCreated?.apply(this, arguments);
                forceSize(this, SELECT_NODE_WIDTH, SELECT_NODE_HEIGHT);
            };

            nodeType.prototype.onExecuted = function (message) {

            };

            nodeType.prototype.onConfigure = function () {

            };

        }
    },
});
