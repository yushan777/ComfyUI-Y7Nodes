import { app } from "../../../scripts/app.js";

// Default node size
const DEFAULT_NODE_WIDTH = 300;
const DEFAULT_NODE_HEIGHT = 150;


app.registerExtension({
    name: "Y7Nodes.ImageSizePresets",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "Y7Nodes_ImageSizePresets") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function() {
                onNodeCreated?.apply(this, arguments);

                this.size = [DEFAULT_NODE_WIDTH, DEFAULT_NODE_HEIGHT];

                const presetWidget = this.widgets.find(w => w.name === "preset");
                const dimensionWidget = this.widgets.find(w => w.name === "dimension");

                if (!presetWidget || !dimensionWidget) return;

                const updateDimensions = async (presetValue) => {
                    try {
                        const response = await fetch(`/y7nodes/image_size_dims?preset=${encodeURIComponent(presetValue)}`);
                        if (!response.ok) return;
                        const data = await response.json();

                        dimensionWidget.options.values = data.labels;

                        // Reset to first option only if the current value isn't in the new list
                        if (!data.labels.includes(dimensionWidget.value)) {
                            dimensionWidget.value = data.labels[0];
                        }

                        app.graph.setDirtyCanvas(true);
                    } catch (e) {
                        console.error("Y7Nodes ImageSizePresets: failed to fetch dims", e);
                    }
                };

                const origCallback = presetWidget.callback;
                presetWidget.callback = function(value) {
                    origCallback?.call(this, value);
                    updateDimensions(value);
                };
            };
        }
    },
});
