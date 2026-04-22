import { app } from "../../../scripts/app.js";

const MAX_IMAGES = 8;

function updateImageInputs(node, count) {
    count = Math.max(2, Math.min(MAX_IMAGES, parseInt(count) || 2));

    // Collect indices of image sockets (image1 … imageN)
    const imageIndices = [];
    for (let i = 0; i < node.inputs.length; i++) {
        if (/^image\d+$/.test(node.inputs[i].name)) {
            imageIndices.push(i);
        }
    }
    const currentCount = imageIndices.length;

    if (currentCount < count) {
        for (let i = currentCount + 1; i <= count; i++) {
            node.addInput(`image${i}`, "IMAGE");
        }
    } else if (currentCount > count) {
        // Remove from the highest index downward so earlier indices stay stable
        for (let i = currentCount; i > count; i--) {
            const idx = node.inputs.findIndex(inp => inp.name === `image${i}`);
            if (idx !== -1) node.removeInput(idx);
        }
    }

    app.graph?.setDirtyCanvas(true, false);
}

app.registerExtension({
    name: "Y7Nodes.ImageStitcher",

    nodeCreated(node) {
        if (node.comfyClass !== "Y7Nodes_ImageStitcher") return;

        const countWidget = node.widgets?.find(w => w.name === "image_count");
        if (!countWidget) return;

        // Defer so ComfyUI can finish constructing / configuring the node first.
        // On workflow load, configure() rebuilds node.inputs and restores the
        // saved count widget value before this callback fires.
        setTimeout(() => updateImageInputs(node, countWidget.value), 0);

        // Keep inputs in sync when the user changes the widget
        const origCB = countWidget.callback;
        countWidget.callback = function (value, ...args) {
            origCB?.call(this, value, ...args);
            updateImageInputs(node, value);
        };
    },
});
