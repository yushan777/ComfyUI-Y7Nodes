import { app } from "../../../scripts/app.js";

// Makes the +/- arrow increment on left/top/right/bottom follow the value
// typed into the `step` widget, instead of staying fixed.
app.registerExtension({
    name: "Y7Nodes.ImagePadForOutpaint",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "Y7Nodes_ImagePadForOutpaint") {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);

            const stepWidget = this.widgets?.find((w) => w.name === "step");
            const paddingWidgets = ["left", "top", "right", "bottom"]
                .map((name) => this.widgets?.find((w) => w.name === name))
                .filter(Boolean);

            if (!stepWidget || paddingWidgets.length === 0) {
                return r;
            }

            // ComfyUI's number widgets store the real increment in `step2`
            // and keep `step` at 10x that for internal slider precision.
            const applyStep = (value) => {
                const step = Math.max(1, Math.round(value));
                for (const w of paddingWidgets) {
                    w.options.step = step * 10;
                    w.options.step2 = step;
                }
            };

            applyStep(stepWidget.value);

            const origCallback = stepWidget.callback;
            stepWidget.callback = function (value) {
                const res = origCallback?.apply(this, arguments);
                applyStep(value);
                return res;
            };

            return r;
        };
    },
});
