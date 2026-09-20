import { app } from "../../../scripts/app.js";

// Steps the `value` widget when a prompt is queued, the same way the native
// control_after_generate widget steps a seed - except the amount comes from the node's own
// `step` widget instead of the value widget's built-in step.

// ComfyUI setting: "after" (the default) steps once the prompt has been queued, "before" steps
// on the way out, so the value on the node is the one that was just sent.
function runBefore() {
    try {
        return app.ui?.settings?.getSettingValue?.("Comfy.WidgetControlMode", "after") === "before";
    } catch (e) {
        return false;
    }
}

// 0.1 + 0.2 = 0.30000000000000004, and that drift compounds over a run, so trim it back off.
// Left alone for huge values, where toFixed() has nothing useful to say.
function trimFloatError(v) {
    return Math.abs(v) < 1e12 ? parseFloat(v.toFixed(10)) : v;
}

app.registerExtension({
    name: "Y7Nodes.Float",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "Y7Nodes_Float") {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);

            const valueWidget = this.widgets?.find((w) => w.name === "value");
            const controlWidget = this.widgets?.find((w) => w.name === "control_after_generate");
            const stepWidget = this.widgets?.find((w) => w.name === "step");
            if (!valueWidget || !controlWidget || !stepWidget) {
                return r;
            }

            // In "before" mode the first queue sends the value as it stands; only later queues
            // step it. This matches the native control widget.
            let queuedOnce = false;

            const applyControl = () => {
                const mode = controlWidget.value;
                if (mode !== "increment" && mode !== "decrement") {
                    return;
                }

                // value is converted to an input and driven by a link, so it isn't ours to change
                if (this.inputs?.some((i) => i.widget?.name === "value" && i.link != null)) {
                    return;
                }

                const step = Number(stepWidget.value);
                if (!step) {
                    return;
                }

                const options = valueWidget.options ?? {};
                const min = typeof options.min === "number" ? options.min : -Infinity;
                const max = typeof options.max === "number" ? options.max : Infinity;

                let next = Number(valueWidget.value) + (mode === "increment" ? step : -step);
                next = trimFloatError(Math.min(Math.max(next, min), max));

                valueWidget.value = next;
                valueWidget.callback?.(next);
                app.graph?.setDirtyCanvas(true, false);
            };

            valueWidget.beforeQueued = ({ isPartialExecution } = {}) => {
                if (isPartialExecution || !runBefore()) {
                    return;
                }
                if (queuedOnce) {
                    applyControl();
                }
                queuedOnce = true;
            };

            valueWidget.afterQueued = ({ isPartialExecution } = {}) => {
                if (isPartialExecution || runBefore()) {
                    return;
                }
                applyControl();
            };

            return r;
        };
    },
});
