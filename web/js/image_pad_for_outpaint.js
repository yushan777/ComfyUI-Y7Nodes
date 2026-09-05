import { app } from "../../../scripts/app.js";

const NODE_NAME = "Y7Nodes_ImagePadForOutpaint";
const DEFAULT_NODE_WIDTH = 300;
const EXTRA_HEIGHT_FOR_TEXT = 22;

const SIDES = ["left", "top", "right", "bottom"];

function snapToStep(value, step) {
    if (step <= 1) return value;
    const quotient = Math.floor(value / step);
    const remainder = value - quotient * step;
    return (remainder * 2 >= step ? quotient + 1 : quotient) * step;
}

// Mirrors resolve_padding() in the Python node.
function resolvePadding(pad, step) {
    return {
        left: snapToStep(pad.left, step),
        top: snapToStep(pad.top, step),
        right: snapToStep(pad.right, step),
        bottom: snapToStep(pad.bottom, step),
    };
}

app.registerExtension({
    name: "Y7Nodes.ImagePadForOutpaint",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== NODE_NAME) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        const onDrawForeground = nodeType.prototype.onDrawForeground;

        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);

            const byName = (name) => this.widgets?.find((w) => w.name === name);
            const stepWidget = byName("step");
            const sideWidgets = SIDES.map(byName).filter(Boolean);

            if (!stepWidget || sideWidgets.length !== 4) {
                return r;
            }

            const redraw = () => {
                this.setDirtyCanvas(true, true);
            };

            // ComfyUI's number widgets keep the real increment in `step2` and hold
            // `step` at 10x that for internal slider precision, so both are set.
            const applyStep = (value) => {
                const s = Math.max(1, Math.round(value));
                for (const w of [...sideWidgets, quickAll, quickX, quickY]) {
                    if (!w) continue;
                    w.options.step = s * 10;
                    w.options.step2 = s;
                }
            };

            const setSides = (values) => {
                for (const [name, value] of Object.entries(values)) {
                    const w = byName(name);
                    if (w) w.value = value;
                }
                syncQuick();
                redraw();
            };

            // The quick-set fields track the sides whenever the sides agree with each
            // other; when the sides are uneven there is no single number to show, so
            // the field is left holding whatever was last applied through it.
            let syncing = false;
            const syncQuick = () => {
                if (syncing) return;
                syncing = true;
                const [l, t, rt, b] = SIDES.map((n) => byName(n)?.value ?? 0);
                if (l === rt) quickX.value = l;
                if (t === b) quickY.value = t;
                if (l === rt && t === b && l === t) quickAll.value = l;
                syncing = false;
            };

            // The frontend reads `serialize` off the widget itself, not off its options,
            // so these controls have to be marked directly or their values end up in
            // widgets_values and shift every saved value along by one.
            const noSave = (w) => { w.serialize = false; return w; };
            const numberOpts = { min: 0, max: 16384, step: 160, step2: 16, precision: 0 };

            const quickAll = noSave(this.addWidget("number", "set all sides", 0, (v) => {
                const n = Math.max(0, Math.round(v));
                quickAll.value = n;
                setSides({ left: n, top: n, right: n, bottom: n });
            }, { ...numberOpts }));

            const quickX = noSave(this.addWidget("number", "set left + right", 0, (v) => {
                const n = Math.max(0, Math.round(v));
                quickX.value = n;
                setSides({ left: n, right: n });
            }, { ...numberOpts }));

            const quickY = noSave(this.addWidget("number", "set top + bottom", 0, (v) => {
                const n = Math.max(0, Math.round(v));
                quickY.value = n;
                setSides({ top: n, bottom: n });
            }, { ...numberOpts }));

            noSave(this.addWidget("button", "Reset sides", null, () => {
                quickAll.value = 0;
                quickX.value = 0;
                quickY.value = 0;
                setSides({ left: 0, top: 0, right: 0, bottom: 0 });
            }));

            // Typing into a side directly should pull the quick-set fields back in line.
            for (const w of sideWidgets) {
                const orig = w.callback;
                w.callback = function () {
                    const res = orig?.apply(this, arguments);
                    syncQuick();
                    return res;
                };
            }

            applyStep(stepWidget.value);
            const origStepCallback = stepWidget.callback;
            stepWidget.callback = function (value) {
                const res = origStepCallback?.apply(this, arguments);
                applyStep(value);
                return res;
            };

            // The readout is drawn on the node body below the last widget, so the
            // space for it has to be part of computeSize - anything that re-fits the
            // node (addWidget's expandToFitContent, a double-click resize, loading the
            // workflow) would otherwise shrink it back over the Reset button.
            const origComputeSize = this.computeSize;
            this.computeSize = function () {
                const size = origComputeSize.apply(this, arguments);
                size[1] += EXTRA_HEIGHT_FOR_TEXT;
                return size;
            };

            let resized = false;
            if (this.size[0] < DEFAULT_NODE_WIDTH) {
                this.size[0] = DEFAULT_NODE_WIDTH;
                resized = true;
            }
            const minHeight = this.computeSize()[1];
            if (this.size[1] < minHeight) {
                this.size[1] = minHeight;
                resized = true;
            }
            if (resized) {
                this.onResize?.(this.size);
                app.graph.setDirtyCanvas(true, false);
            }

            return r;
        };

        // The size of the incoming image. An upstream preview is checked first because
        // it updates the moment you pick a different image; the size the backend last
        // reported is the fallback for upstream nodes that show no preview.
        nodeType.prototype.y7GetSourceSize = function () {
            try {
                const slot = this.inputs?.findIndex((i) => i.name === "image") ?? -1;
                if (slot >= 0 && this.graph) {
                    const origin = this.getInputNode?.(slot);
                    const img = origin?.imgs?.[origin.imageIndex ?? 0];
                    if (img?.naturalWidth) {
                        return { w: img.naturalWidth, h: img.naturalHeight };
                    }
                }
            } catch (e) {
                // getInputNode can throw on a detached node; fall through to the backend value
            }

            const out = app.nodeOutputs?.[this.id + ""];
            if (out?.src?.length === 2) {
                return { w: out.src[0], h: out.src[1] };
            }

            return null;
        };

        nodeType.prototype.onDrawForeground = function (ctx) {
            const r = onDrawForeground?.apply?.(this, arguments);
            if (this.flags.collapsed) {
                return r;
            }

            const byName = (name) => this.widgets?.find((w) => w.name === name);
            const src = this.y7GetSourceSize();

            let line1;
            let line2 = "";

            if (!src) {
                line1 = "connect an image";
            } else {
                const pad = {
                    left: byName("left")?.value ?? 0,
                    top: byName("top")?.value ?? 0,
                    right: byName("right")?.value ?? 0,
                    bottom: byName("bottom")?.value ?? 0,
                };
                const step = Math.max(1, Math.round(byName("step")?.value ?? 1));
                const p = resolvePadding(pad, step);
                const newW = src.w + p.left + p.right;
                const newH = src.h + p.top + p.bottom;

                line1 = `${src.w} x ${src.h}  →  ${newW} x ${newH}`;
                line2 = `L ${p.left}   T ${p.top}   R ${p.right}   B ${p.bottom}`;
            }

            ctx.save();
            ctx.textAlign = "center";
            ctx.font = "bold 12px sans-serif";
            ctx.fillStyle = "dodgerblue";
            ctx.fillText(line1, this.size[0] / 2, this.size[1] - 16);
            if (line2) {
                ctx.font = "10px sans-serif";
                ctx.fillStyle = "#999";
                ctx.fillText(line2, this.size[0] / 2, this.size[1] - 4);
            }
            ctx.restore();

            return r;
        };
    },
});
