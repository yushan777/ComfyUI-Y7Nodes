import { app } from "../../../scripts/app.js";

// Keeps the `select` widget pointing at a model socket that's actually connected, so the number on
// the node is the one that runs. The backend (nodes/switch_model.py) clamps the same way at
// execution time; that stays as the safety net for when `select` is driven by a link, which this
// can't see.
//
// - The widget's max follows the highest connected socket, so it can't be dragged or typed past it.
// - When sockets are connected or disconnected, or the workflow loads, an out-of-range `select`
//   moves to the highest connected socket at or below it, else the lowest connected one - the same
//   rule as _resolve_index in the Python.
//
// The sockets are Autogrow inputs, which the frontend names "models.model_1", "models.model_2"...
// and adds/removes inside its own onConnectionsChange. Syncing is deferred a frame so it reads the
// socket list after that has settled.

const SOCKET_PATTERN = /^models\.model_(\d+)$/;
const MAX_MODELS = 10;

function connectedIndices(node) {
    const indices = [];
    for (const input of node.inputs ?? []) {
        const match = input.name?.match(SOCKET_PATTERN);
        if (match && input.link != null) {
            indices.push(Number(match[1]));
        }
    }
    return indices.sort((a, b) => a - b);
}

function resolveIndex(select, connected) {
    const atOrBelow = connected.filter((i) => i <= select);
    return atOrBelow.length ? atOrBelow[atOrBelow.length - 1] : connected[0];
}

function syncSelect(node) {
    const selectWidget = node.widgets?.find((w) => w.name === "select");
    if (!selectWidget) {
        return;
    }

    const connected = connectedIndices(node);
    selectWidget.options ??= {};
    // Nothing connected yet: leave the full range, there's nothing to clamp to
    selectWidget.options.max = connected.length ? connected[connected.length - 1] : MAX_MODELS;

    // select is converted to an input and driven by a link, so it isn't ours to change
    if (node.inputs?.some((i) => i.widget?.name === "select" && i.link != null)) {
        return;
    }
    if (!connected.length) {
        return;
    }

    const current = Number(selectWidget.value);
    const next = resolveIndex(current, connected);
    if (next !== current) {
        selectWidget.value = next;
        selectWidget.callback?.(next);
    }
    node.setDirtyCanvas?.(true, false);
}

function scheduleSync(node) {
    requestAnimationFrame(() => syncSelect(node));
}

app.registerExtension({
    name: "Y7Nodes.SwitchModel",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "Y7Nodes_SwitchModel") {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);

            // Catches a typed-in value landing on a gap left by a disconnected middle socket
            const selectWidget = this.widgets?.find((w) => w.name === "select");
            if (selectWidget) {
                const callback = selectWidget.callback;
                let syncing = false;
                selectWidget.callback = (...args) => {
                    const cr = callback?.apply(selectWidget, args);
                    if (!syncing) {
                        syncing = true;
                        try {
                            syncSelect(this);
                        } finally {
                            syncing = false;
                        }
                    }
                    return cr;
                };
            }

            scheduleSync(this);
            return r;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const r = onConnectionsChange?.apply(this, arguments);
            scheduleSync(this);
            return r;
        };

        // Links are restored after the node is configured, so wait a frame before reading them
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure?.apply(this, arguments);
            scheduleSync(this);
            return r;
        };
    },
});
