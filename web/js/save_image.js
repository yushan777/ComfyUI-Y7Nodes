import { app } from "../../../scripts/app.js";

// Starting size, so there's room for the widgets and a usable preview
const DEFAULT_NODE_WIDTH = 360;
const DEFAULT_NODE_HEIGHT = 460;

// The native Save Image gets %date:...% and %Node name.widget% substitution in filename_prefix
// from the frontend's Comfy.SaveImageExtraOutput extension, but that only covers a hard-coded
// list of core node types, so the same hook is added here, for sub_folder as well.

// Characters the frontend swaps for "_" in a %Node.widget% value
const UNSAFE_FILENAME_CHARS = /[/?<>\\:*|"\x00-\x1F\x7F]/g;

// Every node in the graph and its subgraphs, in the order the frontend searches them
function collectAllNodes(graph) {
    const nodes = [];
    for (const node of graph.nodes ?? []) {
        if (node.isSubgraphNode?.() && node.subgraph) {
            nodes.push(...collectAllNodes(node.subgraph));
        }
        nodes.push(node);
    }
    return nodes;
}

// The input socket the frontend gives a widget, which carries its type (INT, FLOAT, ...)
function widgetInput(node, widget) {
    return node.inputs?.find((i) => i.widget?.name === widget.name);
}

// The first widget, across `nodes`, whose label is `label`, with its node. Renaming a widget keeps
// its real name and stores the new one as a label on the widget or its input socket.
function findWidgetByLabel(nodes, label) {
    for (const node of nodes) {
        for (const widget of node.widgets ?? []) {
            if (widget.label === label || widgetInput(node, widget)?.label === label) {
                return { node, widget };
            }
        }
    }
    return null;
}

// A widget value as text, rounded to `decimals` places when it's a number and decimals is set.
// Without decimals, a FLOAT keeps at least one decimal place, so 1.0 gives "1.0" rather than "1".
function formatValue(value, decimals, isFloat) {
    if (typeof value === "number" && Number.isFinite(value)) {
        if (decimals !== null) {
            return value.toFixed(decimals);
        }
        if (isFloat && Number.isInteger(value)) {
            return value.toFixed(1);
        }
    }
    return String(value ?? "");
}

// The frontend's applyTextReplacements, plus three additions for %Node.widget%: renamed widgets,
// nodes referred to by ID, and rounding numbers to a fixed number of decimal places.
//
// The frontend only matches a widget's real name, and renaming a widget (e.g. a Float's `value`
// to `denoise`) only sets a label, so %Float.denoise% is normally left unreplaced. It also never
// matches node IDs, so %73.denoise% is too. Each token is resolved as follows:
//   1. Nodes are matched as the frontend does: "Node name for S&R" first, then title, including
//      nodes inside subgraphs.
//   2. If neither matches and the node part is all digits, the top-level node with that ID is
//      used. Subgraph nodes are skipped because their IDs can repeat those at the top level.
//      Checked last, so a node titled "73" still wins over node 73.
//   3. If the first matching node has a widget with that real name, it is used. For S&R and
//      title matches of non-FLOAT widgets the token goes to the frontend unchanged, so
//      %KSampler.seed% behaves exactly as native; ID matches and FLOAT widgets are resolved
//      here, the first because the frontend can't find them, the second so they keep a
//      decimal place (the frontend writes 1.0 as "1").
//   4. Otherwise the first widget across all matching nodes whose label (on the widget or its
//      input socket) matches is used, so with two Floats %Float.denoise% finds the renamed one.
//   5. Otherwise the token goes to the frontend, which leaves it as is, as native does.
//
// An optional ":N" after the widget rounds a number to N decimal places, e.g. %Float.value:2%
// gives 0.50 rather than 0.5. Non-number values are used as they are. The frontend doesn't know
// this suffix, so these tokens are always resolved here.
function replaceTokens(graph, value, applyTextReplacements) {
    const nodes = collectAllNodes(graph);
    return value.replace(/%([^%]+)%/g, (token, inner) => {
        const parts = inner.split(".");
        if (parts.length === 2) {
            const [nodeName, widgetSpec] = parts;
            const decimalsSuffix = /^(.+):(\d{1,2})$/.exec(widgetSpec);  // 99 max; toFixed throws past 100
            const widgetName = decimalsSuffix ? decimalsSuffix[1] : widgetSpec;
            const decimals = decimalsSuffix ? Number(decimalsSuffix[2]) : null;

            // Same node match as the frontend: "Node name for S&R" first, then title
            let matches = nodes.filter((n) => n.properties?.["Node name for S&R"] === nodeName);
            if (!matches.length) {
                matches = nodes.filter((n) => n.title === nodeName);
            }
            const matchedById = !matches.length && /^\d+$/.test(nodeName);
            if (matchedById) {
                matches = (graph.nodes ?? []).filter((n) => String(n.id) === nodeName);
            }

            const realNameWidget = matches[0]?.widgets?.find((w) => w.name === widgetName);
            const found = realNameWidget
                ? { node: matches[0], widget: realNameWidget }
                : findWidgetByLabel(matches, widgetName);
            if (found) {
                const isFloat = widgetInput(found.node, found.widget)?.type === "FLOAT";
                if (!realNameWidget || matchedById || decimals !== null || isFloat) {
                    return formatValue(found.widget.value, decimals, isFloat).replace(UNSAFE_FILENAME_CHARS, "_");
                }
            }
        }
        // Everything else (real widget names, %date:...%, unknown tokens) is left to the frontend
        return applyTextReplacements(graph, token);
    });
}

app.registerExtension({
    name: "Y7Nodes.SaveImage",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "Y7Nodes_SaveImage") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);

                // Read from comfyAPI rather than importing scripts/utils.js, which logs an
                // "internal module" warning on import
                const applyTextReplacements = window.comfyAPI?.utils?.applyTextReplacements;
                for (const widget of this.widgets ?? []) {
                    if (applyTextReplacements && (widget.name === "sub_folder" || widget.name === "filename_prefix")) {
                        widget.serializeValue = () => replaceTokens(app.graph, widget.value, applyTextReplacements);
                    }
                }

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

                return result;
            };
        }
    },
});
