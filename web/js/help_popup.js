import { app } from "../../../scripts/app.js";

// Code based on VIDEO HELPER SUITE nodes by Kosinkadink https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite

// Create a shared help DOM element
var helpDOM;
var parentDOM;
if (!app.Y7Nodes_Help) {
    helpDOM = document.createElement("div");
    app.Y7Nodes_Help = helpDOM;
}

// Initialize the help DOM with proper styling and positioning logic
function initHelpDOM() {
    parentDOM = document.createElement("div");
    parentDOM.className = "Y7Nodes_floatinghelp";
    document.body.appendChild(parentDOM);
    parentDOM.appendChild(helpDOM);
    helpDOM.className = "litegraph";
    
    // Add styling
    let scrollbarStyle = document.createElement('style');
    scrollbarStyle.innerHTML = `
        .Y7Nodes_floatinghelp {
            scrollbar-width: 6px;
            scrollbar-color: #0003 #0000;
            background-color: var(--comfy-menu-bg);
            color: var(--comfy-menu-fg);
            box-shadow: 0 0 10px black;
            border-radius: 4px;
            padding: 10px;
            z-index: 1000;
            position: absolute;
            width: 500px;
            min-height: 100px;
            max-height: 600px;
            overflow-y: auto;
        }
        .Y7Nodes_floatinghelp::-webkit-scrollbar {
            background: transparent;
            width: 6px;
        }
        .Y7Nodes_floatinghelp::-webkit-scrollbar-thumb {
            background: #0005;
            border-radius: 20px;
        }
        .Y7Nodes_floatinghelp::-webkit-scrollbar-button {
            display: none;
        }
        .Y7Nodes_collapse {
            margin-bottom: 5px;
        }
        .Y7Nodes_collapse > div:first-child {
            cursor: pointer;
        }
        .Y7Nodes_collapse .Y7Nodes_collapse {
            margin-top: 8px;
        }
        .Y7Nodes_close_button {
            position: absolute;
            top: 2px;
            right: 2px;
            cursor: pointer;
            color: orange;
            font-size: 14px;
            font-weight: bold;
            z-index: 1001;
        }
        .Y7Nodes_close_button:hover {
            color: #FF6666;
        }
    `;
    
    scrollbarStyle.id = 'y7nodes-scroll-properties';
    parentDOM.appendChild(scrollbarStyle);
    
    // Add close button
    const closeButton = document.createElement('div');
    closeButton.className = 'Y7Nodes_close_button';
    closeButton.innerHTML = '✕';
    closeButton.addEventListener('click', function() {
        Y7Nodes_HideHelpPopup();
    });
    parentDOM.appendChild(closeButton);
    
    // Handle positioning updates
    chainCallback(app.canvas, "onDrawForeground", function(ctx, visible_rect) {
        let n = helpDOM.node;
        if (!n || !n?.graph) {
            Y7Nodes_HideHelpPopup();
            return;
        }
        
        // Calculate correct position based on node position and canvas transforms
        const transform = ctx.getTransform();
        const scale = app.canvas.ds.scale;
        const bcr = app.canvas.canvas.getBoundingClientRect();
        const x = transform.e*scale/transform.a + bcr.x;
        const y = transform.f*scale/transform.a + bcr.y;
        
        // Set position and styling
        Object.assign(parentDOM.style, {
            left: (x + (n.pos[0] + n.size[0] + 15) * scale) + "px",
            top: (y + (n.pos[1] - LiteGraph.NODE_TITLE_HEIGHT) * scale) + "px",
            transformOrigin: '0 0',
            transform: 'scale(' + scale + ',' + scale + ')',
            fontSize: '18px',
            display: 'inline',
        });
    });
    
    // Handle collapsible sections
    function setCollapse(el, doCollapse) {
        if (doCollapse) {
            el.children[0].children[0].innerHTML = '+';
            Object.assign(el.children[1].style, {
                color: '#CCC',
                overflowX: 'hidden',
                width: '0px',
                minWidth: 'calc(100% - 20px)',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
            });
            for (let child of el.children[1].children) {
                if (child.style.display != 'none') {
                    child.origDisplay = child.style.display;
                }
                child.style.display = 'none';
            }
        } else {
            el.children[0].children[0].innerHTML = '-';
            Object.assign(el.children[1].style, {
                color: '',
                overflowX: '',
                width: '100%',
                minWidth: '',
                textOverflow: '',
                whiteSpace: '',
            });
            for (let child of el.children[1].children) {
                child.style.display = child.origDisplay;
            }
        }
    }
    
    // Click handler for collapse/expand
    helpDOM.collapseOnClick = function() {
        let doCollapse = this.children[0].innerHTML == '-';
        setCollapse(this.parentElement, doCollapse);
    };
    
    // Function to hide the help popup
    window.Y7Nodes_HideHelpPopup = function() {
        if (parentDOM) {
            parentDOM.style.display = 'none';
        }
        if (helpDOM) {
            helpDOM.node = undefined;
        }
    };
    
    // Hide the popup by default after initialization to prevent it from appearing 
    // at the upper left corner of the interface when ComfyUI starts up.
    // This ensures the help popup only appears when a user clicks on a question mark.
    Y7Nodes_HideHelpPopup();
    
    // Function to show the help popup
    window.Y7Nodes_ShowHelpPopup = function() {
        if (parentDOM) {
            parentDOM.style.display = 'inline';
        }
    };
    
    // Function to navigate to specific sections
    helpDOM.selectHelp = function(name, value) {
        function collapseUnlessMatch(items, t) {
            var match = items.querySelector('[Y7Nodes_title="' + t + '"]');
            if (!match) {
                for (let i of items.children) {
                    if (i.innerHTML.slice(0, t.length + 5).includes(t)) {
                        match = i;
                        break;
                    }
                }
            }
            if (!match) {
                return null;
            }
            match.scrollIntoView(false);
            window.scrollTo(0, 0);
            for (let i of items.querySelectorAll('.Y7Nodes_collapse')) {
                if (i.contains(match)) {
                    setCollapse(i, false);
                } else {
                    setCollapse(i, true);
                }
            }
            return match;
        }
        let target = collapseUnlessMatch(helpDOM, name);
        if (target && value) {
            collapseUnlessMatch(target, value);
        }
    };
    
    // Function to add help functionality to nodes
    helpDOM.addHelp = function(node, nodeType, description) {
        if (!description) {
            return;
        }
        
        // Add question mark to node
        chainCallback(node, "onDrawForeground", function(ctx) {
            if (this?.flags?.collapsed) {
                return;
            }
            
            // Draw question mark
            ctx.save();
            ctx.font = 'bold 20px Arial';
            ctx.fillStyle = 'orange'; 
            ctx.fillText("?", this.size[0] - 17, -8);
            ctx.restore();
        });
        
        // Store description in node
        node.description = description;
        
        // Handle click on question mark
        chainCallback(node, "onMouseDown", function(e, pos, canvas) {
            if (this?.flags?.collapsed) {
                return;
            }
            
            // Check if click is on question mark
            if (pos[1] < 0 && pos[0] + LiteGraph.NODE_TITLE_HEIGHT > this.size[0]) {
                // Toggle help display
                if (helpDOM.node == this) {
                    Y7Nodes_HideHelpPopup();
                } else {
                    helpDOM.node = this;
                    helpDOM.innerHTML = this.description || "no help provided";
                    
                    // Setup collapsible sections
                    for (let e of helpDOM.querySelectorAll('.Y7Nodes_collapse')) {
                        e.children[0].onclick = helpDOM.collapseOnClick;
                        e.children[0].style.cursor = 'pointer';
                    }
                    
                    // Auto-collapse precollapsed sections
                    for (let e of helpDOM.querySelectorAll('.Y7Nodes_precollapse')) {
                        setCollapse(e, true);
                    }
                                        
                    // Show the popup and scroll to top
                    Y7Nodes_ShowHelpPopup();
                    parentDOM.scrollTo(0, 0);
                }
                return true;
            }
        });
    };
}

// Utility function to chain callbacks
function chainCallback(object, property, callback) {
    if (object == undefined) {
        console.error("Tried to add callback to non-existent object");
        return;
    }
    if (property in object && object[property]) {
        const callback_orig = object[property];
        object[property] = function() {
            const r = callback_orig.apply(this, arguments);
            return callback.apply(this, arguments) ?? r;
        };
    } else {
        object[property] = callback;
    }
}

// Main extension registration
app.registerExtension({
    name: "Y7Nodes.HelpPopup",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.category?.startsWith("Y7Nodes")) {
            if (nodeData.description) {
                // Extract short description for tooltips
                const el = document.createElement("div");
                el.innerHTML = nodeData.description;
                const shortDesc = el.querySelector('#Y7_shortdesc')?.textContent || 
                                 el.innerHTML.split('<div')[0] || nodeData.description;
                
                // Store full description but set node description to short text for tooltips
                const fullContent = nodeData.description;
                nodeData.description = shortDesc;
                
                // Ensure our help system is initialized
                if (!app.Y7Nodes_Help || !app.Y7Nodes_Help.parentElement) {
                    initHelpDOM();
                }
                
                // Add help to node on creation
                chainCallback(nodeType.prototype, "onNodeCreated", function() {
                    helpDOM.addHelp(this, nodeType, fullContent);
                    this.setSize(this.computeSize());
                });
            }
        }
    },
    
    async beforeConfigureGraph(graphData, missingNodeTypes) {
        if (helpDOM?.node) {
            Y7Nodes_HideHelpPopup();
        }
    },
    
    async setup() {
        // Initialize if needed
        if (app.Y7Nodes_Help !== helpDOM) {
            helpDOM = app.Y7Nodes_Help;
        } else {
            initHelpDOM();
        }
        
        // Ensure the popup is hidden on startup to prevent the unwanted rectangular window
        // from appearing at the upper left of the ComfyUI interface.
        // This is a safeguard in case the popup wasn't properly hidden during initialization.
        Y7Nodes_HideHelpPopup();
    }
});
