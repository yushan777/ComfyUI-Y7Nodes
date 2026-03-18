// Based on a custom node : mxToolkit.Slider2D by Max Smirnov 
// Cloned as Y7Nodes AspectRatioPicker
import { app } from "../../../scripts/app.js";

class Y7AspectRatioPicker
{
    constructor(node)
    {
        this.node = node;
        this.node.properties = { valueX:1024, valueY:1024, minX:0, minY:0, maxX:2048, maxY:2048, stepX:64, stepY:64, snap: true, dots: true };
        this.node.helpIconOffsetX = 14;
        this.node.intpos = { x:0.5, y:0.5 };
        const min_width_height = 160;
        const shiftLeftTopBottom = 10;  // padding inset on the left, top, and bottom edges of the drawing canvas
        const shiftRight = 60; // reserved width on the right side for the output values/labels panel

        this.node.size = [min_width_height + shiftRight - shiftLeftTopBottom, min_width_height];
        const fontsize = LiteGraph.NODE_SUBTEXT_SIZE;
        const shX = (this.node.slot_start_y || 0)+fontsize*1.5;
        const shY = shX + LiteGraph.NODE_SLOT_HEIGHT;
        const shZ = shY + LiteGraph.NODE_SLOT_HEIGHT;
        const shW = shZ + LiteGraph.NODE_SLOT_HEIGHT;
        function gcd(a, b) { return b === 0 ? a : gcd(b, a % b); }
        function simplifiedRatio(x, y) {
            if (x <= 0 || y <= 0) return `${x}:${y}`;
            const d = gcd(x, y);
            return `${x/d}:${y/d}`;
        }

        for (let i=0; i<2; i++) { this.node.widgets[i].hidden = true; this.node.widgets[i].type = "hidden"; }

        this.node.onAdded = function ()
        {
            this.outputs[0].name = this.outputs[0].localized_name = "";
            this.outputs[1].name = this.outputs[1].localized_name = "";
            this.widgets_start_y = -4.8e8*LiteGraph.NODE_SLOT_HEIGHT;
            this.intpos.x = (this.properties.valueX - this.properties.minX) / (this.properties.maxX - this.properties.minX);
            this.intpos.y = (this.properties.valueY - this.properties.minY) / (this.properties.maxY - this.properties.minY);
            if (this.size[1] > this.size[0]-shiftRight+shiftLeftTopBottom) {this.size[1] = this.size[0]-shiftRight+shiftLeftTopBottom} else {this.size[0] = this.size[1]+shiftRight-shiftLeftTopBottom}
        };

        this.node.onConfigure = function () {}

        this.node.onGraphConfigured = function ()
        {
            this.configured = true;
            this.onPropertyChanged();
        }

        this.node.onPropertyChanged = function (propName)
        {
            if (!this.configured) return;
            if (this.properties.stepX <= 0) this.properties.stepX = 1;
            if (this.properties.stepY <= 0) this.properties.stepY = 1;
            if ( isNaN(this.properties.valueX) ) this.properties.valueX = this.properties.minX;
            if ( isNaN(this.properties.valueY) ) this.properties.valueY = this.properties.minY;
            if ( this.properties.minX >= this.properties.maxX ) this.properties.maxX = this.properties.minX+1;
            if ( this.properties.minY >= this.properties.maxY ) this.properties.maxY = this.properties.minY+1;
            if ((propName === "minX" || propName === "valueX") && ( this.properties.valueX < this.properties.minX )) this.properties.valueX = this.properties.minX;
            if ((propName === "minY" || propName === "valueY") && ( this.properties.valueY < this.properties.minY )) this.properties.valueY = this.properties.minY;
            if ((propName === "maxX" || propName === "valueX") && ( this.properties.valueX > this.properties.maxX )) this.properties.valueX = this.properties.maxX;
            if ((propName === "maxY" || propName === "valueY") && ( this.properties.valueY > this.properties.maxY )) this.properties.valueY = this.properties.maxY;
            this.properties.valueX = Math.round(this.properties.valueX);
            this.properties.valueY = Math.round(this.properties.valueY);
            this.intpos.x = Math.max(0, Math.min(1, (this.properties.valueX-this.properties.minX)/(this.properties.maxX-this.properties.minX)));
            this.intpos.y = Math.max(0, Math.min(1, (this.properties.valueY-this.properties.minY)/(this.properties.maxY-this.properties.minY)));
            this.widgets[0].value = this.properties.valueX;
            this.widgets[1].value = this.properties.valueY;
        }

        const _origDrawForeground = this.node.onDrawForeground;
        this.node.onDrawForeground = function(ctx)
        {
            this.configured = true;
            if ( this.flags.collapsed ) return false;

            ctx.fillStyle="rgba(20,20,20,0.8)";
            ctx.beginPath();
            ctx.roundRect( shiftLeftTopBottom-4, shiftLeftTopBottom-4, this.size[0]-shiftRight-shiftLeftTopBottom+8, this.size[1]-shiftLeftTopBottom-shiftLeftTopBottom+8,4);
            ctx.fill();

            // Dots
            if (this.properties.dots)
            {
                ctx.fillStyle="rgba(200,200,200,0.7)";
                ctx.beginPath();
                let swX = (this.size[0]-shiftRight-shiftLeftTopBottom);
                let swY = (this.size[1]-shiftLeftTopBottom-shiftLeftTopBottom);
                let stX = (swX*this.properties.stepX/(this.properties.maxX-this.properties.minX));
                let stY = (swY*this.properties.stepY/(this.properties.maxY-this.properties.minY));
                for (var ix=0; ix<swX+stX/2; ix+=stX) for (var iy=0; iy<swY+stY/2; iy+=stY) ctx.rect(shiftLeftTopBottom+ix-0.5, shiftLeftTopBottom+iy-0.5, 1, 1);
                ctx.fill();
            }

            ctx.fillStyle="rgba(183, 144, 255, 0.2)";
            ctx.strokeStyle="rgba(228, 17, 17, 0.7)";
            ctx.beginPath();
            ctx.rect(shiftLeftTopBottom, shiftLeftTopBottom+(this.size[1]-shiftLeftTopBottom-shiftLeftTopBottom)*(1-this.intpos.y),(this.size[0]-shiftRight-shiftLeftTopBottom)*this.intpos.x,(this.size[1]-shiftLeftTopBottom-shiftLeftTopBottom)*(this.intpos.y));
            ctx.fill();
            ctx.stroke();

            ctx.fillStyle=LiteGraph.NODE_TEXT_COLOR;
            ctx.beginPath();
            ctx.arc(shiftLeftTopBottom+(this.size[0]-shiftRight-shiftLeftTopBottom)*this.intpos.x, shiftLeftTopBottom+(this.size[1]-shiftLeftTopBottom-shiftLeftTopBottom)*(1-this.intpos.y), 5, 0, 2 * Math.PI, false);
            ctx.fill();

            ctx.lineWidth = 1.5;
            ctx.strokeStyle=this.bgcolor || LiteGraph.NODE_DEFAULT_BGCOLOR;
            ctx.beginPath();
            ctx.arc(shiftLeftTopBottom+(this.size[0]-shiftRight-shiftLeftTopBottom)*this.intpos.x, shiftLeftTopBottom+(this.size[1]-shiftLeftTopBottom-shiftLeftTopBottom)*(1-this.intpos.y), 3, 0, 2 * Math.PI, false);
            ctx.stroke();

            ctx.font = (fontsize) + "px Arial";
            ctx.fillStyle = "rgba(200,200,200,0.45)";
            ctx.textAlign = "right";
            ctx.fillText("w", this.size[0]-14, shX);
            ctx.fillText("h", this.size[0]-14, shY);
            ctx.fillStyle = LiteGraph.NODE_TEXT_COLOR;
            ctx.textAlign = "center";
            ctx.fillText(this.properties.valueX, this.size[0]-shiftRight+21, shX);
            ctx.fillText(this.properties.valueY, this.size[0]-shiftRight+21, shY);

            ctx.fillStyle = "rgba(200,200,200,0.5)";
            ctx.font = (fontsize - 2) + "px Arial";
            ctx.textAlign = "center";
            ctx.fillText(simplifiedRatio(this.properties.valueX, this.properties.valueY), this.size[0]-shiftRight+30, shZ);

            const mp = (this.properties.valueX * this.properties.valueY / 1_000_000).toFixed(1);
            ctx.fillText(mp + "MP", this.size[0]-shiftRight+30, shW);
            if (_origDrawForeground) _origDrawForeground.apply(this, arguments);
        }

        this.node.onDblClick = function(e, pos, canvas)
        {
            if ( e.canvasY - this.pos[1] < 0 ) return false;
            if ( e.canvasX > this.pos[0]+this.size[0]-shiftRight+10 )
            {
                if (e.canvasY - this.pos[1] - 5 < shX)
                {
                    canvas.prompt("valueX", this.properties.valueX, function(v) {if (!isNaN(Number(v))) { this.properties.valueX = Math.round(Number(v)); this.onPropertyChanged("valueX");}}.bind(this), e);
                    return true;
                }
                else if (e.canvasY - this.pos[1] - 5 < shY)
                {
                    canvas.prompt("valueY", this.properties.valueY, function(v) {if (!isNaN(Number(v))) { this.properties.valueY = Math.round(Number(v)); this.onPropertyChanged("valueY");}}.bind(this), e);
                    return true;
                }
            }
        }

        const _origMouseDown = this.node.onMouseDown;
        this.node.onMouseDown = function(e)
        {
            if (e.canvasY - this.pos[1] < 0) return _origMouseDown ? _origMouseDown.apply(this, arguments) : false;
            if (e.shiftKey &&
                e.canvasX < this.pos[0]+this.size[0]-15 &&
                e.canvasX > this.pos[0]+this.size[0]-shiftRight+15 &&
                e.canvasY < this.pos[1]+shY &&
                this.properties.valueX <= this.properties.maxY &&
                this.properties.valueX >= this.properties.minY &&
                this.properties.valueY <= this.properties.maxX &&
                this.properties.valueY >= this.properties.minX)
            {
                let tmpX = this.properties.valueX;
                this.properties.valueX = this.properties.valueY;
                this.properties.valueY = tmpX;
                this.intpos.x = (this.properties.valueX-this.properties.minX)/(this.properties.maxX-this.properties.minX);
                this.intpos.y = (this.properties.valueY-this.properties.minY)/(this.properties.maxY-this.properties.minY);
                this.onPropertyChanged("valueX");
                this.onPropertyChanged("valueY");
                this.updateThisNodeGraph();
                this.graph.setisChangedFlag(this.id);
                return true;
            }

if ( e.canvasX < this.pos[0]+shiftLeftTopBottom-5 || e.canvasX > this.pos[0]+this.size[0]-shiftRight+5 ) return false;
            if ( e.canvasY < this.pos[1]+shiftLeftTopBottom-5 || e.canvasY > this.pos[1]+this.size[1]-shiftLeftTopBottom+5 )  return false;
            this.capture = true;
            this.captureInput(true);
            this.valueUpdate(e);
            return true;
        }

        this.node.onMouseMove = function(e, pos, canvas)
        {
            if (!this.capture) return;
            if ( canvas.pointer.isDown === false ) { this.onMouseUp(e); return; }
            this.valueUpdate(e);
        }

        this.node.valueUpdate = function(e)
        {
            let prevX = this.properties.valueX;
            let prevY = this.properties.valueY;
            let vX = (e.canvasX - this.pos[0] - shiftLeftTopBottom)/(this.size[0]-shiftRight-shiftLeftTopBottom);
            let vY = 1-(e.canvasY - this.pos[1] - shiftLeftTopBottom)/(this.size[1]-shiftLeftTopBottom-shiftLeftTopBottom);
            if (e.shiftKey !== this.properties.snap)
            {
                let sX = this.properties.stepX/(this.properties.maxX - this.properties.minX);
                let sY = this.properties.stepY/(this.properties.maxY - this.properties.minY);
                vX = Math.round(vX/sX)*sX;
                vY = Math.round(vY/sY)*sY;
            }
            if ( vX < 0 ) { vX = 0 } else if ( vX > 1 ) { vX = 1 }
            if ( vY < 0 ) { vY = 0 } else if ( vY > 1 ) { vY = 1 }
            this.intpos.x = vX;
            this.intpos.y = vY;
            this.properties.valueX = Math.round(this.properties.minX + (this.properties.maxX - this.properties.minX) * this.intpos.x);
            this.properties.valueY = Math.round(this.properties.minY + (this.properties.maxY - this.properties.minY) * this.intpos.y);
            this.updateThisNodeGraph?.();
            if ( this.properties.valueX !== prevX || this.properties.valueY !== prevY ) this.graph.setisChangedFlag(this.id);
        }

        this.node.onMouseUp = function()
        {
            if (!this.capture) return;
            this.capture = false;
            this.captureInput(false);
            this.widgets[0].value = this.properties.valueX;
            this.widgets[1].value = this.properties.valueY;
        }

        this.node.onSelected = function(e) { this.onMouseUp(e) }
        this.node.resizable = true;
        this.node.onResize = function(size) {
            if (size[1] < min_width_height) size[1] = min_width_height;
            size[0] = size[1] + shiftRight - shiftLeftTopBottom;
        };
        this.node.computeSize = () => [min_width_height + shiftRight - shiftLeftTopBottom, min_width_height];
    }
}

app.registerExtension(
{
    name: "Y7Nodes.AspectRatioPicker",
    async beforeRegisterNodeDef(nodeType, nodeData, _app)
    {
        if (nodeData.name === "Y7Nodes_AspectRatioPicker")
        {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) onNodeCreated.apply(this, []);
                this.y7AspectRatioPicker = new Y7AspectRatioPicker(this);
            }
        }
    }
});
