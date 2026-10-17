// Y7 Prompt Library: a card gallery of saved prompts with optional example images.
// Based on NO8D-Prompt-libraries (krea_style_gallery.js) from ComfyUI-NO8D-controls by NO8D (MIT)
// https://github.com/NO8D/ComfyUI-NO8D-controls
import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_NAME = "Y7Nodes_PromptLibrary";
const API = "/y7nodes/prompt_library";
const HIDDEN_WIDGETS = new Set(["library", "entry_id"]);
const MIN_WIDTH = 360;
const MIN_HEIGHT = 420;
const DEFAULT_WIDTH = 560;
const DEFAULT_HEIGHT = 640;
const IMPORT_ACCEPT = ".txt,.png,.jpg,.jpeg,.webp,.zip";

let catalogPromise = null;

// ---------------------------------------------------------------------------
// Catalog
// ---------------------------------------------------------------------------

function getCatalog() {
    catalogPromise ??= api.fetchApi(`${API}/catalog`).then(async (response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json();
    }).catch((error) => {
        catalogPromise = null;
        throw error;
    });
    return catalogPromise;
}

// Reloads the catalog and re-renders every Prompt Library node in the graph
async function refreshAll() {
    catalogPromise = null;
    const catalog = await getCatalog();
    for (const node of app.graph?._nodes || []) {
        if ((node.comfyClass || node.type) !== NODE_NAME || !node._y7pl) continue;
        node._y7pl.catalog = catalog;
        render(node);
    }
    return catalog;
}

function findEntry(catalog, id) {
    if (!id) return null;
    for (const library of catalog?.libraries || []) {
        const entry = library.entries.find((e) => e.id === id);
        if (entry) return { library: library.name, entry };
    }
    return null;
}

function previewUrl(entry) {
    return api.apiURL(`${API}/preview?id=${entry.id}&v=${entry.preview_version}`);
}

async function postJson(path, body) {
    const response = await api.fetchApi(`${API}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    return readResult(response);
}

async function readResult(response) {
    let result = {};
    try { result = await response.json(); } catch {}
    if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
    return result;
}

function notify(severity, summary, detail = "") {
    if (app.extensionManager?.toast?.add) {
        app.extensionManager.toast.add({ severity, summary, detail, life: 3500 });
    } else if (severity === "error") {
        alert(`${summary}\n${detail}`);
    }
}

// ---------------------------------------------------------------------------
// Widgets / node state
// ---------------------------------------------------------------------------

function findWidget(node, name) {
    return (node.widgets || []).find((widget) => widget.name === name);
}

function hideNativeWidgets(node) {
    for (const widget of node.widgets || []) {
        if (!HIDDEN_WIDGETS.has(widget.name)) continue;
        widget.hidden = true;
        widget.type = "converted-widget";
        widget.options = { ...(widget.options || {}), hidden: true };
        widget.computeSize = () => [0, -4];
        widget.draw = () => {};
    }
}

function setWidgetValue(node, name, value) {
    const widget = findWidget(node, name);
    if (!widget || widget.value === value) return;
    widget.value = value;
    widget.callback?.(value);
}

function markGraphChanged(node) {
    node.graph?.change?.();
    node.setDirtyCanvas?.(true, true);
}

function currentLibrary(node) {
    return String(findWidget(node, "library")?.value || "");
}

function selectedId(node) {
    return String(findWidget(node, "entry_id")?.value || "");
}

function setLibrary(node, name) {
    setWidgetValue(node, "library", name);
    node._y7pl.els.grid.scrollTop = 0;
    render(node);
    markGraphChanged(node);
}

function selectEntry(node, id, library = null) {
    setWidgetValue(node, "entry_id", id || "");
    if (library) setWidgetValue(node, "library", library);
    render(node);
    markGraphChanged(node);
}

// ---------------------------------------------------------------------------
// Styles and small DOM helpers
// ---------------------------------------------------------------------------

function ensureStyleSheet() {
    if (document.getElementById("y7pl-style")) return;
    const style = document.createElement("style");
    style.id = "y7pl-style";
    style.textContent = `
.y7pl-root { display:flex; flex-direction:column; gap:6px; width:100%; height:100%; box-sizing:border-box; padding:2px 6px 6px; font:12px sans-serif; color:var(--fg-color,#ddd); }
.y7pl-root button, .y7pl-dialog button { background:var(--comfy-input-bg,#222); color:var(--input-text,#ddd); border:1px solid var(--border-color,#444); border-radius:5px; padding:4px 10px; cursor:pointer; font:12px sans-serif; white-space:nowrap; }
.y7pl-root button:hover, .y7pl-dialog button:hover { border-color:#2563eb; }
.y7pl-root button.primary, .y7pl-dialog button.primary { background:#2563eb; border-color:#2563eb; color:#fff; }
.y7pl-dialog button.danger { background:#991b1b; border-color:#991b1b; color:#fff; }
.y7pl-dialog button:disabled { opacity:.5; cursor:default; }
.y7pl-root input, .y7pl-root select, .y7pl-dialog input, .y7pl-dialog textarea, .y7pl-dialog select { background:var(--comfy-input-bg,#222); color:var(--input-text,#ddd); border:1px solid var(--border-color,#444); border-radius:5px; padding:5px 8px; font:12px sans-serif; box-sizing:border-box; }
.y7pl-row { display:flex; gap:4px; align-items:center; }
.y7pl-library { flex:1; min-width:0; cursor:pointer; }
.y7pl-search { flex:1; min-width:0; }
.y7pl-grid { flex:1; min-height:0; overflow-y:auto; display:grid; grid-template-columns:repeat(auto-fill, minmax(110px, 1fr)); gap:6px; align-content:start; padding:6px; background:rgba(0,0,0,.25); border:1px solid var(--border-color,#444); border-radius:6px; scrollbar-width:thin; }
.y7pl-empty { grid-column:1 / -1; padding:24px 8px; text-align:center; color:var(--descrip-text,#999); line-height:1.6; }
.y7pl-card { display:flex; flex-direction:column; border:2px solid transparent; border-radius:6px; overflow:hidden; background:var(--comfy-input-bg,#222); cursor:pointer; user-select:none; }
.y7pl-card:hover { border-color:#666; }
.y7pl-card.selected { border-color:#2563eb; }
.y7pl-thumb { display:block; width:100%; aspect-ratio:1; object-fit:cover; background:#111; }
.y7pl-noimg { width:100%; aspect-ratio:1; padding:6px; box-sizing:border-box; overflow:hidden; font:11px/1.35 sans-serif; color:var(--descrip-text,#999); white-space:pre-wrap; word-break:break-word; -webkit-mask-image:linear-gradient(#000 70%, transparent); mask-image:linear-gradient(#000 70%, transparent); }
.y7pl-name { padding:4px 6px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; font-weight:600; }
.y7pl-card.selected .y7pl-name { background:#2563eb; color:#fff; }
.y7pl-grid.list { grid-template-columns:minmax(0,1fr); gap:3px; }
.y7pl-row-item { display:grid; grid-template-columns:44px minmax(0,1fr); gap:8px; align-items:center; padding:3px; border:2px solid transparent; border-radius:6px; background:var(--comfy-input-bg,#222); cursor:pointer; user-select:none; }
.y7pl-row-item:hover { border-color:#666; }
.y7pl-row-item.selected { border-color:#2563eb; background:rgba(37,99,235,.25); }
.y7pl-row-item .y7pl-mini { width:44px; height:44px; border-radius:4px; object-fit:cover; background:#111; display:block; }
.y7pl-row-text { min-width:0; display:flex; flex-direction:column; gap:2px; }
.y7pl-row-text b { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.y7pl-row-text span { color:var(--descrip-text,#999); font-size:11px; line-height:1.35; overflow:hidden; display:-webkit-box; -webkit-box-orient:vertical; -webkit-line-clamp:2; word-break:break-word; }
.y7pl-status { color:var(--descrip-text,#999); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; min-height:15px; }
.y7pl-overlay { position:fixed; inset:0; z-index:10000; display:flex; align-items:center; justify-content:center; padding:24px; box-sizing:border-box; background:rgba(0,0,0,.6); }
.y7pl-dialog { display:flex; flex-direction:column; max-height:calc(100vh - 48px); width:100%; overflow:hidden; border:1px solid var(--border-color,#444); border-radius:8px; background:var(--comfy-menu-bg,#242424); color:var(--fg-color,#eee); box-shadow:0 18px 56px rgba(0,0,0,.55); font:13px sans-serif; }
.y7pl-dialog-head { padding:12px 16px; border-bottom:1px solid var(--border-color,#444); font-weight:700; font-size:15px; }
.y7pl-dialog-body { flex:1; min-height:0; overflow:auto; padding:14px 16px; display:flex; flex-direction:column; gap:12px; }
.y7pl-dialog-foot { display:flex; gap:8px; justify-content:flex-end; padding:10px 16px; border-top:1px solid var(--border-color,#444); }
.y7pl-field { display:flex; flex-direction:column; gap:5px; font-weight:600; }
.y7pl-field > :not(span) { width:100%; font-weight:normal; }
.y7pl-drop { position:relative; display:flex; align-items:center; justify-content:center; min-height:140px; border:2px dashed var(--border-color,#555); border-radius:6px; color:var(--descrip-text,#999); text-align:center; cursor:pointer; padding:10px; line-height:1.5; }
.y7pl-drop.dragging { border-color:#2563eb; color:#fff; }
.y7pl-drop img { max-width:100%; max-height:260px; border-radius:4px; }
.y7pl-import-list { display:flex; flex-direction:column; gap:4px; max-height:320px; overflow:auto; }
.y7pl-import-row { display:grid; grid-template-columns:48px minmax(0,1fr); gap:8px; align-items:center; }
.y7pl-import-row img, .y7pl-import-row .y7pl-blank { width:48px; height:48px; object-fit:cover; border-radius:4px; background:#111; }
.y7pl-import-row div { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.y7pl-muted { color:var(--descrip-text,#999); font-weight:normal; }
.y7pl-menu { position:fixed; z-index:10001; min-width:150px; padding:4px; border:1px solid var(--border-color,#444); border-radius:6px; background:var(--comfy-menu-bg,#242424); box-shadow:0 8px 24px rgba(0,0,0,.5); font:12px sans-serif; }
.y7pl-menu div { padding:6px 10px; border-radius:4px; cursor:pointer; color:var(--fg-color,#ddd); }
.y7pl-menu div:hover { background:#2563eb; color:#fff; }
.y7pl-menu div.danger { color:#f87171; }
.y7pl-menu div.danger:hover { background:#991b1b; color:#fff; }
`;
    document.head.append(style);
}

function el(tag, props = {}, ...children) {
    const element = document.createElement(tag);
    for (const [key, value] of Object.entries(props)) {
        if (key === "class") element.className = value;
        else if (key.startsWith("on")) element.addEventListener(key.slice(2), value);
        else element[key] = value;
    }
    element.append(...children.filter((c) => c != null));
    return element;
}

// Keeps clicks, typing and scrolling inside the widget from reaching the canvas / keybindings
function stopCanvasEvents(root) {
    for (const type of ["pointerdown", "mousedown", "click", "dblclick", "contextmenu", "keydown", "keyup", "keypress", "paste"]) {
        root.addEventListener(type, (event) => event.stopPropagation());
    }
}

function openMenu(event, items) {
    event.preventDefault();
    document.querySelector(".y7pl-menu")?.remove();
    const menu = el("div", { class: "y7pl-menu" });
    for (const item of items) {
        menu.append(el("div", {
            class: item.danger ? "danger" : "",
            textContent: item.label,
            onclick: () => { menu.remove(); item.action(); },
        }));
    }
    document.body.append(menu);
    const rect = menu.getBoundingClientRect();
    menu.style.left = `${Math.min(event.clientX, innerWidth - rect.width - 8)}px`;
    menu.style.top = `${Math.min(event.clientY, innerHeight - rect.height - 8)}px`;
    const close = (e) => {
        if (menu.contains(e.target)) return;
        menu.remove();
        document.removeEventListener("pointerdown", close, true);
    };
    setTimeout(() => document.addEventListener("pointerdown", close, true), 0);
}

function modal(title, width = "520px") {
    const body = el("div", { class: "y7pl-dialog-body" });
    const foot = el("div", { class: "y7pl-dialog-foot" });
    const dialog = el("div", { class: "y7pl-dialog" }, el("div", { class: "y7pl-dialog-head", textContent: title }), body, foot);
    dialog.style.maxWidth = width;
    const overlay = el("div", { class: "y7pl-overlay" }, dialog);
    const close = () => overlay.remove();
    overlay.addEventListener("pointerdown", (event) => { if (event.target === overlay) close(); });
    overlay.addEventListener("keydown", (event) => { if (event.key === "Escape") close(); });
    stopCanvasEvents(overlay);
    document.body.append(overlay);
    return { overlay, body, foot, close };
}

function field(label, control) {
    return el("label", { class: "y7pl-field" }, el("span", { textContent: label }), control);
}

function askText(title, label, initial = "") {
    return new Promise((resolve) => {
        const { body, foot, close } = modal(title, "400px");
        const input = el("input", { value: initial });
        const finish = (value) => { close(); resolve(value); };
        input.addEventListener("keydown", (event) => {
            if (event.key === "Enter") finish(input.value.trim() || null);
        });
        body.append(field(label, input));
        foot.append(
            el("button", { textContent: "Cancel", onclick: () => finish(null) }),
            el("button", { class: "primary", textContent: "OK", onclick: () => finish(input.value.trim() || null) }),
        );
        setTimeout(() => { input.focus(); input.select(); }, 0);
    });
}

function askConfirm(title, message, okLabel = "Delete") {
    return new Promise((resolve) => {
        const { body, foot, close } = modal(title, "400px");
        const finish = (value) => { close(); resolve(value); };
        body.append(el("div", { textContent: message, style: "line-height:1.5" }));
        const ok = el("button", { class: "danger", textContent: okLabel, onclick: () => finish(true) });
        foot.append(el("button", { textContent: "Cancel", onclick: () => finish(false) }), ok);
        setTimeout(() => ok.focus(), 0);
    });
}

async function download(response, fallbackName) {
    if (!response.ok) throw new Error((await readResult(response).catch((e) => e)).message || `HTTP ${response.status}`);
    const disposition = response.headers.get("Content-Disposition") || "";
    const match = disposition.match(/filename\*=UTF-8''([^;]+)/);
    const name = match ? decodeURIComponent(match[1]) : fallbackName;
    const url = URL.createObjectURL(await response.blob());
    el("a", { href: url, download: name }).click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// ---------------------------------------------------------------------------
// Library actions
// ---------------------------------------------------------------------------

async function createLibrary(node) {
    const name = await askText("New library", "Library name");
    if (!name) return null;
    try {
        const result = await postJson("/library/create", { name });
        await refreshAll();
        setLibrary(node, result.name);
        return result.name;
    } catch (error) {
        notify("error", "Could not create library", error.message);
        return null;
    }
}

async function renameLibrary(node, name) {
    const next = await askText("Rename library", "Library name", name);
    if (!next || next === name) return;
    try {
        const result = await postJson("/library/rename", { old: name, new: next });
        // Other nodes showing this library follow the rename
        for (const other of app.graph?._nodes || []) {
            if ((other.comfyClass || other.type) === NODE_NAME && currentLibrary(other) === name) {
                setWidgetValue(other, "library", result.name);
            }
        }
        await refreshAll();
        markGraphChanged(node);
    } catch (error) {
        notify("error", "Could not rename library", error.message);
    }
}

async function deleteLibrary(node, library) {
    const count = library.entries.length;
    const ok = await askConfirm(
        "Delete library",
        `Delete "${library.name}" and its ${count} ${count === 1 ? "entry" : "entries"}? Example images are deleted too. This cannot be undone.`,
    );
    if (!ok) return;
    try {
        await postJson("/library/delete", { name: library.name });
        await refreshAll();
    } catch (error) {
        notify("error", "Could not delete library", error.message);
    }
}

async function exportLibrary(name) {
    try {
        await download(await api.fetchApi(`${API}/library/export?name=${encodeURIComponent(name)}`), `${name}.zip`);
    } catch (error) {
        notify("error", "Export failed", error.message);
    }
}

async function deleteEntry(node, entry) {
    const ok = await askConfirm("Delete entry", `Delete "${entry.name}"? This cannot be undone.`);
    if (!ok) return false;
    try {
        await postJson("/entry/delete", { id: entry.id });
        await refreshAll();
        return true;
    } catch (error) {
        notify("error", "Could not delete entry", error.message);
        return false;
    }
}

async function copyPrompt(entry) {
    try {
        await navigator.clipboard.writeText(entry.prompt);
        notify("success", "Prompt copied");
    } catch (error) {
        notify("error", "Copy failed", error.message);
    }
}

// ---------------------------------------------------------------------------
// Entry editor
// ---------------------------------------------------------------------------

function imageFromTransfer(transfer) {
    for (const item of transfer?.items || []) {
        if (item.kind === "file" && item.type.startsWith("image/")) return item.getAsFile();
    }
    return Array.from(transfer?.files || []).find((file) => file.type.startsWith("image/")) || null;
}

async function openEditor(node, entry = null, entryLibrary = null) {
    let catalog = node._y7pl.catalog;
    let defaultLibrary = entryLibrary || currentLibrary(node);
    if (!catalog.libraries.length) {
        defaultLibrary = await createLibrary(node);
        if (!defaultLibrary) return;
        catalog = node._y7pl.catalog;
    }

    const { body, foot, close } = modal(entry ? "Edit entry" : "New entry", "640px");
    const librarySelect = el("select");
    for (const library of catalog.libraries) {
        librarySelect.append(el("option", { value: library.name, textContent: library.name, selected: library.name === defaultLibrary }));
    }
    const nameInput = el("input", { value: entry?.name || "", placeholder: "Shown under the card" });
    const promptInput = el("textarea", { value: entry?.prompt || "", rows: 9, style: "resize:vertical;font-family:monospace" });

    // Example image: keep / replace / remove
    let newImage = null;
    let removeImage = false;
    const fileInput = el("input", { type: "file", accept: "image/png,image/jpeg,image/webp", style: "display:none" });
    const drop = el("div", { class: "y7pl-drop" });
    const removeButton = el("button", { textContent: "Remove image" });
    const showImage = () => {
        drop.replaceChildren();
        let src = null;
        if (newImage) src = URL.createObjectURL(newImage);
        else if (!removeImage && entry?.preview_version) src = previewUrl(entry);
        if (src) drop.append(el("img", { src }));
        else drop.append(el("div", { innerHTML: "Optional example image<br>Click to choose, drop a file here, or paste" }));
        removeButton.style.display = src ? "" : "none";
    };
    const setImage = (file) => {
        if (!file) return;
        newImage = file;
        removeImage = false;
        showImage();
    };
    drop.addEventListener("click", () => fileInput.click());
    drop.addEventListener("dragover", (event) => { event.preventDefault(); drop.classList.add("dragging"); });
    drop.addEventListener("dragleave", () => drop.classList.remove("dragging"));
    drop.addEventListener("drop", (event) => {
        event.preventDefault();
        event.stopPropagation();
        drop.classList.remove("dragging");
        setImage(imageFromTransfer(event.dataTransfer));
    });
    body.parentElement.addEventListener("paste", (event) => {
        const file = imageFromTransfer(event.clipboardData);
        if (file) { event.preventDefault(); setImage(file); }
    });
    fileInput.addEventListener("change", () => setImage(fileInput.files[0]));
    removeButton.addEventListener("click", () => { newImage = null; removeImage = true; showImage(); });
    showImage();

    body.append(
        field("Library", librarySelect),
        field("Name", nameInput),
        field("Prompt", promptInput),
        el("div", { class: "y7pl-field" }, el("span", { textContent: "Example image" }), drop, el("div", {}, removeButton), fileInput),
    );

    const save = el("button", { class: "primary", textContent: "Save" });
    save.addEventListener("click", async () => {
        const name = nameInput.value.trim();
        const prompt = promptInput.value.trim();
        if (!name || !prompt) {
            notify("warn", "Name and prompt are both required");
            return;
        }
        save.disabled = true;
        const form = new FormData();
        form.append("data", JSON.stringify({ id: entry?.id || "", library: librarySelect.value, name, prompt, remove_image: removeImage }));
        if (newImage) form.append("image", newImage, newImage.name || "image.png");
        try {
            const result = await readResult(await api.fetchApi(`${API}/entry`, { method: "POST", body: form }));
            close();
            await refreshAll();
            // A new entry becomes this node's selection; an edited one stays selected if it was
            if (!entry || selectedId(node) === entry.id) {
                selectEntry(node, result.id, librarySelect.value);
            }
            if (result.name !== name) notify("info", `Saved as "${result.name}"`, "That name was already used in this library.");
        } catch (error) {
            save.disabled = false;
            notify("error", "Could not save entry", error.message);
        }
    });
    if (entry) {
        const remove = el("button", { class: "danger", textContent: "Delete", style: "margin-right:auto" });
        remove.addEventListener("click", async () => { if (await deleteEntry(node, entry)) close(); });
        foot.append(remove);
    }
    foot.append(el("button", { textContent: "Cancel", onclick: close }), save);
    setTimeout(() => (entry ? promptInput : nameInput).focus(), 0);
}

// ---------------------------------------------------------------------------
// Import dialog
// ---------------------------------------------------------------------------

async function filesFromDrop(transfer) {
    // Folders dropped from a file manager are walked recursively
    const entries = Array.from(transfer.items || []).map((item) => item.webkitGetAsEntry?.()).filter(Boolean);
    if (!entries.length) return Array.from(transfer.files || []);
    const files = [];
    const walk = async (entry) => {
        if (entry.isFile) {
            files.push(await new Promise((resolve, reject) => entry.file(resolve, reject)));
        } else if (entry.isDirectory) {
            const reader = entry.createReader();
            let batch;
            do {
                batch = await new Promise((resolve, reject) => reader.readEntries(resolve, reject));
                for (const child of batch) await walk(child);
            } while (batch.length);
        }
    };
    for (const entry of entries) await walk(entry);
    return files;
}

function openImport(node) {
    const { body, foot, close } = modal("Import entries", "640px");
    let files = [];

    const fileInput = el("input", { type: "file", multiple: true, accept: IMPORT_ACCEPT, style: "display:none" });
    const folderInput = el("input", { type: "file", multiple: true, webkitdirectory: true, style: "display:none" });
    const drop = el("div", { class: "y7pl-drop", style: "min-height:110px" });
    drop.innerHTML =
        "Drop files, a folder, or a .zip here<br>" +
        "<span class='y7pl-muted'><b>name.txt</b> holds the prompt; <b>name.png / .jpg / .webp</b> (same name) is its image.<br>" +
        "A .zip exported from this node imports as-is.</span>";
    const pickers = el("div", { class: "y7pl-row" },
        el("button", { textContent: "Choose files…", onclick: () => fileInput.click() }),
        el("button", { textContent: "Choose folder…", onclick: () => folderInput.click() }),
    );

    const librarySelect = el("select");
    const newLibraryInput = el("input", { placeholder: "New library name" });
    const target = el("div", { class: "y7pl-row" }, librarySelect, newLibraryInput);
    librarySelect.style.flex = "1";
    newLibraryInput.style.flex = "1";
    const fillLibraries = (suggested) => {
        const libraries = node._y7pl.catalog.libraries.map((l) => l.name);
        librarySelect.replaceChildren(el("option", { value: "", textContent: "New library…" }));
        for (const name of libraries) librarySelect.append(el("option", { value: name, textContent: name }));
        const existing = libraries.find((name) => name.toLowerCase() === suggested.toLowerCase());
        librarySelect.value = existing || (suggested ? "" : currentLibrary(node) || "");
        newLibraryInput.value = existing ? "" : suggested;
        newLibraryInput.style.display = librarySelect.value ? "none" : "";
    };
    librarySelect.addEventListener("change", () => {
        newLibraryInput.style.display = librarySelect.value ? "none" : "";
        if (!librarySelect.value) newLibraryInput.focus();
    });

    const summary = el("div", { class: "y7pl-muted" });
    const list = el("div", { class: "y7pl-import-list" });
    const importButton = el("button", { class: "primary", textContent: "Import", disabled: true });

    const loadFiles = async (chosen) => {
        files = chosen.filter((file) => /\.(txt|png|jpe?g|webp|zip)$/i.test(file.name));
        list.replaceChildren();
        importButton.disabled = true;
        if (!files.length) {
            summary.textContent = "No .txt, image or .zip files found.";
            return;
        }
        summary.textContent = "Reading…";
        const form = new FormData();
        for (const file of files) form.append("files", file, file.name);
        try {
            const result = await readResult(await api.fetchApi(`${API}/import/preview`, { method: "POST", body: form }));
            const withImages = result.rows.filter((row) => row.thumb).length;
            summary.textContent = `${result.rows.length} entries found, ${withImages} with images.`;
            for (const row of result.rows) {
                list.append(el("div", { class: "y7pl-import-row", title: row.prompt },
                    row.thumb ? el("img", { src: row.thumb }) : el("span", { class: "y7pl-blank" }),
                    el("div", {}, el("b", { textContent: row.name }), el("br"), el("span", { class: "y7pl-muted", textContent: row.prompt.split("\n")[0] })),
                ));
            }
            for (const message of result.skipped) {
                list.append(el("div", { class: "y7pl-muted", textContent: `Skipped: ${message}` }));
            }
            const fallback = files.length === 1 ? files[0].name.replace(/\.[^.]+$/, "") : "";
            fillLibraries(result.suggested_library || fallback);
            importButton.disabled = !result.rows.length;
        } catch (error) {
            summary.textContent = `Could not read the files: ${error.message}`;
        }
    };

    fileInput.addEventListener("change", () => loadFiles(Array.from(fileInput.files)));
    folderInput.addEventListener("change", () => loadFiles(Array.from(folderInput.files)));
    drop.addEventListener("click", () => fileInput.click());
    drop.addEventListener("dragover", (event) => { event.preventDefault(); drop.classList.add("dragging"); });
    drop.addEventListener("dragleave", () => drop.classList.remove("dragging"));
    drop.addEventListener("drop", async (event) => {
        event.preventDefault();
        event.stopPropagation();
        drop.classList.remove("dragging");
        loadFiles(await filesFromDrop(event.dataTransfer));
    });

    importButton.addEventListener("click", async () => {
        const library = librarySelect.value || newLibraryInput.value.trim();
        if (!library) {
            notify("warn", "Choose a library to import into");
            return;
        }
        importButton.disabled = true;
        summary.textContent = "Importing…";
        const form = new FormData();
        form.append("library", library);
        for (const file of files) form.append("files", file, file.name);
        try {
            const result = await readResult(await api.fetchApi(`${API}/import/commit`, { method: "POST", body: form }));
            close();
            await refreshAll();
            setLibrary(node, result.library);
            const detail = result.skipped.length ? `${result.skipped.length} skipped:\n${result.skipped.slice(0, 8).join("\n")}` : "";
            notify(result.skipped.length ? "warn" : "success", `Imported ${result.imported} into "${result.library}"`, detail);
        } catch (error) {
            importButton.disabled = false;
            summary.textContent = `Import failed: ${error.message}`;
        }
    });

    fillLibraries("");
    body.append(drop, pickers, fileInput, folderInput, field("Import into", target), summary, list);
    foot.append(el("button", { textContent: "Cancel", onclick: close }), importButton);
}

// ---------------------------------------------------------------------------
// Gallery
// ---------------------------------------------------------------------------

function makeUi(node) {
    const librarySelect = el("select", { class: "y7pl-library", title: "Library" });
    librarySelect.addEventListener("change", () => setLibrary(node, librarySelect.value));
    const addLibrary = el("button", { textContent: "+", title: "New library", onclick: () => createLibrary(node) });
    const libraryMenu = el("button", { textContent: "⋯", title: "Library options" });
    const search = el("input", { class: "y7pl-search", type: "search", placeholder: "Search name or prompt…" });
    const addEntry = el("button", { class: "primary", textContent: "+ Entry", onclick: () => openEditor(node) });
    const importButton = el("button", { textContent: "Import", onclick: () => openImport(node) });
    const viewToggle = el("button", {
        onclick: () => {
            node.properties.y7pl_view = node.properties.y7pl_view === "list" ? "tiles" : "list";
            render(node);
            markGraphChanged(node);
        },
    });
    const grid = el("div", { class: "y7pl-grid" });
    const status = el("div", { class: "y7pl-status", textContent: "Loading…" });

    libraryMenu.addEventListener("click", (event) => {
        const library = node._y7pl.catalog?.libraries.find((l) => l.name === currentLibrary(node));
        if (!library) return;
        openMenu(event, [
            { label: "Rename library…", action: () => renameLibrary(node, library.name) },
            { label: "Export library (.zip)", action: () => exportLibrary(library.name) },
            { label: "Reload from disk", action: () => refreshAll() },
            { label: "Delete library…", danger: true, action: () => deleteLibrary(node, library) },
        ]);
    });
    search.addEventListener("input", () => {
        node._y7pl.search = search.value;
        render(node);
    });
    search.addEventListener("keydown", (event) => {
        if (event.key === "Escape") { search.value = ""; node._y7pl.search = ""; render(node); }
    });
    // Only swallow the wheel when the grid can actually scroll, so the canvas still zooms otherwise
    grid.addEventListener("wheel", (event) => {
        if (grid.scrollHeight > grid.clientHeight) event.stopPropagation();
    });

    const root = el("div", { class: "y7pl-root" },
        el("div", { class: "y7pl-row" }, librarySelect, addLibrary, libraryMenu),
        el("div", { class: "y7pl-row" }, search, viewToggle, addEntry, importButton),
        grid,
        status,
    );
    stopCanvasEvents(root);
    node._y7pl.els = { root, librarySelect, libraryMenu, search, viewToggle, grid, status };
    return root;
}

function makeCard(node, entry, library, selected, asList) {
    let card;
    if (asList) {
        const thumb = entry.preview_version
            ? el("img", { class: "y7pl-mini", src: previewUrl(entry), loading: "lazy", draggable: false })
            : el("span", { class: "y7pl-mini" });
        card = el("div", { class: `y7pl-row-item${selected ? " selected" : ""}`, title: entry.prompt },
            thumb,
            el("div", { class: "y7pl-row-text" }, el("b", { textContent: entry.name }), el("span", { textContent: entry.prompt })),
        );
    } else {
        const visual = entry.preview_version
            ? el("img", { class: "y7pl-thumb", src: previewUrl(entry), loading: "lazy", draggable: false })
            : el("div", { class: "y7pl-noimg", textContent: entry.prompt });
        card = el("div", { class: `y7pl-card${selected ? " selected" : ""}`, title: entry.prompt },
            visual,
            el("div", { class: "y7pl-name", textContent: entry.name }),
        );
    }
    // Clicking the selected card again clears the selection (output becomes blank)
    card.addEventListener("click", () => selectEntry(node, selected ? "" : entry.id, library));
    card.addEventListener("dblclick", () => openEditor(node, entry, library));
    card.addEventListener("contextmenu", (event) => openMenu(event, [
        { label: "Edit…", action: () => openEditor(node, entry, library) },
        { label: "Copy prompt", action: () => copyPrompt(entry) },
        { label: selected ? "Deselect" : "Select", action: () => selectEntry(node, selected ? "" : entry.id, library) },
        { label: "Delete…", danger: true, action: () => deleteEntry(node, entry) },
    ]));
    return card;
}

function render(node) {
    hideNativeWidgets(node);
    const state = node._y7pl;
    const catalog = state?.catalog;
    if (!catalog || !state.els) return;
    const { librarySelect, libraryMenu, search, viewToggle, grid, status } = state.els;

    // Fall back to the library holding the selection, then the first library
    const selection = findEntry(catalog, selectedId(node));
    let libraryName = currentLibrary(node);
    if (!catalog.libraries.some((l) => l.name === libraryName)) {
        libraryName = selection?.library || catalog.libraries[0]?.name || "";
        setWidgetValue(node, "library", libraryName);
    }
    const library = catalog.libraries.find((l) => l.name === libraryName);

    librarySelect.replaceChildren(...catalog.libraries.map((l) => el("option", {
        value: l.name,
        textContent: `${l.name}  (${l.entries.length})`,
    })));
    if (!catalog.libraries.length) librarySelect.append(el("option", { value: "", textContent: "No libraries" }));
    librarySelect.value = libraryName;
    librarySelect.disabled = !catalog.libraries.length;
    libraryMenu.disabled = !library;
    if (search.value !== (state.search || "")) search.value = state.search || "";

    const query = (state.search || "").trim().toLowerCase();
    const entries = (library?.entries || []).filter((e) =>
        !query || e.name.toLowerCase().includes(query) || e.prompt.toLowerCase().includes(query));
    const selected = selectedId(node);
    const scrollTop = grid.scrollTop;
    const asList = node.properties?.y7pl_view === "list";
    grid.classList.toggle("list", asList);
    // The button shows the view it switches to
    viewToggle.textContent = asList ? "▦" : "☰";
    viewToggle.title = asList ? "Show as tiles" : "Show as list";
    if (!catalog.libraries.length) {
        grid.replaceChildren(el("div", { class: "y7pl-empty", innerHTML: "No libraries yet.<br>Click <b>+</b> to create one, or <b>Import</b>." }));
    } else if (!entries.length) {
        grid.replaceChildren(el("div", { class: "y7pl-empty", textContent: query ? "No matches." : "This library is empty. Click + Entry to add a prompt." }));
    } else {
        grid.replaceChildren(...entries.map((entry) => makeCard(node, entry, libraryName, entry.id === selected, asList)));
    }
    grid.scrollTop = scrollTop;

    if (selection) {
        const where = selection.library === libraryName ? "" : `  (in ${selection.library})`;
        status.textContent = `Selected: ${selection.entry.name}${where}`;
    } else if (selected) {
        status.textContent = "The selected entry no longer exists. Output is blank.";
    } else {
        status.textContent = "Nothing selected. Output is blank.";
    }
}

function installUi(node) {
    if (node._y7pl || typeof node.addDOMWidget !== "function") return;
    ensureStyleSheet();
    node._y7pl = { catalog: null, search: "", els: null };
    node.properties ??= {};
    node.properties.y7pl_view ??= "tiles";  // "tiles" or "list"; saved with the workflow
    hideNativeWidgets(node);
    const root = makeUi(node);
    const widget = node.addDOMWidget("y7_prompt_library", "y7_prompt_library", root, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => MIN_HEIGHT - 60,
    });
    widget.serialize = false;
    widget.computeLayoutSize = () => ({ minWidth: MIN_WIDTH, minHeight: MIN_HEIGHT - 60, maxWidth: 1_000_000, maxHeight: 1_000_000 });

    getCatalog()
        .then((catalog) => {
            node._y7pl.catalog = catalog;
            render(node);
        })
        .catch((error) => {
            node._y7pl.els.status.textContent = `Could not load the prompt library: ${error.message || error}`;
        });
}

app.registerExtension({
    name: "Y7Nodes.PromptLibrary",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            installUi(this);
            // Deferred so it runs after other extensions' setSize calls (e.g. the help popup).
            // Nodes loaded from a workflow keep their saved size.
            requestAnimationFrame(() => {
                if (this._y7plConfigured) return;
                this.setSize([Math.max(this.size[0], DEFAULT_WIDTH), Math.max(this.size[1], DEFAULT_HEIGHT)]);
                this.setDirtyCanvas(true, true);
            });
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            this._y7plConfigured = true;
            hideNativeWidgets(this);
            // Widget values are restored by now; re-render so the saved library and selection show
            render(this);
            return result;
        };

        const onResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function (size) {
            size[0] = Math.max(size[0], MIN_WIDTH);
            size[1] = Math.max(size[1], MIN_HEIGHT);
            return onResize?.apply(this, arguments);
        };
    },
});
