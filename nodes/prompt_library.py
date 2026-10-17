"""
Y7Nodes Prompt Library

A visual library of saved prompts, each with an optional example image. The node outputs the
prompt of the selected card, or an empty string when nothing is selected.

UI ideas and the storage/route approach are based on NO8D-Prompt-libraries (krea_style_selector.py)
from ComfyUI-NO8D-controls by NO8D (MIT): https://github.com/NO8D/ComfyUI-NO8D-controls

Storage, under ComfyUI's user directory:
    y7_prompt_lib/libraries/<library>.json   {"version": 1, "entries": [{id, name, prompt, preview}]}
    y7_prompt_lib/previews/<id>.webp         downscaled example images

Entries are keyed by a random id rather than by name, so renaming an entry keeps saved workflows
pointing at it.
"""

import base64
import json
import os
import re
import tempfile
import threading
import zipfile
from io import BytesIO
from pathlib import Path, PurePosixPath
from urllib.parse import quote
from uuid import uuid4

from PIL import Image, ImageOps

import folder_paths
from aiohttp import web
from server import PromptServer

from comfy_api.latest import io


_API_PREFIX = "/y7nodes/prompt_library"
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
_PREVIEW_MAX_SIDE = 512
_THUMB_MAX_SIDE = 96  # import dialog thumbnails, sent inline as data URIs
_MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # whole import request
_MAX_ZIP_UNCOMPRESSED = 500 * 1024 * 1024
_MAX_IMAGE_BYTES = 30 * 1024 * 1024
_MAX_ENTRIES = 5000
_ID_RE = re.compile(r"[0-9a-f]{32}")
_LIBRARY_RE = re.compile(r"[\w .+()-]{1,80}", re.UNICODE)

# Every read-modify-write of the library files goes through this lock. aiohttp handlers run on the
# event loop thread while node execution runs on the prompt worker thread.
_lock = threading.RLock()


# ---------------------------------------------------------------------------
# Paths and validation
# ---------------------------------------------------------------------------

def _root() -> Path:
    return Path(folder_paths.get_user_directory()) / "y7_prompt_lib"


def _libraries_dir() -> Path:
    return _root() / "libraries"


def _previews_dir() -> Path:
    return _root() / "previews"


def _safe_library(value) -> str:
    name = " ".join(str(value or "").split())
    if not _LIBRARY_RE.fullmatch(name) or name.startswith("."):
        raise ValueError("Library names may use letters, numbers, spaces and . + ( ) - _ (max 80)")
    return name


def _clean_text(value, field: str, max_length: int) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} cannot be empty")
    if len(text) > max_length:
        raise ValueError(f"{field} is too long (max {max_length} characters)")
    return text


def _clean_name(value) -> str:
    # Names are single-line labels; prompts keep their newlines
    return " ".join(_clean_text(value, "Name", 160).split())


def _library_path(name: str) -> Path:
    return _libraries_dir() / f"{_safe_library(name)}.json"


def _preview_path(entry_id: str) -> Path:
    if not _ID_RE.fullmatch(entry_id or ""):
        raise ValueError("Invalid entry id")
    return _previews_dir() / f"{entry_id}.webp"


# ---------------------------------------------------------------------------
# Library files
# ---------------------------------------------------------------------------

def _atomic_write(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.stem}-", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(temp_name, target)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _library_names() -> list[str]:
    folder = _libraries_dir()
    if not folder.is_dir():
        return []
    names = []
    for path in folder.glob("*.json"):
        try:
            names.append(_safe_library(path.stem))
        except ValueError:
            continue
    return sorted(names, key=str.casefold)


def _read_library(name: str) -> list[dict]:
    path = _library_path(name)
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return []
    entries = []
    for item in raw.get("entries", []) if isinstance(raw, dict) else []:
        if not isinstance(item, dict) or not _ID_RE.fullmatch(str(item.get("id", ""))):
            continue
        entries.append({
            "id": item["id"],
            "name": str(item.get("name") or ""),
            "prompt": str(item.get("prompt") or ""),
            "preview": bool(item.get("preview")),
        })
    return entries


def _write_library(name: str, entries: list[dict]) -> None:
    payload = {"version": 1, "entries": entries}
    data = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    _atomic_write(_library_path(name), data)


def _find_entry(entry_id: str) -> tuple[str, int, list[dict]] | None:
    """Returns (library, index, entries) for an entry id, searching every library."""
    for library in _library_names():
        entries = _read_library(library)
        for index, entry in enumerate(entries):
            if entry["id"] == entry_id:
                return library, index, entries
    return None


def _unique_name(name: str, entries: list[dict], ignore_id: str | None = None) -> str:
    used = {e["name"].casefold() for e in entries if e["id"] != ignore_id}
    if name.casefold() not in used:
        return name
    suffix = 2
    while f"{name} {suffix}".casefold() in used:
        suffix += 1
    return f"{name} {suffix}"


def _delete_preview(entry_id: str) -> None:
    try:
        _preview_path(entry_id).unlink()
    except (OSError, ValueError):
        pass


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

def _open_image(content: bytes) -> Image.Image:
    if not content:
        raise ValueError("Image is empty")
    if len(content) > _MAX_IMAGE_BYTES:
        raise ValueError("Image is too large")
    try:
        image = Image.open(BytesIO(content))
        image.load()
    except Exception as error:
        raise ValueError("Not a readable image") from error
    image = ImageOps.exif_transpose(image)
    return image.convert("RGBA" if image.mode in ("RGBA", "LA", "P") else "RGB")


def _save_preview(entry_id: str, content: bytes) -> None:
    image = _open_image(content)
    image.thumbnail((_PREVIEW_MAX_SIDE, _PREVIEW_MAX_SIDE), Image.LANCZOS)
    buffer = BytesIO()
    image.save(buffer, format="WEBP", quality=88)
    _atomic_write(_preview_path(entry_id), buffer.getvalue())


def _thumb_data_uri(content: bytes) -> str:
    try:
        image = _open_image(content)
    except ValueError:
        return ""
    image.thumbnail((_THUMB_MAX_SIDE, _THUMB_MAX_SIDE), Image.LANCZOS)
    buffer = BytesIO()
    image.save(buffer, format="WEBP", quality=70)
    return "data:image/webp;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Import / export
# ---------------------------------------------------------------------------

def _expand_uploads(files: list[tuple[str, bytes]]) -> list[tuple[str, bytes]]:
    """Flattens uploaded files and the contents of any uploaded .zip into (posix path, bytes)."""
    result = []
    for filename, content in files:
        if PurePosixPath(filename).suffix.lower() != ".zip":
            result.append((PurePosixPath(filename.replace("\\", "/")).name, content))
            continue
        try:
            archive = zipfile.ZipFile(BytesIO(content))
        except zipfile.BadZipFile as error:
            raise ValueError(f"{filename} is not a valid zip file") from error
        with archive:
            members = [m for m in archive.infolist() if not m.is_dir()]
            if sum(m.file_size for m in members) > _MAX_ZIP_UNCOMPRESSED:
                raise ValueError(f"{filename} is too large once unzipped")
            for member in members:
                path = PurePosixPath(member.filename.replace("\\", "/"))
                if path.is_absolute() or ".." in path.parts:
                    continue
                if any(part.startswith(".") or part == "__MACOSX" for part in path.parts):
                    continue
                result.append((path.as_posix(), archive.read(member)))
    return result


def _parse_library_json(files: dict[str, bytes], json_path: str) -> tuple[list[dict], str]:
    """Format 3: a library exported by this node (library.json + previews/)."""
    try:
        raw = json.loads(files[json_path].decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("library.json is not valid JSON") from error
    if not isinstance(raw, dict) or not isinstance(raw.get("entries"), list):
        raise ValueError("library.json has no entries list")
    base = PurePosixPath(json_path).parent
    rows = []
    for item in raw["entries"]:
        if not isinstance(item, dict):
            continue
        image = None
        preview = str(item.get("preview") or "")
        if preview:
            image = files.get((base / preview).as_posix())
        rows.append({"name": str(item.get("name") or ""), "prompt": str(item.get("prompt") or ""), "image": image})
    return rows, str(raw.get("name") or "")


def _parse_paired(files: dict[str, bytes]) -> tuple[list[dict], list[str]]:
    """Format 1: name.txt holds the prompt, name.png/.jpg/.webp (same folder and stem) the image."""
    texts, images = {}, {}
    for path, content in files.items():
        posix = PurePosixPath(path)
        key = posix.with_suffix("").as_posix()
        suffix = posix.suffix.lower()
        if suffix == ".txt":
            texts[key] = content
        elif suffix in _IMAGE_SUFFIXES:
            images.setdefault(key, content)
    rows = []
    for key in sorted(texts, key=str.casefold):
        name = " ".join(PurePosixPath(key).name.replace("_", " ").split())
        prompt = texts[key].decode("utf-8-sig", errors="replace").strip()
        rows.append({"name": name, "prompt": prompt, "image": images.get(key)})
    orphans = sorted((PurePosixPath(k).name for k in images if k not in texts), key=str.casefold)
    return rows, orphans


def _parse_import(files: list[tuple[str, bytes]]) -> dict:
    expanded = dict(_expand_uploads(files))
    json_paths = sorted((p for p in expanded if PurePosixPath(p).name == "library.json"), key=len)
    suggested = ""
    if json_paths:
        rows, suggested = _parse_library_json(expanded, json_paths[0])
        skipped = []
    else:
        rows, orphans = _parse_paired(expanded)
        skipped = [f"{name} (image without a .txt)" for name in orphans]
    zips = [PurePosixPath(name).stem for name, _ in files if name.lower().endswith(".zip")]
    if not suggested and len(zips) == 1:
        suggested = zips[0]

    valid = []
    for row in rows:
        try:
            row["name"] = _clean_name(row["name"])
            row["prompt"] = _clean_text(row["prompt"], "Prompt", 20000)
        except ValueError as error:
            skipped.append(f"{row['name'] or '(unnamed)'}: {error}")
            continue
        valid.append(row)
    if len(valid) > _MAX_ENTRIES:
        raise ValueError(f"Too many entries (max {_MAX_ENTRIES})")
    try:
        suggested = _safe_library(suggested)
    except ValueError:
        suggested = ""
    return {"rows": valid, "skipped": skipped, "suggested_library": suggested}


async def _read_uploads(request) -> tuple[list[tuple[str, bytes]], dict[str, str]]:
    reader = await request.multipart()
    files, fields, total = [], {}, 0
    while True:
        part = await reader.next()
        if part is None:
            break
        if part.filename:
            content = await part.read(decode=False)
            total += len(content)
            if total > _MAX_UPLOAD_BYTES:
                raise ValueError("Upload is too large")
            files.append((part.filename, content))
        else:
            fields[part.name] = await part.text()
    return files, fields


def _export_library(name: str) -> bytes:
    entries = _read_library(name)
    output = BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        exported = []
        for entry in entries:
            preview = ""
            path = _preview_path(entry["id"])
            if entry["preview"] and path.is_file():
                preview = f"previews/{entry['id']}.webp"
                archive.write(path, preview)
            exported.append({"name": entry["name"], "prompt": entry["prompt"], "preview": preview})
        manifest = {"format": "y7_prompt_lib", "version": 1, "name": name, "entries": exported}
        archive.writestr("library.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return output.getvalue()


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def _prompt_for(entry_id: str) -> str:
    if not entry_id:
        return ""
    with _lock:
        found = _find_entry(entry_id)
    if found is None:
        return ""
    library, index, entries = found
    return entries[index]["prompt"]


class Y7Nodes_PromptLibrary(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Y7Nodes_PromptLibrary",
            display_name="Y7 Prompt Library",
            category="Y7Nodes/Prompt",
            inputs=[
                # Both are driven by the gallery UI (web/js/prompt_library.js) and hidden there.
                io.String.Input("library", default="", socketless=True),
                io.String.Input("entry_id", default="", socketless=True),
            ],
            outputs=[
                io.String.Output(display_name="prompt"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, library, entry_id):
        # The prompt text itself, so editing the selected entry in the library re-runs the node
        return _prompt_for(entry_id)

    @classmethod
    def execute(cls, library, entry_id) -> io.NodeOutput:
        return io.NodeOutput(_prompt_for(entry_id))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

def _error(error: Exception, status: int = 400):
    return web.json_response({"error": str(error)}, status=status)


routes = PromptServer.instance.routes


@routes.get(f"{_API_PREFIX}/catalog")
async def y7_prompt_lib_catalog(_request):
    with _lock:
        libraries = []
        for name in _library_names():
            entries = []
            for entry in _read_library(name):
                path = _preview_path(entry["id"])
                has_preview = entry["preview"] and path.is_file()
                entries.append({
                    "id": entry["id"],
                    "name": entry["name"],
                    "prompt": entry["prompt"],
                    "preview_version": path.stat().st_mtime_ns if has_preview else 0,
                })
            libraries.append({"name": name, "entries": entries})
    return web.json_response({"libraries": libraries})


@routes.get(f"{_API_PREFIX}/preview")
async def y7_prompt_lib_preview(request):
    try:
        path = _preview_path(request.rel_url.query.get("id", ""))
    except ValueError:
        return web.Response(status=404)
    if not path.is_file():
        return web.Response(status=404)
    return web.FileResponse(path, headers={"Cache-Control": "max-age=31536000"})


@routes.post(f"{_API_PREFIX}/entry")
async def y7_prompt_lib_save_entry(request):
    """Creates or updates an entry. Multipart: `data` (JSON) and an optional `image` file.

    data: {id?, library, name, prompt, remove_image?}. Changing `library` on an existing entry
    moves it.
    """
    try:
        files, fields = await _read_uploads(request)
        data = json.loads(fields.get("data") or "{}")
        library = _safe_library(data.get("library"))
        name = _clean_name(data.get("name"))
        prompt = _clean_text(data.get("prompt"), "Prompt", 20000)
        image = next((content for filename, content in files), None)
        if image is not None:
            _open_image(image)  # validate before touching any files

        with _lock:
            entry_id = str(data.get("id") or "")
            found = _find_entry(entry_id) if entry_id else None
            if found is None:
                entry_id = uuid4().hex
                entry = {"id": entry_id, "name": name, "prompt": prompt, "preview": False}
            else:
                old_library, index, old_entries = found
                entry = dict(old_entries[index])
                entry.update(name=name, prompt=prompt)
                if old_library != library:
                    del old_entries[index]
                    _write_library(old_library, old_entries)

            if image is not None:
                _save_preview(entry_id, image)
                entry["preview"] = True
            elif data.get("remove_image"):
                _delete_preview(entry_id)
                entry["preview"] = False

            entries = _read_library(library)
            entry["name"] = _unique_name(entry["name"], entries, ignore_id=entry_id)
            index = next((i for i, e in enumerate(entries) if e["id"] == entry_id), -1)
            if index >= 0:
                entries[index] = entry
            else:
                entries.append(entry)
            _write_library(library, entries)
        return web.json_response({"ok": True, "id": entry_id, "name": entry["name"]})
    except (ValueError, json.JSONDecodeError) as error:
        return _error(error)


@routes.post(f"{_API_PREFIX}/entry/delete")
async def y7_prompt_lib_delete_entry(request):
    try:
        body = await request.json()
        entry_id = str(body.get("id") or "")
        with _lock:
            found = _find_entry(entry_id)
            if found is None:
                raise ValueError("Entry not found")
            library, index, entries = found
            del entries[index]
            _write_library(library, entries)
            _delete_preview(entry_id)
        return web.json_response({"ok": True})
    except (ValueError, json.JSONDecodeError) as error:
        return _error(error)


@routes.post(f"{_API_PREFIX}/library/create")
async def y7_prompt_lib_create_library(request):
    try:
        body = await request.json()
        name = _safe_library(body.get("name"))
        with _lock:
            if name.casefold() in {n.casefold() for n in _library_names()}:
                raise ValueError("A library with that name already exists")
            _write_library(name, [])
        return web.json_response({"ok": True, "name": name})
    except (ValueError, json.JSONDecodeError) as error:
        return _error(error)


@routes.post(f"{_API_PREFIX}/library/rename")
async def y7_prompt_lib_rename_library(request):
    try:
        body = await request.json()
        old = _safe_library(body.get("old"))
        new = _safe_library(body.get("new"))
        with _lock:
            names = _library_names()
            if old not in names:
                raise ValueError("Library not found")
            if new != old and new.casefold() in {n.casefold() for n in names if n != old}:
                raise ValueError("A library with that name already exists")
            # os.replace handles case-only renames too
            os.replace(_library_path(old), _library_path(new))
        return web.json_response({"ok": True, "name": new})
    except (ValueError, json.JSONDecodeError, OSError) as error:
        return _error(error)


@routes.post(f"{_API_PREFIX}/library/delete")
async def y7_prompt_lib_delete_library(request):
    try:
        body = await request.json()
        name = _safe_library(body.get("name"))
        with _lock:
            if name not in _library_names():
                raise ValueError("Library not found")
            entries = _read_library(name)
            _library_path(name).unlink()
            for entry in entries:
                _delete_preview(entry["id"])
        return web.json_response({"ok": True, "removed": len(entries)})
    except (ValueError, json.JSONDecodeError, OSError) as error:
        return _error(error)


@routes.get(f"{_API_PREFIX}/library/export")
async def y7_prompt_lib_export_library(request):
    try:
        name = _safe_library(request.rel_url.query.get("name"))
        with _lock:
            if name not in _library_names():
                raise ValueError("Library not found")
            content = _export_library(name)
        filename = quote(f"{name}.zip")
        return web.Response(
            body=content,
            content_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename*=UTF-8''{filename}"},
        )
    except (ValueError, OSError) as error:
        return _error(error)


@routes.post(f"{_API_PREFIX}/import/preview")
async def y7_prompt_lib_import_preview(request):
    try:
        files, _fields = await _read_uploads(request)
        if not files:
            raise ValueError("No files uploaded")
        parsed = _parse_import(files)
        rows = [
            {"name": row["name"], "prompt": row["prompt"], "thumb": _thumb_data_uri(row["image"]) if row["image"] else ""}
            for row in parsed["rows"]
        ]
        return web.json_response({"rows": rows, "skipped": parsed["skipped"], "suggested_library": parsed["suggested_library"]})
    except ValueError as error:
        return _error(error)


@routes.post(f"{_API_PREFIX}/import/commit")
async def y7_prompt_lib_import_commit(request):
    """Re-sends the same files as /import/preview plus a `library` field. Entries whose name already
    exists in the target library are skipped."""
    try:
        files, fields = await _read_uploads(request)
        library = _safe_library(fields.get("library"))
        parsed = _parse_import(files)
        skipped = list(parsed["skipped"])
        imported = 0
        with _lock:
            entries = _read_library(library)
            used = {e["name"].casefold() for e in entries}
            for row in parsed["rows"]:
                if row["name"].casefold() in used:
                    skipped.append(f"{row['name']}: already in {library}")
                    continue
                entry = {"id": uuid4().hex, "name": row["name"], "prompt": row["prompt"], "preview": False}
                if row["image"]:
                    try:
                        _save_preview(entry["id"], row["image"])
                        entry["preview"] = True
                    except ValueError as error:
                        skipped.append(f"{row['name']}: image skipped ({error})")
                entries.append(entry)
                used.add(row["name"].casefold())
                imported += 1
            _write_library(library, entries)
        return web.json_response({"ok": True, "library": library, "imported": imported, "skipped": skipped})
    except ValueError as error:
        return _error(error)
