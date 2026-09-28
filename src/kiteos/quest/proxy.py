# Copyright 2026 Kite ML.
"""KiteOS Quest teleop proxy: lets a USB-tethered Quest reach the dimOS WebXR teleop server.

The headset talks to this machine over USB: ``adb reverse`` maps the headset's
``localhost:PORT`` to this proxy on ``127.0.0.1:PORT``, and the proxy forwards to the
dimOS WebXR server on ``https://127.0.0.1:8443`` (self-signed, so TLS verification is off).

- The Quest opens ``http://localhost:PORT/teleop``. Browsers treat http://localhost as a
  secure context, so WebXR works without a certificate prompt.
- ``/ws`` (the teleop WebSocket) is relayed both ways.
- ``teleop.js`` imports ``https://esm.sh/jsr/@dimos/msgs@<version>``; the proxy rewrites
  that to its own origin and serves the vendored module graph, so the headset needs no
  internet access.
- ``/teleop`` is branded as KiteOS on the fly (title, heading, stylesheet and a header with
  a live status pill). If dimOS changes the page and the expected markup is missing, the
  stock page is served unmodified: branding must never break teleop.
- ``adb reverse`` is re-applied whenever the headset reconnects (sleep, replug).

Usage: ``kiteos quest proxy`` or ``python -m kiteos.quest.proxy [--port 8080] [--no-adb]``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import html
import ipaddress
import logging
import mimetypes
import os
import re
import shutil
import signal
import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple

import aiohttp
from aiohttp import WSCloseCode, WSMsgType, web
from multidict import CIMultiDict

from kiteos import BRAND_NAME, __version__

logger = logging.getLogger("kiteos.quest")

PACKAGE_DIR = Path(__file__).resolve().parent
VENDOR_DIR = PACKAGE_DIR / "vendor" / "esm"
STATIC_DIR = PACKAGE_DIR / "static"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
DEFAULT_UPSTREAM = "https://127.0.0.1:8443"
ADB_POLL_INTERVAL_S = 3.0

HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "host",
        # Dropping these makes upstream bodies arrive uncompressed and lets aiohttp
        # recompute the length, so HTML and JS can be rewritten safely.
        "content-length",
        "content-encoding",
        "accept-encoding",
    }
)

# The session never times out as a whole, because the teleop WebSocket handshake runs
# under the session timeout and a session lasts as long as the operator wants.
_SESSION_TIMEOUT = aiohttp.ClientTimeout(total=None, sock_connect=5)
_REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=30, sock_connect=5)
_PROBE_TIMEOUT = aiohttp.ClientTimeout(total=1.5, sock_connect=1.0)

_CONTENT_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".mjs": "application/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".json": "application/json",
    ".html": "text/html; charset=utf-8",
}


# --------------------------------------------------------------------------------------
# Branding
# --------------------------------------------------------------------------------------

STOCK_TITLE = "DimOS WebXR Teleop"
BRANDED_TITLE = f"{BRAND_NAME} Teleop"

_BODY_OPEN_RE = re.compile(r"<body\b[^>]*>", re.IGNORECASE)

BRAND_HEAD = (
    '    <link rel="stylesheet" href="/kiteos/brand.css">\n'
    '    <script src="/kiteos/status.js" defer></script>\n'
)

BRAND_HEADER = (
    '\n    <header id="kiteos-header" class="kiteos-header">\n'
    '        <img class="kiteos-header__logo" src="/kiteos/logo.svg" alt="" width="44" height="44">\n'
    f'        <span class="kiteos-header__name">{BRAND_NAME}</span>\n'
    '        <span id="kiteos-status" class="kiteos-pill" data-state="checking"'
    ' role="status" aria-live="polite">Checking&hellip;</span>\n'
    "    </header>"
)


def brand_teleop_html(page: str) -> str | None:
    """Return the dimOS teleop page branded as KiteOS, or None if its markup is unexpected.

    Every edit is count-checked against the stock dimOS ``index.html``. If any anchor is
    missing or ambiguous nothing is changed at all: a half-branded page is worse than the
    stock one, and a broken page would stop teleop.
    """
    title = f"<title>{STOCK_TITLE}</title>"
    heading = f"<h1>{STOCK_TITLE}</h1>"
    problems: list[str] = []
    for needle in (title, heading, "</head>"):
        count = page.count(needle)
        if count != 1:
            problems.append(f"expected one {needle!r}, found {count}")
    bodies = _BODY_OPEN_RE.findall(page)
    if len(bodies) != 1:
        problems.append(f"expected one <body> tag, found {len(bodies)}")
    head_end = page.find("</head>")
    stylesheet = page.find("/static/teleop.css")
    if not 0 <= stylesheet < head_end:
        problems.append("teleop.css is not linked in <head>")
    if problems:
        logger.warning(
            "Quest page markup changed (%s); serving the stock page without %s branding",
            "; ".join(problems),
            BRAND_NAME,
        )
        return None

    page = page.replace(title, f"<title>{BRANDED_TITLE}</title>", 1)
    page = page.replace(heading, f"<h1>{BRANDED_TITLE}</h1>", 1)
    # Injected last in <head> so brand.css comes after teleop.css and wins the cascade.
    page = page.replace("</head>", BRAND_HEAD + "</head>", 1)
    return _BODY_OPEN_RE.sub(lambda match: match.group(0) + BRAND_HEADER, page, count=1)


def _offline_page(upstream: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="refresh" content="3">
    <title>{BRANDED_TITLE}: offline</title>
{BRAND_HEAD}</head>
<body class="kiteos-offline">{BRAND_HEADER}
    <main class="kiteos-offline__card">
        <h1>Teleop server is not running</h1>
        <p>Start it on the robot computer with <code>kiteos teleop</code> or
        <code>kiteos collect start</code>. This page reloads by itself.</p>
        <p class="kiteos-offline__detail">Waiting for {html.escape(upstream)}</p>
    </main>
</body>
</html>
"""


# --------------------------------------------------------------------------------------
# Local files
# --------------------------------------------------------------------------------------


def _safe_file(root: Path, rel: str) -> Path | None:
    """Resolve ``rel`` under ``root``; None if it escapes ``root`` or is not a file."""
    rel = rel.lstrip("/")
    if not rel or "\\" in rel or ".." in rel.split("/"):
        return None
    root = root.resolve()
    candidate = (root / rel).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        return None
    return candidate


def vendored(vendor_dir: Path, path: str) -> Path | None:
    """Map a request path to a vendored esm.sh module, mirroring vendor_esm.py's layout.

    ``.mjs``/``.js`` paths map to files; any other path is a bare module URL whose body
    was stored as ``<path>/__entry__.mjs``.
    """
    rel = path.lstrip("/")
    if not rel:
        return None
    if not rel.endswith((".mjs", ".js")):
        rel = rel.rstrip("/") + "/__entry__.mjs"
    return _safe_file(vendor_dir, rel)


def _content_type(path: Path) -> str:
    return _CONTENT_TYPES.get(path.suffix.lower()) or (
        mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    )


def _strip_hop_by_hop(headers: Mapping[str, str]) -> CIMultiDict[str]:
    # Headers named in Connection are hop-by-hop too (RFC 9110 section 7.6.1). A
    # multidict keeps repeated headers such as Set-Cookie intact.
    listed = {token.strip().lower() for token in headers.get("Connection", "").split(",")}
    drop = HOP_BY_HOP | listed
    return CIMultiDict((k, v) for k, v in headers.items() if k.lower() not in drop)


def _has_files(directory: Path) -> bool:
    return directory.is_dir() and any(directory.iterdir())


# --------------------------------------------------------------------------------------
# adb reverse keeper
# --------------------------------------------------------------------------------------

AdbState = Literal["connected", "unauthorized", "absent", "disabled"]
CommandRunner = Callable[[Sequence[str]], Awaitable[tuple[int, str]]]

ADB_INSTALL_HINT = "macOS: brew install android-platform-tools; Linux: sudo apt install adb"


async def run_command(argv: Sequence[str], timeout_s: float = 10.0) -> tuple[int, str]:
    """Run a command and return (exit code, stdout, plus stderr when it failed)."""
    proc = await asyncio.create_subprocess_exec(
        *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout_s)
    except asyncio.TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        await proc.wait()
        return 124, f"timed out after {timeout_s:g}s"
    except BaseException:  # cancelled: never leave a stuck adb behind
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        raise
    text = out.decode(errors="replace")
    code = proc.returncode if proc.returncode is not None else 1
    if code != 0:
        text += err.decode(errors="replace")
    return code, text


def _is_executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def find_adb(
    *,
    which: Callable[[str], str | None] = shutil.which,
    environ: Mapping[str, str] | None = None,
    platform: str | None = None,
    home: Path | None = None,
    is_executable: Callable[[Path], bool] = _is_executable,
) -> str | None:
    """Locate the adb binary: PATH, the Android SDK env vars, then per-platform defaults."""
    found = which("adb")
    if found:
        return found
    environ = os.environ if environ is None else environ
    platform = sys.platform if platform is None else platform
    home = Path.home() if home is None else home
    candidates = [
        Path(environ[var]) / "platform-tools" / "adb"
        for var in ("ANDROID_HOME", "ANDROID_SDK_ROOT")
        if environ.get(var)
    ]
    if platform == "darwin":
        candidates += [
            Path("/opt/homebrew/bin/adb"),
            Path("/usr/local/bin/adb"),
            home / "Library/Android/sdk/platform-tools/adb",
            Path("/usr/bin/adb"),
        ]
    else:
        candidates += [
            Path("/usr/bin/adb"),
            Path("/usr/local/bin/adb"),
            home / "Android/Sdk/platform-tools/adb",
            Path("/opt/homebrew/bin/adb"),
        ]
    for candidate in candidates:
        if is_executable(candidate):
            return str(candidate)
    return None


@dataclass(frozen=True)
class AdbDevice:
    serial: str
    state: str  # "device", "unauthorized", "offline", "no permissions (...)", ...


def parse_adb_devices(output: str) -> list[AdbDevice]:
    """Parse ``adb devices``; lines before the header (daemon start-up noise) are ignored."""
    devices: list[AdbDevice] = []
    in_list = False
    for raw in output.splitlines():
        line = raw.strip()
        if line.startswith("List of devices attached"):
            in_list = True
            continue
        if not in_list or not line or line.startswith("*"):
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            devices.append(AdbDevice(parts[0], parts[1].strip()))
    return devices


def parse_adb_reverse_list(output: str) -> set[tuple[str, str]]:
    """Parse ``adb reverse --list`` lines such as ``UsbFfs tcp:8080 tcp:8080``."""
    pairs: set[tuple[str, str]] = set()
    for line in output.splitlines():
        tokens = line.split()
        if len(tokens) >= 2:
            pairs.add((tokens[-2], tokens[-1]))
    return pairs


class _Observation(NamedTuple):
    state: AdbState
    detail: str
    level: int = logging.INFO


class AdbReverseKeeper:
    """Keeps ``adb reverse tcp:PORT tcp:PORT`` applied on the tethered headset.

    Polls ``adb devices`` every ``interval_s`` seconds and re-applies the mapping after
    sleep or a replug. State changes are logged once. Nothing here ever raises out of
    :meth:`run`; adb trouble only degrades :attr:`state`.
    """

    def __init__(
        self,
        port: int,
        *,
        serial: str | None = None,
        adb_path: str | None = None,
        runner: CommandRunner = run_command,
        discover: Callable[[], str | None] = find_adb,
        interval_s: float = ADB_POLL_INTERVAL_S,
    ) -> None:
        self.port = port
        self.serial = serial
        self.interval_s = interval_s
        self.state: AdbState = "absent"
        self.detail = "waiting for the first adb check"
        self._explicit_adb = adb_path
        self._adb = adb_path
        self._runner = runner
        self._discover = discover
        self._last_logged: tuple[AdbState, str] | None = None

    async def run(self) -> None:
        while True:
            await self.poll_once()
            await asyncio.sleep(self.interval_s)

    async def poll_once(self) -> AdbState:
        try:
            observation = await self._poll()
        except asyncio.CancelledError:
            raise
        except FileNotFoundError:
            missing = self._adb
            if self._explicit_adb is None:
                self._adb = None  # discovered binary vanished: look again next poll
            observation = _Observation(
                "absent", f"adb not found at {missing} ({ADB_INSTALL_HINT})", logging.WARNING
            )
        except Exception as exc:  # adb trouble must never take the proxy down
            observation = _Observation("absent", f"adb check failed: {exc}", logging.WARNING)
        self._record(observation)
        return observation.state

    def _record(self, observation: _Observation) -> None:
        self.state, self.detail = observation.state, observation.detail
        key = (observation.state, observation.detail)
        if key != self._last_logged:
            logger.log(observation.level, "[adb] %s", observation.detail)
            self._last_logged = key

    async def _poll(self) -> _Observation:
        if self._adb is None:
            self._adb = self._discover()
        adb = self._adb
        if adb is None:
            return _Observation("absent", f"adb not found ({ADB_INSTALL_HINT})", logging.WARNING)

        code, out = await self._runner([adb, "devices"])
        if code != 0:
            return _Observation(
                "absent", f"'adb devices' failed: {_first_line(out)}", logging.WARNING
            )
        devices = parse_adb_devices(out)
        if self.serial:
            devices = [d for d in devices if d.serial == self.serial]
        who = f"Quest {self.serial}" if self.serial else "Quest"

        ready = [d for d in devices if d.state == "device"]
        if ready:
            applied: list[str] = []
            failures: list[str] = []
            for device in ready:
                error = await self._ensure_reverse(adb, device.serial)
                if error is None:
                    applied.append(device.serial)
                else:
                    failures.append(f"{device.serial}: {error}")
            if applied:
                return _Observation(
                    "connected",
                    f"{who} connected ({', '.join(applied)}): open "
                    f"http://localhost:{self.port}/teleop in the Quest browser",
                )
            return _Observation(
                "absent",
                f"adb reverse tcp:{self.port} failed ({'; '.join(failures)})",
                logging.WARNING,
            )

        states = {d.state for d in devices}
        if states & {"unauthorized", "authorizing"}:
            return _Observation(
                "unauthorized",
                f"{who} unauthorized: put on the headset and accept 'Allow USB debugging'"
                " (tick 'Always allow')",
                logging.WARNING,
            )
        if any(s.startswith("no permissions") for s in states):
            return _Observation(
                "unauthorized",
                f"{who} visible but USB access is denied: install the KiteOS udev rule "
                "(deploy/udev) and replug the headset",
                logging.WARNING,
            )
        if "offline" in states:
            return _Observation("absent", f"{who} offline: wake the headset or replug the cable")
        return _Observation("absent", f"{who} not detected over USB (check cable / Developer Mode)")

    async def _ensure_reverse(self, adb: str, serial: str) -> str | None:
        """Apply the reverse mapping if missing; return an error message or None."""
        mapping = (f"tcp:{self.port}", f"tcp:{self.port}")
        code, out = await self._runner([adb, "-s", serial, "reverse", "--list"])
        if code == 0 and mapping in parse_adb_reverse_list(out):
            return None
        code, out = await self._runner([adb, "-s", serial, "reverse", *mapping])
        if code != 0:
            return _first_line(out) or f"exit code {code}"
        logger.info("[adb] applied adb reverse tcp:%d on %s", self.port, serial)
        return None


def _first_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else ""


# --------------------------------------------------------------------------------------
# HTTP / WebSocket proxy
# --------------------------------------------------------------------------------------

_SESSION = web.AppKey("kiteos_quest_session", aiohttp.ClientSession)

# Close codes a peer may legitimately send in a close frame (RFC 6455 section 7.4).
_SENDABLE_CLOSE_CODES = (
    frozenset(range(1000, 1004)) | frozenset(range(1007, 1015)) | frozenset(range(3000, 5000))
)
_ESM_IMPORT_RE = re.compile(rb"""https://esm\.sh/([^"'`\s]+)""")


def _ws_url(upstream: str) -> str:
    if upstream.startswith("https://"):
        return "wss://" + upstream[len("https://") :]
    return "ws://" + upstream[len("http://") :]


async def _pump(
    src: web.WebSocketResponse | aiohttp.ClientWebSocketResponse,
    dst: web.WebSocketResponse | aiohttp.ClientWebSocketResponse,
) -> tuple[int, str] | None:
    """Copy messages until ``src`` closes; return its close (code, reason) if it sent one."""
    while True:
        msg = await src.receive()
        if msg.type == WSMsgType.TEXT:
            await dst.send_str(msg.data)
        elif msg.type == WSMsgType.BINARY:
            await dst.send_bytes(msg.data)
        elif msg.type == WSMsgType.CLOSE:
            return int(msg.data), str(msg.extra or "")
        else:  # CLOSING, CLOSED or ERROR
            return None


class QuestProxy:
    """Request handlers; one instance per application."""

    def __init__(
        self,
        upstream: str,
        vendor_dir: Path,
        static_dir: Path,
        adb: AdbReverseKeeper | None,
    ) -> None:
        self.upstream = upstream
        self.ws_upstream = _ws_url(upstream)
        self.vendor_dir = vendor_dir
        self.static_dir = static_dir
        self.adb = adb
        self._upstream_down = False
        self._missing_modules: set[str] = set()
        self._clients: set[web.WebSocketResponse] = set()

    async def handle(self, request: web.Request) -> web.StreamResponse:
        if request.path == "/":
            raise web.HTTPFound("/teleop")
        if request.headers.get("Upgrade", "").lower() == "websocket":
            return await self.relay_ws(request)
        if request.method in ("GET", "HEAD"):
            local = vendored(self.vendor_dir, request.path)
            if local is not None:
                return web.FileResponse(
                    local, headers={"Content-Type": "application/javascript; charset=utf-8"}
                )
        return await self.forward(request)

    async def forward(self, request: web.Request) -> web.StreamResponse:
        session = request.app[_SESSION]
        body = await request.read()
        forwarded = _strip_hop_by_hop(request.headers)
        # Otherwise aiohttp would ask for gzip itself; plain bodies keep the rewrites simple.
        forwarded["Accept-Encoding"] = "identity"
        try:
            async with session.request(
                request.method,
                self.upstream + str(request.rel_url),
                headers=forwarded,
                data=body or None,
                allow_redirects=False,
                timeout=_REQUEST_TIMEOUT,
            ) as resp:
                payload = await resp.read()
                headers = _strip_hop_by_hop(resp.headers)
                status = resp.status
                charset = resp.charset or "utf-8"
                content_type = resp.content_type
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            return self._unavailable(request, exc)
        if self._upstream_down:
            logger.info("Teleop server is reachable again at %s", self.upstream)
            self._upstream_down = False

        if request.path.endswith("teleop.js"):
            payload = self._rewrite_vendored_imports(payload)
        elif (
            request.method == "GET"
            and request.path in ("/teleop", "/teleop/")
            and status == 200
            and content_type == "text/html"
        ):
            payload = _brand_payload(payload, charset)
        location = headers.get("Location")
        if location:
            headers["Location"] = self._local_location(location)
        headers["Cache-Control"] = "no-store"
        return web.Response(status=status, body=payload, headers=headers)

    def _rewrite_vendored_imports(self, script: bytes) -> bytes:
        """Point teleop.js's esm.sh imports at the vendored copies the proxy serves.

        Only imports that are vendored are rewritten; any other import keeps its esm.sh URL, so a
        dimOS pin bump degrades to "needs internet" instead of breaking the page. A warning is
        logged once per module that is not vendored.
        """

        def rewrite(match: re.Match[bytes]) -> bytes:
            spec = "/" + match.group(1).decode(errors="replace")
            if vendored(self.vendor_dir, spec) is not None:
                return spec.encode()
            if spec not in self._missing_modules:
                self._missing_modules.add(spec)
                logger.warning(
                    "teleop.js imports %s, which is not vendored in %s; it is loaded from "
                    "esm.sh, so the Quest needs internet access until it is mirrored "
                    "(see kiteos/quest/README.md)",
                    spec,
                    self.vendor_dir,
                )
            return match.group(0)

        return _ESM_IMPORT_RE.sub(rewrite, script)

    def _local_location(self, location: str) -> str:
        """Keep upstream redirects on the proxy's origin (the headset cannot reach :8443)."""
        if location.startswith(self.upstream):
            rest = location[len(self.upstream) :]
            if not rest or rest.startswith(("/", "?")):
                return rest if rest.startswith("/") else "/" + rest
        return location

    def _unavailable(self, request: web.Request, exc: BaseException) -> web.Response:
        if not self._upstream_down:
            logger.warning(
                "Teleop server not reachable at %s (%s). Start `kiteos teleop` or "
                "`kiteos collect start`; the Quest page retries by itself.",
                self.upstream,
                exc.__class__.__name__,
            )
            self._upstream_down = True
        headers = {"Cache-Control": "no-store"}
        if "text/html" in request.headers.get("Accept", ""):
            return web.Response(
                status=502,
                content_type="text/html",
                text=_offline_page(self.upstream),
                headers=headers,
            )
        return web.Response(
            status=502,
            content_type="text/plain",
            text=(
                f"{BRAND_NAME} teleop server is not running ({self.upstream}). "
                "Start `kiteos teleop` or `kiteos collect start`, then reload."
            ),
            headers=headers,
        )

    async def relay_ws(self, request: web.Request) -> web.WebSocketResponse:
        # compress=False: the stream is small poses plus already-compressed JPEG frames;
        # deflate would only add latency.
        client = web.WebSocketResponse(max_msg_size=0, compress=False)
        await client.prepare(request)
        self._clients.add(client)
        try:
            code, reason = await self._relay(request, client)
            await client.close(code=code, message=reason.encode()[:123])
        finally:
            self._clients.discard(client)
        return client

    async def _relay(self, request: web.Request, client: web.WebSocketResponse) -> tuple[int, str]:
        """Relay until either side closes; return the close code and reason for the headset."""
        session = request.app[_SESSION]
        try:
            async with session.ws_connect(
                self.ws_upstream + str(request.rel_url), ssl=False, max_msg_size=0
            ) as upstream:
                to_upstream = asyncio.create_task(_pump(client, upstream))
                to_client = asyncio.create_task(_pump(upstream, client))
                try:
                    done, _ = await asyncio.wait(
                        {to_upstream, to_client}, return_when=asyncio.FIRST_COMPLETED
                    )
                finally:
                    to_upstream.cancel()
                    to_client.cancel()
                    await asyncio.gather(to_upstream, to_client, return_exceptions=True)
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            logger.warning("[ws] teleop server unavailable at %s: %s", self.ws_upstream, exc)
            return WSCloseCode.TRY_AGAIN_LATER, f"{BRAND_NAME} teleop server is not running"

        for task in done:
            if task.exception() is not None:
                logger.debug("[ws] relay ended: %r", task.exception())
        if to_client not in done:
            return WSCloseCode.OK, ""  # the headset hung up
        upstream_close = None if to_client.exception() else to_client.result()
        if upstream_close is None:
            return WSCloseCode.GOING_AWAY, ""
        if upstream_close[0] in _SENDABLE_CLOSE_CODES:
            # Pass dimOS's reason on, e.g. 1008 "A WebXR control client is already
            # connected", so the headset sees why it was dropped.
            return upstream_close
        return WSCloseCode.OK, ""

    async def close_clients(self, app: web.Application) -> None:
        """on_shutdown hook: end live relays so shutdown does not wait for the headset."""
        for client in list(self._clients):
            await client.close(
                code=WSCloseCode.GOING_AWAY, message=f"{BRAND_NAME} Quest proxy stopped".encode()
            )

    async def status(self, request: web.Request) -> web.Response:
        up = await self._probe_upstream(request.app[_SESSION])
        if self.adb is None:
            adb_state: AdbState = "disabled"
            adb_detail = "adb management disabled"
        else:
            adb_state, adb_detail = self.adb.state, self.adb.detail
        return web.json_response(
            {
                "proxy": "ok",
                "upstream": "up" if up else "down",
                "adb": adb_state,
                "adb_detail": adb_detail,
                "version": __version__,
            },
            headers={"Cache-Control": "no-store"},
        )

    async def _probe_upstream(self, session: aiohttp.ClientSession) -> bool:
        try:
            async with session.get(
                self.upstream + "/teleop", allow_redirects=False, timeout=_PROBE_TIMEOUT
            ) as resp:
                return resp.status < 500
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
            return False

    async def brand_asset(self, request: web.Request) -> web.StreamResponse:
        path = _safe_file(self.static_dir, request.match_info["path"])
        if path is None:
            raise web.HTTPNotFound()
        return web.FileResponse(
            path, headers={"Content-Type": _content_type(path), "Cache-Control": "no-store"}
        )


def _brand_payload(payload: bytes, charset: str) -> bytes:
    try:
        page = payload.decode(charset)
    except (LookupError, UnicodeDecodeError):
        logger.warning("Quest page is not valid %s; serving it without branding", charset)
        return payload
    branded = brand_teleop_html(page)
    return payload if branded is None else branded.encode(charset)


def create_app(
    upstream: str,
    vendor_dir: Path | str = VENDOR_DIR,
    static_dir: Path | str = STATIC_DIR,
    *,
    adb: AdbReverseKeeper | None = None,
) -> web.Application:
    """Build the proxy application.

    ``upstream`` is the dimOS WebXR server origin (``https://host:port``; ``http://`` is
    accepted for tests). ``adb`` only feeds ``/kiteos/api/status``; running the keeper
    is the caller's job (see :func:`serve`).
    """
    upstream = upstream.rstrip("/")
    if not upstream.startswith(("http://", "https://")):
        raise ValueError(f"upstream must be an http(s) URL, got {upstream!r}")
    proxy = QuestProxy(upstream, Path(vendor_dir), Path(static_dir), adb)
    app = web.Application(client_max_size=0)

    async def open_session(app: web.Application) -> None:
        app[_SESSION] = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(ssl=False),
            timeout=_SESSION_TIMEOUT,
            # Shared by every headset request, so it must not collect upstream cookies.
            cookie_jar=aiohttp.DummyCookieJar(),
        )

    async def close_session(app: web.Application) -> None:
        await app[_SESSION].close()

    app.on_startup.append(open_session)
    app.on_shutdown.append(proxy.close_clients)
    app.on_cleanup.append(close_session)
    app.router.add_get("/kiteos/api/status", proxy.status)
    app.router.add_get("/kiteos/{path:.*}", proxy.brand_asset)
    app.router.add_route("*", "/{tail:.*}", proxy.handle)
    return app


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


async def serve(
    host: str,
    port: int,
    upstream: str,
    *,
    adb: AdbReverseKeeper | None = None,
    vendor_dir: Path | str = VENDOR_DIR,
    static_dir: Path | str = STATIC_DIR,
    access_log: bool = False,
) -> None:
    """Run the proxy (and the adb keeper) until SIGINT/SIGTERM."""
    app = create_app(upstream, vendor_dir, static_dir, adb=adb)
    access_logger = logging.getLogger("aiohttp.access") if access_log else None
    # Live relays are closed by on_shutdown; the timeout only bounds stragglers.
    runner = web.AppRunner(app, access_log=access_logger, shutdown_timeout=5.0)
    await runner.setup()
    loop = asyncio.get_running_loop()
    handled: list[signal.Signals] = []
    try:
        await web.TCPSite(runner, host, port).start()
        logger.info(
            "%s Quest proxy %s: http://%s:%d -> %s", BRAND_NAME, __version__, host, port, upstream
        )
        logger.info("On the Quest, open http://localhost:%d/teleop", port)
        if not _has_files(Path(vendor_dir)):
            logger.warning(
                "No vendored esm modules in %s; the Quest will need internet", vendor_dir
            )
        if not _is_loopback(host):
            logger.warning(
                "Listening on %s: anyone on this network can open the teleop page and drive "
                "the arms. Prefer 127.0.0.1 with adb reverse.",
                host,
            )
        if adb is None:
            logger.info("[adb] adb reverse management disabled")

        stop = asyncio.Event()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
                loop.add_signal_handler(sig, stop.set)
                handled.append(sig)
        tasks: set[asyncio.Task[object]] = {asyncio.create_task(stop.wait())}
        if adb is not None:
            tasks.add(asyncio.create_task(adb.run()))
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            task.result()
        logger.info("%s Quest proxy stopped", BRAND_NAME)
    finally:
        for sig in handled:
            loop.remove_signal_handler(sig)
        await runner.cleanup()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kiteos quest proxy",
        description=(
            f"{BRAND_NAME} Quest teleop proxy: serves the branded teleop page to a "
            "USB-tethered Quest on http://localhost and keeps adb reverse applied."
        ),
    )
    parser.add_argument(
        "--host", default=DEFAULT_HOST, help="listen address (default: %(default)s)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="listen port, also reversed onto the headset (default: %(default)s)",
    )
    parser.add_argument(
        "--upstream", default=DEFAULT_UPSTREAM, help="dimOS WebXR server (default: %(default)s)"
    )
    parser.add_argument("--no-adb", action="store_true", help="do not manage adb reverse")
    parser.add_argument(
        "--adb-serial", default=None, help="only manage this headset (adb -s SERIAL)"
    )
    parser.add_argument("--adb-path", default=None, help="adb binary (default: auto-discover)")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging and access log")
    return parser


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s [kiteos quest] %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )
    logger.setLevel(level)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.upstream.startswith(("http://", "https://")):
        parser.error(f"--upstream must be an http(s) URL, got {args.upstream!r}")
    if not 0 < args.port < 65536:
        parser.error(f"--port must be 1-65535, got {args.port}")
    _configure_logging(args.verbose)
    keeper = None
    if not args.no_adb:
        keeper = AdbReverseKeeper(args.port, serial=args.adb_serial, adb_path=args.adb_path)
    try:
        asyncio.run(serve(args.host, args.port, args.upstream, adb=keeper, access_log=args.verbose))
    except KeyboardInterrupt:
        pass
    except OSError as exc:
        logger.error("Cannot listen on %s:%d: %s", args.host, args.port, exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
