# Copyright 2026 Kite ML.
"""Tests for the KiteOS Quest proxy.

Everything runs against in-process aiohttp servers on 127.0.0.1; adb is replaced by a
fake command runner, so no headset, adb binary or robot stack is needed.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import functools
import importlib.util
import ipaddress
import json
import logging
import socket
import ssl
import sys
from collections.abc import AsyncIterator, Callable, Coroutine, Sequence
from pathlib import Path
from typing import Any

import aiohttp
import pytest
from aiohttp import WSMsgType, web
from aiohttp.test_utils import TestClient, TestServer

from kiteos import __version__
from kiteos.quest.proxy import (
    STATIC_DIR,
    VENDOR_DIR,
    AdbReverseKeeper,
    brand_teleop_html,
    build_parser,
    create_app,
    find_adb,
    main,
    parse_adb_devices,
    parse_adb_reverse_list,
    run_command,
    serve,
    vendored,
)

# Same structure as the runtime's dimos/teleop/webxr/web/static/index.html (Apache-2.0) at the
# pinned commit; test_branding_matches_pinned_dimos_page checks the real file as well.
STOCK_INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DimOS WebXR Teleop</title>
    <link rel="stylesheet" href="/static/teleop.css">
</head>
<body>
    <div id="ui">
        <h1>DimOS WebXR Teleop</h1>
        <div id="status">Ready to connect</div>
        <button id="connectBtn" onclick="connect()">Connect</button>
        <button id="disconnectBtn" class="disconnect hidden" onclick="disconnect()">Disconnect</button>
    </div>
    <canvas id="canvas"></canvas>
    <img id="videoFeed" style="display:none" crossorigin="anonymous"/>

    <script type="module" src="/static/teleop.js"></script>
</body>
</html>
"""

TELEOP_JS = """window.onerror = (msg) => console.error(msg);

import { geometry_msgs, std_msgs, sensor_msgs } from "https://esm.sh/jsr/@dimos/msgs@0.1.4";
import { captureBody } from "./webxr_body.mjs";
"""

MSGS_ENTRY = "/jsr/@dimos/msgs@0.1.4"
MSGS_BUNDLE = "/@jsr/dimos__msgs@0.1.4/es2022/std_msgs.mjs"
BUSY_REASON = "A WebXR control client is already connected"


def async_test(fn: Callable[..., Coroutine[Any, Any, None]]) -> Callable[..., None]:
    """Run an async test with asyncio.run (no pytest-asyncio / pytest-aiohttp needed)."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        asyncio.run(fn(*args, **kwargs))

    return wrapper


# --------------------------------------------------------------------------------------
# Fake runtime WebXR server
# --------------------------------------------------------------------------------------


def make_upstream(
    index_html: str = STOCK_INDEX_HTML, teleop_js: str = TELEOP_JS
) -> web.Application:
    async def teleop(request: web.Request) -> web.Response:
        return web.Response(
            text=index_html, content_type="text/html", headers={"X-Upstream": "dimos"}
        )

    async def teleop_config(request: web.Request) -> web.Response:
        return web.json_response({"body_tracking": False})

    async def script(request: web.Request) -> web.Response:
        return web.Response(text=teleop_js, content_type="application/javascript")

    async def echo(request: web.Request) -> web.Response:
        return web.json_response(
            {
                "method": request.method,
                "query": request.query_string,
                "body": await request.text(),
                "headers": {k.lower(): v for k, v in request.headers.items()},
            }
        )

    async def old(request: web.Request) -> web.Response:
        raise web.HTTPTemporaryRedirect(f"{request.scheme}://{request.host}/teleop")

    async def cookies(request: web.Request) -> web.Response:
        response = web.Response(text="ok")
        response.headers.add("Set-Cookie", "a=1; Path=/")
        response.headers.add("Set-Cookie", "b=2; Path=/")
        return response

    async def ws_echo(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(max_msg_size=0)
        await ws.prepare(request)
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                await ws.send_str("echo:" + msg.data)
            elif msg.type == WSMsgType.BINARY:
                await ws.send_bytes(msg.data[::-1])
        return ws

    async def ws_busy(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.close(code=1008, message=BUSY_REASON.encode())
        return ws

    app = web.Application()
    app.router.add_get("/teleop", teleop)
    app.router.add_get("/teleop/config", teleop_config)
    app.router.add_get("/static/teleop.js", script)
    app.router.add_route("*", "/echo", echo)
    app.router.add_get("/old", old)
    app.router.add_get("/cookies", cookies)
    app.router.add_get("/ws", ws_echo)
    app.router.add_get("/ws-busy", ws_busy)
    return app


@contextlib.asynccontextmanager
async def running(app: web.Application, **start_kwargs: Any) -> AsyncIterator[TestServer]:
    server = TestServer(app)
    await server.start_server(**start_kwargs)
    try:
        yield server
    finally:
        await server.close()


@contextlib.asynccontextmanager
async def proxy_client(
    upstream_url: str,
    *,
    adb: AdbReverseKeeper | None = None,
    vendor_dir: Path = VENDOR_DIR,
    static_dir: Path = STATIC_DIR,
) -> AsyncIterator[TestClient]:
    app = create_app(upstream_url, vendor_dir, static_dir, adb=adb)
    async with TestClient(TestServer(app)) as client:
        yield client


@contextlib.asynccontextmanager
async def stack(
    index_html: str = STOCK_INDEX_HTML,
    teleop_js: str = TELEOP_JS,
    *,
    adb: AdbReverseKeeper | None = None,
) -> AsyncIterator[tuple[TestClient, str]]:
    async with running(make_upstream(index_html, teleop_js)) as upstream:
        url = str(upstream.make_url(""))
        async with proxy_client(url, adb=adb) as client:
            yield client, url.rstrip("/")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def dead_upstream() -> str:
    return f"http://127.0.0.1:{free_port()}"


# --------------------------------------------------------------------------------------
# Branding
# --------------------------------------------------------------------------------------


def assert_branded(page: str) -> None:
    assert "<title>KiteOS Teleop</title>" in page
    assert "<h1>KiteOS Teleop</h1>" in page
    assert "DimOS WebXR Teleop" not in page
    teleop_css = page.index("/static/teleop.css")
    brand_css = page.index('<link rel="stylesheet" href="/kiteos/brand.css">')
    head_end = page.index("</head>")
    assert teleop_css < brand_css < head_end
    assert page.index('<script src="/kiteos/status.js" defer></script>') < head_end
    header = page.index('<header id="kiteos-header"')
    assert page.index("<body>") < header < page.index('<div id="ui">')
    assert 'src="/kiteos/logo.svg"' in page
    assert 'id="kiteos-status"' in page
    # The runtime's own page keeps working.
    for untouched in (
        '<div id="status">Ready to connect</div>',
        '<button id="connectBtn" onclick="connect()">Connect</button>',
        '<script type="module" src="/static/teleop.js"></script>',
    ):
        assert untouched in page


def test_brand_teleop_html_brands_stock_page() -> None:
    branded = brand_teleop_html(STOCK_INDEX_HTML)
    assert branded is not None
    assert_branded(branded)


def test_branding_matches_pinned_runtime_page() -> None:
    """Drift check against the real runtime checkout, when it is importable."""
    spec = importlib.util.find_spec("dimos")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("the runtime is not installed in this environment")
    static = Path(next(iter(spec.submodule_search_locations))) / "teleop/webxr/web/static"
    index = static / "index.html"
    if not index.is_file():
        pytest.skip(f"{index} not found")
    branded = brand_teleop_html(index.read_text())
    assert branded is not None, "the runtime's teleop page changed; update kiteos.quest.proxy branding"
    assert_branded(branded)
    # Every esm.sh module the pinned teleop.js imports must be vendored for the offline Quest.
    script = (static / "teleop.js").read_text()
    specs = [part.split('"')[0] for part in script.split("https://esm.sh/")[1:]]
    assert specs
    for spec_path in specs:
        assert vendored(VENDOR_DIR, "/" + spec_path) is not None, spec_path


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda s: s.replace("<title>DimOS WebXR Teleop", "<title>DimOS Teleop"),
            id="title-renamed",
        ),
        pytest.param(lambda s: s.replace("<h1>DimOS WebXR Teleop</h1>", ""), id="heading-missing"),
        pytest.param(
            lambda s: s.replace("</h1>", "</h1><h1>DimOS WebXR Teleop</h1>"),
            id="heading-duplicated",
        ),
        pytest.param(lambda s: s.replace("</head>", ""), id="head-close-missing"),
        pytest.param(
            lambda s: s.replace("/static/teleop.css", "/static/app.css"), id="stylesheet-renamed"
        ),
        pytest.param(lambda s: s.replace("<body>", "<body><body>"), id="body-duplicated"),
    ],
)
def test_brand_teleop_html_refuses_unexpected_markup(
    mutate: Callable[[str], str], caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="kiteos.quest"):
        assert brand_teleop_html(mutate(STOCK_INDEX_HTML)) is None
    assert "serving the stock page" in caplog.text


@async_test
async def test_teleop_page_is_branded_through_proxy() -> None:
    async with stack() as (client, _):
        response = await client.get("/teleop", headers={"Accept": "text/html"})
        assert response.status == 200
        assert response.content_type == "text/html"
        assert response.headers["Cache-Control"] == "no-store"
        assert response.headers["X-Upstream"] == "dimos"
        assert_branded(await response.text())


@async_test
async def test_head_request_does_not_trip_branding_check(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="kiteos.quest"):
        async with stack() as (client, _):
            response = await client.head("/teleop")
            assert response.status == 200
    assert "markup changed" not in caplog.text


@async_test
async def test_teleop_page_served_unmodified_when_markup_changed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    changed = STOCK_INDEX_HTML.replace("DimOS WebXR Teleop", "DimOS Teleop v2")
    with caplog.at_level(logging.WARNING, logger="kiteos.quest"):
        async with stack(index_html=changed) as (client, _):
            response = await client.get("/teleop")
            assert response.status == 200
            assert await response.text() == changed
    assert "serving the stock page" in caplog.text


@async_test
async def test_non_html_responses_pass_through() -> None:
    async with stack() as (client, _):
        response = await client.get("/teleop/config")
        assert response.status == 200
        assert await response.json() == {"body_tracking": False}
        assert response.headers["Cache-Control"] == "no-store"


# --------------------------------------------------------------------------------------
# esm.sh rewrite and vendored modules
# --------------------------------------------------------------------------------------


@async_test
async def test_teleop_js_imports_are_rewritten_to_proxy_origin(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="kiteos.quest"):
        async with stack() as (client, _):
            response = await client.get("/static/teleop.js")
            assert response.status == 200
            body = await response.text()
    assert "https://esm.sh/" not in body
    assert f'from "{MSGS_ENTRY}"' in body
    assert 'from "./webxr_body.mjs"' in body
    assert "not vendored" not in caplog.text


@async_test
async def test_unvendored_esm_import_is_reported(caplog: pytest.LogCaptureFixture) -> None:
    bumped = TELEOP_JS.replace("msgs@0.1.4", "msgs@9.9.9")
    with caplog.at_level(logging.WARNING, logger="kiteos.quest"):
        async with stack(teleop_js=bumped) as (client, _):
            for _ in range(2):
                body = await (await client.get("/static/teleop.js")).text()
    # Not vendored: the import keeps its esm.sh URL (works with internet) instead of a local 404.
    assert 'from "https://esm.sh/jsr/@dimos/msgs@9.9.9"' in body
    assert caplog.text.count("/jsr/@dimos/msgs@9.9.9, which is not vendored") == 1
    assert "needs internet" in caplog.text


@async_test
async def test_vendored_modules_served_locally_even_without_upstream() -> None:
    async with proxy_client(dead_upstream()) as client:
        entry = await client.get(MSGS_ENTRY)
        assert entry.status == 200
        assert entry.headers["Content-Type"].startswith("application/javascript")
        expected = (VENDOR_DIR / MSGS_ENTRY.lstrip("/") / "__entry__.mjs").read_text()
        assert await entry.text() == expected
        assert "https://esm.sh/" not in expected

        bundle = await client.get(MSGS_BUNDLE)
        assert bundle.status == 200
        assert await bundle.text() == (VENDOR_DIR / MSGS_BUNDLE.lstrip("/")).read_text()


def test_vendored_path_resolution(tmp_path: Path) -> None:
    (tmp_path / "jsr/pkg@1").mkdir(parents=True)
    (tmp_path / "jsr/pkg@1/__entry__.mjs").write_text("export {}")
    (tmp_path / "mod.mjs").write_text("export {}")
    (tmp_path.parent / "outside.mjs").write_text("secret")

    assert vendored(tmp_path, "/jsr/pkg@1") == (tmp_path / "jsr/pkg@1/__entry__.mjs").resolve()
    assert vendored(tmp_path, "/jsr/pkg@1/") == (tmp_path / "jsr/pkg@1/__entry__.mjs").resolve()
    assert vendored(tmp_path, "/mod.mjs") == (tmp_path / "mod.mjs").resolve()
    assert vendored(tmp_path, "/") is None
    assert vendored(tmp_path, "/missing.mjs") is None
    assert vendored(tmp_path, "/teleop") is None
    assert vendored(tmp_path, "/../outside.mjs") is None
    assert vendored(tmp_path, "/jsr/../../outside.mjs") is None


# --------------------------------------------------------------------------------------
# /kiteos/* brand assets
# --------------------------------------------------------------------------------------


@async_test
async def test_brand_assets_served_offline() -> None:
    async with proxy_client(dead_upstream()) as client:
        css = await client.get("/kiteos/brand.css")
        assert css.status == 200
        assert css.headers["Content-Type"].startswith("text/css")
        css_text = await css.text()
        assert "#0B1F3A" in css_text and "#2F80ED" in css_text

        logo = await client.get("/kiteos/logo.svg")
        assert logo.status == 200
        assert logo.headers["Content-Type"] == "image/svg+xml"
        assert (await logo.text()).lstrip().startswith("<svg")

        script = await client.get("/kiteos/status.js")
        assert script.status == 200
        assert "javascript" in script.headers["Content-Type"]
        assert "/kiteos/api/status" in await script.text()

        assert (await client.get("/kiteos/nope.css")).status == 404
        assert (await client.get("/kiteos/")).status == 404
        # Encoded so the client does not normalise the dot segments away.
        assert (await client.get("/kiteos/..%2Fproxy.py")).status == 404
        assert (await client.get("/kiteos/%2E%2E%2F..%2Fquest%2Fproxy.py")).status == 404


def test_brand_assets_need_no_network() -> None:
    for name in ("brand.css", "status.js", "logo.svg"):
        text = (STATIC_DIR / name).read_text()
        assert "@import" not in text, name
        assert "https://" not in text, name
        # The only http:// allowed is the SVG namespace identifier.
        assert text.replace("http://www.w3.org/2000/svg", "").count("http://") == 0, name


# --------------------------------------------------------------------------------------
# Plain proxying
# --------------------------------------------------------------------------------------


@async_test
async def test_request_headers_body_and_query_forwarded_without_hop_by_hop() -> None:
    async with stack() as (client, upstream_url):
        response = await client.post(
            "/echo?x=1&y=two",
            data=b"payload",
            headers={
                "X-Custom": "kept",
                "Proxy-Authorization": "Basic c2VjcmV0",
                "Connection": "keep-alive, X-Per-Hop",
                "X-Per-Hop": "dropped",
                "TE": "trailers",
            },
        )
        assert response.status == 200
        assert response.headers["Cache-Control"] == "no-store"
        seen = await response.json()
    assert seen["method"] == "POST"
    assert seen["query"] == "x=1&y=two"
    assert seen["body"] == "payload"
    headers = seen["headers"]
    assert headers["x-custom"] == "kept"
    for dropped in ("proxy-authorization", "x-per-hop", "te"):
        assert dropped not in headers
    assert headers["accept-encoding"] == "identity"  # the client's gzip request is not passed on
    assert headers["host"] == upstream_url.split("://", 1)[1]


@async_test
async def test_repeated_response_headers_preserved() -> None:
    async with stack() as (client, _):
        response = await client.get("/cookies")
        assert sorted(response.headers.getall("Set-Cookie")) == ["a=1; Path=/", "b=2; Path=/"]


@async_test
async def test_root_redirects_to_teleop() -> None:
    async with stack() as (client, _):
        response = await client.get("/", allow_redirects=False)
        assert response.status == 302
        assert response.headers["Location"] == "/teleop"


@async_test
async def test_upstream_redirects_stay_on_proxy_origin() -> None:
    async with stack() as (client, _):
        response = await client.get("/old", allow_redirects=False)
        assert response.status == 307
        assert response.headers["Location"] == "/teleop"


@async_test
async def test_502_page_when_teleop_server_is_down(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="kiteos.quest"):
        async with proxy_client(dead_upstream()) as client:
            page = await client.get("/teleop", headers={"Accept": "text/html,*/*"})
            assert page.status == 502
            assert page.content_type == "text/html"
            assert page.headers["Cache-Control"] == "no-store"
            html = await page.text()

            plain = await client.get("/static/teleop.js", headers={"Accept": "*/*"})
            assert plain.status == 502
            assert plain.content_type == "text/plain"
            text = await plain.text()
    assert "KiteOS" in html and "not running" in html
    assert "kiteos teleop" in html
    assert '<meta http-equiv="refresh"' in html
    assert "/kiteos/brand.css" in html and "/kiteos/status.js" in html
    assert "DimOS" not in html
    assert text.startswith("KiteOS teleop server is not running")
    assert caplog.text.count("Teleop server not reachable") == 1


# --------------------------------------------------------------------------------------
# Status API
# --------------------------------------------------------------------------------------


@async_test
async def test_status_reports_upstream_up() -> None:
    async with stack() as (client, _):
        response = await client.get("/kiteos/api/status")
        assert response.status == 200
        assert response.headers["Cache-Control"] == "no-store"
        status = await response.json()
    assert status["proxy"] == "ok"
    assert status["upstream"] == "up"
    assert status["adb"] == "disabled"
    assert status["version"] == __version__


@async_test
async def test_status_reports_upstream_down() -> None:
    async with proxy_client(dead_upstream()) as client:
        status = await (await client.get("/kiteos/api/status")).json()
    assert status["proxy"] == "ok"
    assert status["upstream"] == "down"


@async_test
async def test_status_reports_adb_keeper_state() -> None:
    fake = FakeAdb(devices=[("1WMHH000000000", "unauthorized")])
    keeper = AdbReverseKeeper(8080, runner=fake, discover=lambda: "adb")
    await keeper.poll_once()
    async with stack(adb=keeper) as (client, _):
        status = await (await client.get("/kiteos/api/status")).json()
    assert status["adb"] == "unauthorized"
    assert "Allow USB debugging" in status["adb_detail"]


# --------------------------------------------------------------------------------------
# WebSocket relay
# --------------------------------------------------------------------------------------


@async_test
async def test_websocket_relays_text_and_binary() -> None:
    async with stack() as (client, _):
        ws = await client.ws_connect("/ws", max_msg_size=0)
        await ws.send_str('{"type":"pose"}')
        assert await ws.receive_str(timeout=5) == 'echo:{"type":"pose"}'
        await ws.send_bytes(b"\x01\x02\x03")
        assert await ws.receive_bytes(timeout=5) == b"\x03\x02\x01"
        big = b"x" * (6 * 1024 * 1024)  # larger than aiohttp's 4 MiB default limit
        await ws.send_bytes(big)
        assert await ws.receive_bytes(timeout=10) == big
        await ws.close()


@async_test
async def test_websocket_close_reason_from_runtime_reaches_headset() -> None:
    async with stack() as (client, _):
        ws = await client.ws_connect("/ws-busy")
        msg = await ws.receive(timeout=5)
        assert msg.type == WSMsgType.CLOSE
        assert msg.data == 1008
        assert msg.extra == BUSY_REASON


@async_test
async def test_websocket_closes_when_teleop_server_is_down() -> None:
    async with proxy_client(dead_upstream()) as client:
        ws = await client.ws_connect("/ws")
        msg = await ws.receive(timeout=5)
        assert msg.type == WSMsgType.CLOSE
        assert msg.data == aiohttp.WSCloseCode.TRY_AGAIN_LATER
        assert "KiteOS" in str(msg.extra)


@async_test
async def test_headset_hanging_up_ends_relay_cleanly() -> None:
    async with stack() as (client, _):
        for _ in range(3):  # the upstream slot is released each time
            ws = await client.ws_connect("/ws")
            await ws.send_str("ping")
            assert await ws.receive_str(timeout=5) == "echo:ping"
            await asyncio.wait_for(ws.close(), 5)
            assert ws.close_code == aiohttp.WSCloseCode.OK


@async_test
async def test_shutdown_closes_live_relays_promptly() -> None:
    async with running(make_upstream()) as upstream:
        client = TestClient(TestServer(create_app(str(upstream.make_url("")))))
        await client.start_server()
        try:
            ws = await client.ws_connect("/ws")
            await ws.send_str("hi")
            assert await ws.receive_str(timeout=5) == "echo:hi"
            closing = asyncio.create_task(client.server.close())
            msg = await ws.receive(timeout=5)
            assert msg.type == WSMsgType.CLOSE
            assert msg.data == aiohttp.WSCloseCode.GOING_AWAY
            await asyncio.wait_for(closing, 5)
        finally:
            await client.close()


def self_signed_server_context(tmp_path: Path) -> ssl.SSLContext:
    pytest.importorskip("cryptography")
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "teleop-test")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    cert_path, key_path = tmp_path / "cert.pem", tmp_path / "key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(cert_path, key_path)
    return context


def test_https_upstream_with_self_signed_certificate(tmp_path: Path) -> None:
    context = self_signed_server_context(tmp_path)

    async def scenario() -> None:
        async with running(make_upstream(), ssl=context) as upstream:
            url = str(upstream.make_url(""))
            assert url.startswith("https://")
            async with proxy_client(url) as client:
                page = await client.get("/teleop")
                assert page.status == 200
                assert_branded(await page.text())
                status = await (await client.get("/kiteos/api/status")).json()
                assert status["upstream"] == "up"
                ws = await client.ws_connect("/ws")
                await ws.send_str("hello")
                assert await ws.receive_str(timeout=5) == "echo:hello"
                await ws.close()

    asyncio.run(scenario())


# --------------------------------------------------------------------------------------
# adb reverse keeper
# --------------------------------------------------------------------------------------


class FakeAdb:
    """Scripted stand-in for the adb binary; records every call."""

    def __init__(
        self,
        devices: Sequence[tuple[str, str]] = (),
        reversed_ports: dict[str, list[int]] | None = None,
        *,
        devices_output: str | None = None,
        fail_reverse: bool = False,
        error: BaseException | None = None,
    ) -> None:
        self.devices = list(devices)
        self.reversed_ports = reversed_ports or {}
        self.devices_output = devices_output
        self.fail_reverse = fail_reverse
        self.error = error
        self.calls: list[list[str]] = []

    async def __call__(self, argv: Sequence[str]) -> tuple[int, str]:
        argv = list(argv)
        self.calls.append(argv)
        if self.error is not None:
            raise self.error
        args = argv[1:]
        if args == ["devices"]:
            if self.devices_output is not None:
                return 0, self.devices_output
            rows = "".join(f"{serial}\t{state}\n" for serial, state in self.devices)
            return 0, f"List of devices attached\n{rows}\n"
        assert args[0] == "-s", argv
        serial, command = args[1], args[2:]
        if command == ["reverse", "--list"]:
            ports = self.reversed_ports.get(serial, [])
            return 0, "".join(f"UsbFfs tcp:{p} tcp:{p}\n" for p in ports)
        if command[0] == "reverse":
            if self.fail_reverse:
                return 1, "adb: error: more than one device/emulator\n"
            port = int(command[1].split(":")[1])
            self.reversed_ports.setdefault(serial, []).append(port)
            return 0, f"{port}\n"
        raise AssertionError(f"unexpected adb call {argv}")

    def reverse_calls(self) -> list[list[str]]:
        return [c for c in self.calls if "reverse" in c and "--list" not in c]


def poll(keeper: AdbReverseKeeper) -> str:
    return asyncio.run(keeper.poll_once())


def keeper_for(fake: FakeAdb, **kwargs: Any) -> AdbReverseKeeper:
    return AdbReverseKeeper(8080, runner=fake, discover=lambda: "/fake/adb", **kwargs)


def test_adb_applies_reverse_when_missing() -> None:
    fake = FakeAdb(devices=[("1WMHH812345678", "device")])
    keeper = keeper_for(fake)
    assert poll(keeper) == "connected"
    assert fake.reverse_calls() == [
        ["/fake/adb", "-s", "1WMHH812345678", "reverse", "tcp:8080", "tcp:8080"]
    ]
    assert "http://localhost:8080/teleop" in keeper.detail

    # Already applied: the next poll only checks.
    fake.calls.clear()
    assert poll(keeper) == "connected"
    assert fake.reverse_calls() == []


def test_adb_keeps_existing_reverse_and_ignores_other_ports() -> None:
    fake = FakeAdb(devices=[("Q1", "device")], reversed_ports={"Q1": [80, 18080]})
    assert poll(keeper_for(fake)) == "connected"
    assert fake.reverse_calls() == [["/fake/adb", "-s", "Q1", "reverse", "tcp:8080", "tcp:8080"]]

    fake = FakeAdb(devices=[("Q1", "device")], reversed_ports={"Q1": [8080]})
    assert poll(keeper_for(fake)) == "connected"
    assert fake.reverse_calls() == []


def test_adb_unauthorized_headset() -> None:
    fake = FakeAdb(devices=[("Q1", "unauthorized")])
    keeper = keeper_for(fake)
    assert poll(keeper) == "unauthorized"
    assert "Allow USB debugging" in keeper.detail
    assert fake.reverse_calls() == []


def test_adb_linux_usb_permission_denied() -> None:
    fake = FakeAdb(
        devices=[("Q1", "no permissions (user in plugdev group; are your udev rules wrong?)")]
    )
    keeper = keeper_for(fake)
    assert poll(keeper) == "unauthorized"
    assert "udev" in keeper.detail


@pytest.mark.parametrize(
    ("devices", "expected_detail"),
    [([], "not detected"), ([("Q1", "offline")], "offline")],
)
def test_adb_absent_or_offline(devices: list[tuple[str, str]], expected_detail: str) -> None:
    keeper = keeper_for(FakeAdb(devices=devices))
    assert poll(keeper) == "absent"
    assert expected_detail in keeper.detail


def test_adb_serial_selects_one_headset() -> None:
    fake = FakeAdb(devices=[("PHONE1", "device"), ("QUEST1", "device")])
    keeper = keeper_for(fake, serial="QUEST1")
    assert poll(keeper) == "connected"
    assert fake.reverse_calls() == [
        ["/fake/adb", "-s", "QUEST1", "reverse", "tcp:8080", "tcp:8080"]
    ]
    assert all("PHONE1" not in call for call in fake.calls)

    missing = keeper_for(FakeAdb(devices=[("PHONE1", "device")]), serial="QUEST1")
    assert poll(missing) == "absent"
    assert "QUEST1" in missing.detail


def test_adb_without_serial_reverses_every_ready_device() -> None:
    fake = FakeAdb(devices=[("A", "device"), ("B", "device"), ("C", "unauthorized")])
    assert poll(keeper_for(fake)) == "connected"
    assert sorted(call[2] for call in fake.reverse_calls()) == ["A", "B"]


def test_adb_reverse_failure_is_reported_not_raised() -> None:
    keeper = keeper_for(FakeAdb(devices=[("Q1", "device")], fail_reverse=True))
    assert poll(keeper) == "absent"
    assert "adb reverse tcp:8080 failed" in keeper.detail


def test_adb_devices_output_with_daemon_startup_noise() -> None:
    output = (
        "* daemon not running; starting now at tcp:5037\n"
        "* daemon started successfully\n"
        "List of devices attached\n"
        "1WMHH812345678\tdevice\n\n"
    )
    assert parse_adb_devices(output)[0].serial == "1WMHH812345678"
    fake = FakeAdb(devices_output=output)
    assert poll(keeper_for(fake)) == "connected"


def test_parse_adb_output() -> None:
    assert parse_adb_devices("garbage without header\nX\tdevice\n") == []
    devices = parse_adb_devices(
        "List of devices attached\nA\tdevice\nB\tno permissions (x); see [y]\n"
    )
    assert [(d.serial, d.state) for d in devices] == [
        ("A", "device"),
        ("B", "no permissions (x); see [y]"),
    ]
    assert parse_adb_reverse_list("UsbFfs tcp:8080 tcp:8080\nhost-19 tcp:9000 tcp:9001\n\n") == {
        ("tcp:8080", "tcp:8080"),
        ("tcp:9000", "tcp:9001"),
    }


def test_adb_binary_missing_reports_absent_without_running_anything(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake = FakeAdb(devices=[("Q1", "device")])
    keeper = AdbReverseKeeper(8080, runner=fake, discover=lambda: None)
    with caplog.at_level(logging.INFO, logger="kiteos.quest"):
        assert poll(keeper) == "absent"
        assert poll(keeper) == "absent"
    assert fake.calls == []
    assert "adb not found" in keeper.detail
    assert caplog.text.count("adb not found") == 1  # state changes are logged once


def test_adb_runner_errors_never_escape() -> None:
    keeper = keeper_for(FakeAdb(error=RuntimeError("boom")))
    assert poll(keeper) == "absent"
    assert "boom" in keeper.detail

    discoveries: list[int] = []

    def discover() -> str:
        discoveries.append(1)
        return "/gone/adb"

    vanished = AdbReverseKeeper(
        8080, runner=FakeAdb(error=FileNotFoundError("/gone/adb")), discover=discover
    )
    assert poll(vanished) == "absent"
    assert poll(vanished) == "absent"
    assert len(discoveries) == 2  # a vanished binary is looked up again


def test_adb_state_transitions_logged_once_each(caplog: pytest.LogCaptureFixture) -> None:
    fake = FakeAdb(devices=[("Q1", "unauthorized")])
    keeper = keeper_for(fake)
    with caplog.at_level(logging.INFO, logger="kiteos.quest"):
        poll(keeper)
        poll(keeper)
        fake.devices = [("Q1", "device")]
        poll(keeper)
        poll(keeper)
    assert caplog.text.count("unauthorized") == 1
    assert caplog.text.count("Quest connected") == 1


def test_adb_keeper_loop_survives_errors() -> None:
    fake = FakeAdb(error=RuntimeError("adb exploded"))
    keeper = keeper_for(fake, interval_s=0.01)

    async def scenario() -> None:
        task = asyncio.create_task(keeper.run())
        await asyncio.sleep(0.1)
        assert not task.done()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert len(fake.calls) >= 3
    assert keeper.state == "absent"


def test_find_adb_prefers_path() -> None:
    assert (
        find_adb(which=lambda name: "/somewhere/adb", is_executable=lambda p: True)
        == "/somewhere/adb"
    )


MAC_SDK = "/home/k/Library/Android/sdk"


@pytest.mark.parametrize(
    ("platform", "present", "expected"),
    [
        ("darwin", {"/opt/homebrew/bin/adb", "/usr/local/bin/adb"}, "/opt/homebrew/bin/adb"),
        ("darwin", {"/usr/local/bin/adb", "/usr/bin/adb"}, "/usr/local/bin/adb"),
        ("darwin", {f"{MAC_SDK}/platform-tools/adb"}, f"{MAC_SDK}/platform-tools/adb"),
        ("linux", {"/usr/bin/adb", "/opt/homebrew/bin/adb"}, "/usr/bin/adb"),
        (
            "linux",
            {"/home/k/Android/Sdk/platform-tools/adb"},
            "/home/k/Android/Sdk/platform-tools/adb",
        ),
        ("linux", {"/sdk/platform-tools/adb", "/usr/bin/adb"}, "/sdk/platform-tools/adb"),
        ("linux", set(), None),
    ],
)
def test_find_adb_fallbacks(platform: str, present: set[str], expected: str | None) -> None:
    found = find_adb(
        which=lambda name: None,
        environ={"ANDROID_HOME": "/sdk"},
        platform=platform,
        home=Path("/home/k"),
        is_executable=lambda path: str(path) in present,
    )
    assert found == expected


def test_run_command_captures_output_failures_and_timeouts() -> None:
    ok = asyncio.run(run_command([sys.executable, "-c", "print('hello')"]))
    assert ok == (0, "hello\n")
    code, out = asyncio.run(
        run_command([sys.executable, "-c", "import sys; sys.stderr.write('bad'); sys.exit(3)"])
    )
    assert code == 3 and "bad" in out
    code, out = asyncio.run(
        run_command([sys.executable, "-c", "import time; time.sleep(30)"], timeout_s=0.3)
    )
    assert code == 124 and "timed out" in out


# --------------------------------------------------------------------------------------
# Command line / serve
# --------------------------------------------------------------------------------------


def test_parser_defaults_and_flags() -> None:
    args = build_parser().parse_args([])
    assert (args.host, args.port, args.upstream) == ("127.0.0.1", 8080, "https://127.0.0.1:8443")
    assert args.no_adb is False and args.adb_serial is None
    args = build_parser().parse_args(
        [
            *("--host", "0.0.0.0", "--port", "9090"),
            *("--upstream", "https://10.0.0.2:8443", "--no-adb", "--adb-serial", "Q1"),
        ]
    )
    assert (args.host, args.port, args.upstream, args.no_adb, args.adb_serial) == (
        "0.0.0.0",
        9090,
        "https://10.0.0.2:8443",
        True,
        "Q1",
    )


def test_bad_upstream_rejected() -> None:
    with pytest.raises(SystemExit):
        main(["--upstream", "ftp://127.0.0.1:8443", "--no-adb"])
    with pytest.raises(ValueError):
        create_app("127.0.0.1:8443")


def test_main_reports_port_in_use() -> None:
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        assert main(["--no-adb", "--port", str(port), "--upstream", dead_upstream()]) == 1


def test_serve_runs_until_cancelled() -> None:
    port = free_port()

    async def scenario() -> None:
        task = asyncio.create_task(serve("127.0.0.1", port, dead_upstream()))
        async with aiohttp.ClientSession() as session:
            for _ in range(100):
                with contextlib.suppress(aiohttp.ClientConnectionError):
                    async with session.get(f"http://127.0.0.1:{port}/kiteos/api/status") as resp:
                        status = json.loads(await resp.text())
                        break
                await asyncio.sleep(0.05)
            else:
                pytest.fail("proxy did not start")
        assert status == {
            "proxy": "ok",
            "upstream": "down",
            "adb": "disabled",
            "adb_detail": "adb management disabled",
            "version": __version__,
        }
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
