"""Reverse proxy that lets a USB-tethered Quest reach the dimOS WebXR teleop server.

The Quest cannot reach this Mac over the (corporate) Wi-Fi, so it talks to it over USB:
`adb reverse` maps the headset's localhost:PORT to this proxy on 127.0.0.1:PORT, and the
proxy forwards to dimOS on https://127.0.0.1:8443.

- The Quest opens http://localhost:PORT/teleop. Browsers treat http://localhost as a secure
  context, so WebXR works without a certificate prompt; the proxy speaks HTTPS to dimOS.
- /ws (the teleop WebSocket) is relayed both ways.
- teleop.js imports https://esm.sh/jsr/@dimos/msgs@0.1.4; the proxy rewrites that to its
  own origin and serves the module graph mirrored by vendor_esm.py, so the headset
  needs no internet access.
- `adb reverse` is re-applied whenever the headset reconnects (sleep, replug).

Usage: python quest_proxy.py [--port 8080] [--upstream https://127.0.0.1:8443]
"""

import argparse
import asyncio
import shutil
from pathlib import Path

import aiohttp
from aiohttp import WSMsgType, web

VENDOR = Path(__file__).resolve().parent / "vendor" / "esm"
HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer",
    "transfer-encoding", "upgrade", "host", "content-length", "content-encoding", "accept-encoding",
}


def vendored(path: str) -> Path | None:
    rel = path.lstrip("/")
    if not rel or ".." in rel.split("/"):
        return None
    candidate = VENDOR / rel if rel.endswith((".mjs", ".js")) else VENDOR / rel / "__entry__.mjs"
    return candidate if candidate.is_file() else None


async def relay_ws(request: web.Request, upstream: str, session: aiohttp.ClientSession) -> web.WebSocketResponse:
    client = web.WebSocketResponse(max_msg_size=0)
    await client.prepare(request)
    try:
        async with session.ws_connect(upstream.replace("https://", "wss://") + request.path_qs, ssl=False, max_msg_size=0) as up:

            async def pump(src, dst) -> None:
                async for msg in src:
                    if msg.type == WSMsgType.TEXT:
                        await dst.send_str(msg.data)
                    elif msg.type == WSMsgType.BINARY:
                        await dst.send_bytes(msg.data)
                    else:
                        break

            tasks = [asyncio.create_task(pump(client, up)), asyncio.create_task(pump(up, client))]
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in tasks:
                t.cancel()
    except aiohttp.ClientError as exc:
        print(f"[ws] upstream unavailable: {exc}", flush=True)
    await client.close()
    return client


def make_app(upstream: str) -> web.Application:
    app = web.Application(client_max_size=0)

    async def on_startup(app: web.Application) -> None:
        app["session"] = aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False))

    async def on_cleanup(app: web.Application) -> None:
        await app["session"].close()

    async def handle(request: web.Request) -> web.StreamResponse:
        session = request.app["session"]
        if request.path == "/":
            raise web.HTTPFound("/teleop")
        if request.headers.get("Upgrade", "").lower() == "websocket":
            return await relay_ws(request, upstream, session)
        local = vendored(request.path)
        if local is not None:
            return web.FileResponse(local, headers={"Content-Type": "application/javascript"})
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}
        try:
            async with session.request(
                request.method, upstream + request.path_qs, headers=headers,
                data=await request.read(), allow_redirects=False,
            ) as resp:
                body = await resp.read()
                if request.path.endswith("teleop.js"):
                    body = body.replace(b"https://esm.sh/", b"/")
                out = {k: v for k, v in resp.headers.items() if k.lower() not in HOP_BY_HOP}
                out["Cache-Control"] = "no-store"
                return web.Response(status=resp.status, body=body, headers=out)
        except aiohttp.ClientConnectorError:
            return web.Response(
                status=502, content_type="text/plain",
                text=f"dimOS teleop server is not running ({upstream}). Start the collection blueprint, then reload.",
            )

    app.router.add_route("*", "/{tail:.*}", handle)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


async def keep_adb_reverse(port: int) -> None:
    adb = shutil.which("adb") or "/opt/homebrew/bin/adb"
    last = None
    while True:
        proc = await asyncio.create_subprocess_exec(adb, "devices", stdout=asyncio.subprocess.PIPE)
        out = (await proc.communicate())[0].decode()
        states = [line.split()[1] for line in out.splitlines()[1:] if line.strip()]
        if "device" in states:
            proc = await asyncio.create_subprocess_exec(adb, "reverse", "--list", stdout=asyncio.subprocess.PIPE)
            mappings = (await proc.communicate())[0].decode()
            if f"tcp:{port} tcp:{port}" not in mappings:
                proc = await asyncio.create_subprocess_exec(adb, "reverse", f"tcp:{port}", f"tcp:{port}")
                await proc.wait()
            state = f"Quest connected: open http://localhost:{port}/teleop in the Quest browser"
        elif "unauthorized" in states:
            state = "Quest unauthorized: put on the headset and accept 'Allow USB debugging' (tick 'Always allow')"
        else:
            state = "Quest not detected over USB (check cable / Developer Mode)"
        if state != last:
            print(f"[adb] {state}", flush=True)
            last = state
        await asyncio.sleep(3)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--upstream", default="https://127.0.0.1:8443")
    parser.add_argument("--no-adb", action="store_true", help="don't manage adb reverse (testing)")
    args = parser.parse_args()
    runner = web.AppRunner(make_app(args.upstream))
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", args.port).start()
    print(f"[proxy] http://127.0.0.1:{args.port} -> {args.upstream} (esm.sh served from {VENDOR})", flush=True)
    if args.no_adb:
        await asyncio.Event().wait()
    await keep_adb_reverse(args.port)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
