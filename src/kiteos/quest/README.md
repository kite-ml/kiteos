# KiteOS Quest teleop proxy

Lets a USB-tethered Meta Quest drive the arms through the dimOS WebXR teleop server, with
the page branded as KiteOS and no internet needed on the headset.

```
Quest browser                      robot computer
http://localhost:8080/teleop ──USB (adb reverse)──▶ KiteOS proxy 127.0.0.1:8080
                                                      │  HTTP + WebSocket relay
                                                      ▼
                                              dimOS WebXR server https://127.0.0.1:8443
```

- The headset opens `http://localhost:8080/teleop`. Browsers treat `http://localhost` as a
  secure context, so WebXR works without accepting dimOS's self-signed certificate. The
  proxy speaks HTTPS to dimOS with certificate verification off (it is local and self-signed).
- `adb reverse tcp:8080 tcp:8080` maps the headset's localhost to this machine. The proxy
  re-applies it every 3 s, so sleep and replug recover by themselves.
- The teleop WebSocket (`/ws`) is relayed both ways, including dimOS's close reason (for
  example "A WebXR control client is already connected").

## Run

```bash
kiteos quest proxy                         # the usual way
python -m kiteos.quest.proxy --no-adb      # same module, no adb management
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--host` | `127.0.0.1` | Listen address. Keep it on loopback; the teleop page drives real arms. |
| `--port` | `8080` | Listen port, and the port reversed onto the headset. |
| `--upstream` | `https://127.0.0.1:8443` | dimOS WebXR server. |
| `--no-adb` | off | Do not manage `adb reverse` (status then reports `adb: disabled`). |
| `--adb-serial` | none | Only manage this headset (`adb -s SERIAL`). Without it, every ready device gets the mapping. |
| `--adb-path` | auto | adb binary. Auto-discovery: `PATH`, `$ANDROID_HOME`/`$ANDROID_SDK_ROOT`, then `/opt/homebrew/bin`, `/usr/local/bin`, `~/Library/Android/sdk/platform-tools` on macOS, or `/usr/bin`, `/usr/local/bin`, `~/Android/Sdk/platform-tools` on Linux. |
| `-v` | off | Debug logging and access log. |

SIGINT or SIGTERM stops it cleanly; open relays are closed with 1001 first.

## Headset setup

1. Enable Developer Mode for the headset (Meta Horizon app) and connect it by USB-C.
2. Install adb: `brew install android-platform-tools` (macOS) or `sudo apt install adb` (Linux).
   On Linux also install the udev rule for vendor `2833` from `deploy/udev` and replug.
3. Start the proxy, put the headset on and accept **Allow USB debugging** (tick *Always allow*).
4. Start teleop (`kiteos teleop` or `kiteos collect start`), then open
   `http://localhost:8080/teleop` in the Quest browser.

The proxy logs each adb state change once:

| State | Meaning |
| --- | --- |
| `connected` | Headset authorised and `adb reverse` in place. |
| `unauthorized` | Accept the USB debugging prompt in the headset, or (Linux) the udev rule is missing. |
| `absent` | No headset, headset offline/asleep, adb not installed, or `adb reverse` failed (see the detail text). |
| `disabled` | Started with `--no-adb`. |

## What the proxy serves

| Path | Behaviour |
| --- | --- |
| `/` | Redirects to `/teleop`. |
| `/teleop` | dimOS page with KiteOS branding (see below). |
| `/static/teleop.js` | dimOS script with `https://esm.sh/` imports rewritten to the proxy's origin. |
| esm.sh module paths | Served from `vendor/esm/` (offline). |
| `/ws` (any Upgrade request) | WebSocket relay to dimOS. |
| `/kiteos/*` | `static/` files: `brand.css`, `logo.svg`, `status.js`. |
| `/kiteos/api/status` | `{"proxy": "ok", "upstream": "up"\|"down", "adb": "connected"\|"unauthorized"\|"absent"\|"disabled", "adb_detail": "...", "version": "0.1.0"}` |
| anything else | Proxied to dimOS. Hop-by-hop headers are stripped, bodies are fetched uncompressed and every response gets `Cache-Control: no-store`. Upstream redirects are rewritten to the proxy's origin. |

If dimOS is not running, HTML requests get a KiteOS "Teleop server is not running" page that
reloads every 3 s; other requests get a plain-text 502.

## Branding

dimOS is never modified. For `GET /teleop` returning `text/html`, the proxy:

- replaces `<title>` and `<h1>` "DimOS WebXR Teleop" with "KiteOS Teleop";
- adds `/kiteos/brand.css` and `/kiteos/status.js` just before `</head>`, after `teleop.css`,
  so the KiteOS colours win the cascade;
- adds a header after `<body>` with the logo and a status pill (`#kiteos-status`), which
  `status.js` updates from `/kiteos/api/status`. It never touches dimOS's own elements.

Every anchor must occur exactly once. If dimOS changes the page and any anchor is missing or
repeated, the stock page is served unmodified and a warning is logged: branding must never
break teleop. The test `test_branding_matches_pinned_dimos_page` runs the branding against the
installed dimOS page, so drift shows up in CI rather than on the headset.

Branding only reaches headsets that go through the proxy. Opening `https://<host>:8443/teleop`
directly shows the stock dimOS page. The in-headset HUD (drawn on a canvas by `teleop.js`)
keeps dimOS's colours.

## Vendored modules

`teleop.js` imports `https://esm.sh/jsr/@dimos/msgs@<version>`. `vendor/esm/` holds that
module graph, mirrored by `legacy/tools/vendor_esm.py` (keyed by URL path; bare module URLs
are stored as `<path>/__entry__.mjs`). The version is pinned by dimOS's `teleop.js`. When
dimOS bumps it, the proxy logs "teleop.js imports ..., which is not vendored" and the test
above fails. To refresh the modules, run `vendor_esm.py <new esm.sh URL>` on a machine with
internet access and copy its `vendor/esm` tree here.

## Security

The proxy listens on 127.0.0.1 only, and the headset reaches it over USB. Do not pass
`--host 0.0.0.0`: anyone on the network could then open the teleop page and drive the arms.
The WebXR server has no authentication. dimOS binds it to 0.0.0.0; the KiteOS blueprints bind it
to 127.0.0.1 (`KiteArmTeleopModule`). Block 8443 from the LAN with a firewall anyway, for stacks
started outside KiteOS.

## Python API

```python
from kiteos.quest.proxy import AdbReverseKeeper, create_app, main, serve

app = create_app("https://127.0.0.1:8443", vendor_dir, static_dir)   # aiohttp Application
main(["--port", "8080", "--no-adb"])                                   # returns an exit code
```
