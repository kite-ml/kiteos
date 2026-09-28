"""Mirror an esm.sh module and everything it imports, for a Quest with no internet.

The dimOS teleop page imports https://esm.sh/jsr/@dimos/msgs@0.1.4. This fetches that
module graph once (from a Mac that has internet) into vendor/esm/, keyed by URL path,
so quest_proxy.py can serve it from the same origin as the teleop page.

Usage: python vendor_esm.py [entry_url]
"""

import re
import sys
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

ENTRY = sys.argv[1] if len(sys.argv) > 1 else "https://esm.sh/jsr/@dimos/msgs@0.1.4"
OUT = Path(__file__).resolve().parent / "vendor" / "esm"
# esm.sh tailors the build to the browser; ask for what the Quest Browser (Chromium) gets.
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) OculusBrowser/35.0 Chrome/126.0 Safari/537.36"
IMPORT_RE = re.compile(r"""(?:\bfrom\s*|\bimport\s*\(?\s*)["']([^"']+)["']""")


def local_path(url: str) -> Path:
    path = urlparse(url).path.lstrip("/")
    return OUT / (path if path.endswith((".mjs", ".js")) else path + "/__entry__.mjs")


def main() -> None:
    queue, seen = [ENTRY], set()
    while queue:
        url = queue.pop()
        if url in seen:
            continue
        seen.add(url)
        with urlopen(Request(url, headers={"User-Agent": UA}), timeout=30) as resp:
            body = resp.read().decode()
        for spec in IMPORT_RE.findall(body):
            if spec.startswith(("/", "https://esm.sh/")):
                queue.append(urljoin("https://esm.sh/", spec))
            elif spec.startswith("."):
                queue.append(urljoin(url, spec))
            else:
                sys.exit(f"unsupported import {spec!r} in {url}")
        body = body.replace("https://esm.sh/", "/")  # keep every import on the proxy's origin
        dest = local_path(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(body)
        print(f"{urlparse(url).path}  ({len(body)} bytes)")
    print(f"mirrored {len(seen)} modules into {OUT}")


if __name__ == "__main__":
    main()
