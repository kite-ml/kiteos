"""Live view of the three collection cameras with a sharpness meter for focusing.

Keys: 1/2/3 pick the camera shown zoomed (left_wrist/right_wrist/top),
      r resets the "best" marks, q or Esc quits.
Sharpness = variance of the Laplacian over the centre box; turn the lens to maximise it.
"""

import threading
import time

import cv2
import numpy as np

CAMS = [("left_wrist", 0), ("right_wrist", 2), ("top", 1)]  # (role, OpenCV index)
W, H = 640, 480
ROI = (W // 4, H // 4, 3 * W // 4, 3 * H // 4)  # centre box: x0, y0, x1, y1
WIN = "Camera focus - 1/2/3 zoom, r reset, q quit"


class Reader(threading.Thread):
    def __init__(self, index: int) -> None:
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.frame = None
        self.lock = threading.Lock()
        self.running = True

    def run(self) -> None:
        while self.running:
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.01)

    def latest(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()


def sharpness(frame) -> float:
    x0, y0, x1, y1 = ROI
    gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def label(img, lines, color=(255, 255, 255)) -> None:
    cv2.rectangle(img, (0, 0), (img.shape[1], 30 * len(lines) + 10), (0, 0, 0), -1)
    for n, text in enumerate(lines):
        cv2.putText(img, text, (10, 30 + 30 * n), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


readers = {role: Reader(index) for role, index in CAMS}
for r in readers.values():
    r.start()

smooth = {role: 0.0 for role, _ in CAMS}
best = {role: 0.0 for role, _ in CAMS}
zoom_role = "left_wrist"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, 1280, 960)
try:
    cv2.setWindowProperty(WIN, cv2.WND_PROP_TOPMOST, 1)
except cv2.error:
    pass

last_print = 0.0
while True:
    panels = {}
    for role, index in CAMS:
        frame = readers[role].latest()
        if frame is None:
            view = np.zeros((H, W, 3), np.uint8)
            label(view, [f"{role} (index {index})", "no frames"], (0, 0, 255))
            panels[role] = (view, None)
            continue
        s = sharpness(frame)
        smooth[role] = s if smooth[role] == 0 else 0.7 * smooth[role] + 0.3 * s
        best[role] = max(best[role], smooth[role])
        view = frame.copy()
        x0, y0, x1, y1 = ROI
        cv2.rectangle(view, (x0, y0), (x1, y1), (0, 255, 255), 1)
        ratio = smooth[role] / best[role] if best[role] else 0
        cv2.rectangle(view, (0, H - 16), (int(W * ratio), H), (0, 200, 0), -1)
        label(view, [f"{role} (index {index})", f"sharpness {smooth[role]:.0f}   best {best[role]:.0f}"])
        panels[role] = (view, frame)

    zoom_raw = panels[zoom_role][1]
    if zoom_raw is not None:
        x0, y0, x1, y1 = ROI
        zoom = cv2.resize(zoom_raw[y0:y1, x0:x1], (W, H), interpolation=cv2.INTER_NEAREST)
        label(zoom, [f"ZOOM 2x: {zoom_role}", f"sharpness {smooth[zoom_role]:.0f}   best {best[zoom_role]:.0f}"], (0, 255, 255))
    else:
        zoom = np.zeros((H, W, 3), np.uint8)

    grid = np.vstack([
        np.hstack([panels["left_wrist"][0], panels["right_wrist"][0]]),
        np.hstack([panels["top"][0], zoom]),
    ])
    cv2.imshow(WIN, grid)

    now = time.time()
    if now - last_print > 2:
        print("sharpness  " + "  ".join(f"{role}={smooth[role]:.0f} (best {best[role]:.0f})" for role, _ in CAMS), flush=True)
        last_print = now

    key = cv2.waitKey(30) & 0xFF
    if key in (ord("q"), 27):
        break
    if key == ord("r"):
        best = {role: 0.0 for role, _ in CAMS}
    if key in (ord("1"), ord("2"), ord("3")):
        zoom_role = CAMS[key - ord("1")][0]
    if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
        break

for r in readers.values():
    r.running = False
    r.join(timeout=1)
    r.cap.release()
cv2.destroyAllWindows()
print("live view closed", flush=True)
