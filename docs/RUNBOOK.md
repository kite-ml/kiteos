# KiteOS runbook: dual OpenYAM station

The morning checklist for macOS and Linux. Commands assume the default layout
(`~/teleop_stack/{kiteos,dimos,policies}`) and a configured `config/station.yaml`.
Details: fence in [FENCE.md](FENCE.md), install in [README](../README.md#quickstart).

> **The one rule:** stopping KiteOS (including Ctrl-C in the stack terminal, a crash, or the
> e-stop) disables the motors and **the arms fall**. Park them away from the glass and have
> someone support them *before* anything stops.

Terminals used below: **T1** runs the stack (foreground), **T2** the controls, **T3** the Quest proxy.

## 0. The night before (once per machine)

- [ ] `~/teleop_stack/kiteos/deploy/install.sh` (Linux: add `--with-udev`, then log out and back in).
      Preview with `--dry-run`.
- [ ] Hugging Face token with write access to the dataset org: `~/teleop_stack/dimos/.venv/bin/hf auth login`.
- [ ] Policy checkpoint at the station's `policy.checkpoint` (validated: ACT `act-close-box-step30000`).
- [ ] Site values in `config/station.yaml` (or `config/station.local.yaml`, which is git-ignored and
      merged on top): `station`, `task`, `hub.repo_id`, Linux camera paths.
- [ ] The Quest teleop server (8443) has no authentication. KiteOS binds it to 127.0.0.1 (the Quest
      reaches it over USB through `kiteos quest proxy`). Block it from the LAN anyway, as defence
      in depth. On Linux: `sudo ufw deny 8443/tcp`. On macOS: System Settings > Network > Firewall
      on, plus "Enable stealth mode". Check from another machine that `https://<station-ip>:8443`
      does not answer.
- [ ] `kiteos doctor` passes.

## 1. Power-up order

1. Arms resting in the park pose (low, folded, away from the glass) and supported. E-stop within reach.
2. Station computer on, logged in as the station user.
3. Plug in USB: both CAN adapters, the camera hub, the Quest cable.
4. Arm power on. The motors stay disabled until a `--real` run starts.
5. Linux only: `kiteos can setup` (after **every** plug-in or reboot; asks for sudo).
6. `kiteos doctor`.

## 2. `kiteos doctor`

Checks the environment, the dimOS pin, KiteOS blueprints (`dimos list`), the recorder binary, the URDF,
the station and fence files, an FK self-test, adb, CAN, cameras and the HF login. It exits non-zero on
blocking failures. Fix every blocking item before any `--real` run.

- While any dimOS stack runs on the host, doctor does not list the CAN adapters (listing opens their
  USB descriptors); it prints a WARN instead. Run it again after `kiteos stop`.
- Camera placeholders (`REPLACE`) are a FAIL: `collect` and `rollout` refuse to start with them.
  Teleop alone works with `--no-cameras`.
- On macOS the built-in and iPhone (Continuity) cameras are not counted as station cameras.

## 3. Cameras

- `kiteos cameras list` shows the station mapping and what the host sees. It never opens the cameras.
  On macOS the index shown is OpenCV's probable order (sorted by unique ID). It includes the
  built-in and iPhone cameras, and it is only a hint: the cover test decides.
- `kiteos cameras check` opens a live view with a sharpness meter per role (`1/2/3` zoom, `q` quits).
- **Cover test** (daily, and whenever a cable or the hub moves): run `kiteos cameras check` and cover each
  lens in turn. The matching role (`left_wrist`, `right_wrist`, `top`) must go dark. If not, fix
  `cameras.<role>` in the station file.
- **macOS:** run from a terminal app that has Camera permission (System Settings > Privacy & Security >
  Camera, e.g. Terminal or iTerm). Without it, captures fail or stay black with no prompt. Indices follow
  the USB port, so redo the cover test if the hub moves.
- **Linux:** use stable paths, never integer indices (each camera has two `/dev/video*` nodes).
  `ls -l /dev/v4l/by-path/` lists the `...-video-index0` entries; put those in `cameras.<role>.linux`.
  by-path follows the physical port, so do not move cables after mapping. If frames drop or you see
  "No space left on device", spread the cameras across root ports (`lsusb -t`).

## 4. CAN

- **macOS:** nothing to configure; adapters are addressed by USB serial (`can.<side>.darwin`).
  `kiteos can status` must show both.
- **Linux:** the udev rule names the adapters `can_left` / `can_right` by USB serial
  (`deploy/udev/80-kiteos-can.rules`). After every plug-in or reboot, run `kiteos can setup`
  (runs `sudo ip link` at 1 Mbit/s). `kiteos can status` must show both UP. `can setup` leaves an
  interface that is already UP at 1 Mbit/s alone. It refuses while a dimOS stack runs, because
  taking a link down cuts the motors mid-motion (`--force` overrides).
- New site or swapped adapters: on the first `--real` run, move one arm gently and confirm the expected
  arm follows.

## 5. Quest over USB

1. First time with a headset: enable Developer Mode for it (Meta Horizon app > Devices > Headset settings).
2. Plug the Quest into the station. In the headset, accept **Allow USB debugging** (tick "Always allow").
3. T3: `kiteos quest proxy` (keep it running; it keeps `adb reverse tcp:8080` applied and reports the
   headset state). With a MOCK stack running on its own ports, the proxy relays to that stack.
4. Start the stack (teleop, collect or rollout) in T1, then open **http://localhost:8080/teleop** in the
   headset browser. Before the stack is up, the page reports that the teleop server is not running.
5. Hold **both grips** to engage (dead man's switch); release to stop. Triggers close the grippers.

## 6. Fence at the customer site

Do this once per site, and again after anything moves (arms, table, window, cameras). Someone keeps a
hand on the e-stop throughout. Full procedure and background: [FENCE.md](FENCE.md).

1. `kiteos fence validate` and `kiteos fence show`. Set `mounts` to the measured base positions, or
   teach per-arm zones (below). Measure the wrist cameras and replace the two unmeasured camera
   spheres in `config/fence.yaml` (keep the sphere on the camera's side).
2. **Prepare to teach.** In `config/fence.yaml`, set `max_joint_speed_rad_s: 0.5` and leave the glass
   zone out (or `enabled: false`). While an arm has no zone, `--real` needs `allow_no_zones: true` in
   the file (the shipped lab file has it). Close the grippers, because the fingertip point is defined
   for closed fingers. T1: `kiteos teleop --real`. KiteOS warns loudly that no zone is enabled.
3. **Teach.** T2: `kiteos fence teach --arms left`, then `kiteos fence teach --arms right`. Each run
   writes its own zone (`glass_window_left` / `glass_window_right`, in that arm's own frame), so the
   base spacing does not matter. Touch the glass lightly with the fingertip at **at least 4 points,
   5 or more is better**, spread 20–30 cm apart, holding still while each is recorded. If the fit is
   rejected (RMS above 1 cm, or too uncertain for the standoff), re-touch rather than forcing it. Keep
   the standoff (about 2 cm). After the second arm, teach switches `allow_no_zones` off.
4. **Check it is live.** Teach hot-reloads the running stack. If the fingertip was still touching
   the glass, that arm is now inside its zone: it can only back straight out, and its gripper
   cannot open until it is clear. `kiteos fence check` must show both zones enabled. To be sure,
   park and support the arms, `kiteos stop`, then start `kiteos teleop --real` again.
5. **Offline check.** Run `kiteos fence test`, then `kiteos fence test --dataset <prepared dataset>`.
   Normal task motion must not be held. If it is, the zone is misplaced: fix it first.
6. **Slow approach test**, still at the reduced speed limit, one arm at a time. If possible, do it
   first against a cardboard panel standing where the glass is. Test with the gripper closed and
   then open, with the wrist camera facing the glass, and with the elbow leading. Expected: the arm
   stops and holds at least `margin_m` plus the standoff (about 7 cm) short of the glass, and moving
   away works at once. Near the glass, opening the gripper may be refused: that is the fence. If
   anything touches, hit the e-stop and fix the fence before any other run.
7. Restore `max_joint_speed_rad_s: 2.5` and restart. The first rollout at the site runs at
   `--speed 0.3`, with the same e-stop watch.

## 7. Teleop

- T1: `kiteos teleop --real` (mock arms without `--real`; `--no-cameras` to skip cameras).
  The `--real` prompt shows the station and fence files and the zones, and asks you to type the
  station name.
- T3 + headset as in section 5.
- **Only one stack per host.** Every start command refuses while another dimOS stack runs on this
  machine. That includes stacks started outside KiteOS, such as `kite_rollout`. The check uses
  dimOS's run registry, the KiteOS run files, a process scan, the 8443/8095 listeners and, for
  `--real`, a short passive listen on the bus.
- **Mock runs (no `--real`) are isolated.** They get a private bus and their own WebXR/viser ports,
  and the coordinator refuses real arms. So a practice run can never reach a real stack.
  `--station-ports` keeps 8443/8095 (Quest practice). `--allow-concurrent` starts a mock run next to
  another stack (it still opens the cameras for collect/rollout).
- KiteOS refuses to start when the environment or `~/teleop_stack/dimos/.env` sets
  `CONTROLCOORDINATOR__*`, `G__*` or `TRANSPORTS__*`, and it ignores `~/.config/dimos/config`.

## 8. Collection

1. T1: `kiteos collect start --real` (optional `--session NAME`, `--task "..."`). Each session records
   into the next free `recordings/<prefix>-NNN`. Without `--real` (mock arms) it records into
   `mock-<prefix>-NNN`, which is never uploaded.
2. T2: `kiteos collect controls`. Keys: `Space` start/save, `R` restart take, `D` discard,
   `U` delete last saved, `P` pause and upload, `Q` detach. On the Quest: `B` start/save, `Y` discard.
   `U` and `P` act on exactly the session the running stack records. If the controls cannot tell which
   session that is (a stack not started with `kiteos collect start`), `U` is disabled and `P` does
   not upload.
3. The recorder writes about 12 GB per hour even when idle. Stop the stack when you are not collecting.
4. A recording is only readable after the stack has stopped (see section 11).

## 9. Upload

1. Stop the stack first (section 11). `P` in the collect controls saves, stops (after you type `stop`)
   and uploads in one go.
2. `kiteos upload` converts every finished session and pushes it to `hub.repo_id`, private by default
   (`--session`, `--repo-id`, `--public`, `--dry-run` available).
   - A push replaces the Hub dataset with the merged sessions of one prefix. So `--session` with a
     prefix other than the station's needs its own `--repo-id`.
   - `mock-*` sessions are refused (`--include-mock` overrides).

## 10. Rollout

1. T1: `kiteos rollout start --real --speed 0.3` (checkpoint and device come from the station file;
   override with `--checkpoint`, `--device`; `auto` picks cuda, then mps, then cpu). `--record`
   records the rollout in the collection format into `rollout-<prefix>-NNN` (Quest `B`/`Y` as in
   collection).
   - The chunk guards come from the station's `policy.max_start_jump` (0.3 rad) and
     `policy.max_joint_speed` (3.0 rad/s). `--max-start-jump` and `--max-joint-speed` override
     them.
   - What the arms and grippers actually execute is capped at 1.0 rad/s (or opening/s) per joint by
     the trajectory task, and then fenced.
2. T2: `kiteos rollout controls` (`Space` starts/stops the policy, `Q` detaches).
3. Raise `--speed` only after clean runs. Squeezing either Quest grip stops the policy; hold both to
   take over by teleop.
4. Do not use `pi0fast-close-box-step5000`: it is known broken.

## 11. Safe shutdown

1. Stop the policy or teleop (`Space` in the controls, release the grips).
2. Teleop both arms to the park pose, low and away from the glass.
3. **Support both arms** (a person, or rest them on blocks).
4. `kiteos stop` and type `stop` when prompted. It asks for confirmation for every run that is not
   proven to be mock arms. It sends SIGTERM only, and waits up to 300 s for the stack to exit:
   - dimOS gives the recorder up to 100 s to finalize its file (`--recorder.shutdown-timeout`), then
     kills it;
   - after stopping the modules, dimOS kills leftover worker processes 2 s after terminating them.

   Never `kill -9`, never close T1 early. With several stacks running, `kiteos stop` lists them and
   needs `--pid`. A stack KiteOS did not start (and dimOS did not register) is reported, never
   guessed at: stop it in its own terminal.
5. Arm power off. Ctrl-C the Quest proxy in T3. Unplug the Quest.
6. `kiteos upload` if you collected.

## 12. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `kiteos` not found, or `dimos list` shows no `kiteos.*` blueprints | A plain `uv sync` in dimOS removed the editable KiteOS install | `uv pip install --python ~/teleop_stack/dimos/.venv/bin/python --no-deps -e ~/teleop_stack/kiteos`, or re-run `install.sh`. In dimOS, use `uv run --no-sync` / `uv sync --inexact` only |
| `doctor`: dimOS not at the pin | Checkout moved or has local edits | `git -C ~/teleop_stack/dimos checkout dual-collection` (the installer never resets it for you) |
| macOS: cameras black or fail to open | Terminal lacks Camera permission | Run from a terminal app with Camera permission (section 3) |
| Wrong image for a camera role | Cable or hub moved | Cover test, fix `cameras.<role>` |
| Linux: dropped frames or "No space left on device" | USB bandwidth (uncompressed YUYV) | Spread the cameras across root ports |
| Camera errors right after a restart | dimOS webcam thread did not recover | Wait about 5 s between stop and start |
| Linux: `can_left` missing, `can0`/`can1` present | udev rule missing or serial mismatch | `install.sh --with-udev`, replug. Check the serial: `udevadm info -a -p /sys/class/net/can0 \| grep serial` |
| Linux: CAN "Network is down" at start | Interfaces not set up since the last plug-in | `kiteos can setup` |
| Linux: `gs_usb` module missing | Minimal or cloud kernel | `sudo apt install linux-modules-extra-$(uname -r)` |
| Quest page does not load | Headset unauthorized, or proxy not running | Accept USB debugging in the headset; check `adb devices` and the T3 output |
| Quest page says the teleop server is not running | Stack not started yet | Start teleop, collect or rollout in T1, then reload |
| Quest connected but the arms do not move | Grips not both held, or the fence is holding | Hold both grips; `kiteos fence check` |
| Arm freezes near the glass or an obstacle | Fence hold (expected) | Move away (escape motion is allowed); check `kiteos fence check` |
| `--real` refuses to start | Fence file missing, empty or invalid; `monitor` mode; an arm without a zone (and no `allow_no_zones`); speed limit above 3 rad/s; or an arm already inside a zone | `kiteos fence validate`. Arm inside a zone: power off, reposition by hand, retry |
| A start command says another dimOS stack is running | Another stack, a leftover process, or something on 8443/8095 | `kiteos stop` (or stop it in its own terminal). Mock practice next to it: `--allow-concurrent` |
| Start refused because of `CONTROLCOORDINATOR__...` | A setting in the shell or in `~/teleop_stack/dimos/.env` would bypass the station file | Remove it; the station file and flags are the only source |
| Gripper does not open near the glass | The fence refuses an opening that would put the open fingers within the margin | Move the arm back first; closing always works |
| Arm does not move again after a hold, although the target is now clear | Stale target: the fence keeps refusing the exact request it refused | Move the controller (any new target is checked again) |
| Linux: sudo prompt about locked memory on first run | Memlock limit not active yet | `install.sh --with-udev`, then log out and back in (declining only slows local transport) |
| `kiteos stop` takes minutes | Recorder finalizing a large file (dimOS allows it 100 s) | Wait. KiteOS never kills; re-run `kiteos stop` if it times out |
| Upload skips a session or reports "opcode 255" | Recording still open | `kiteos stop` first, then `kiteos upload` |
| Collection fails at start mentioning `nix` | Recorder binary missing | Re-run `install.sh` without `--no-recorder` |
| First rollout or upload stalls for minutes | LeRobot environment being built | One-off. Pre-warm with `install.sh` (without `--no-prewarm`) |
| macOS: `dimos run --daemon` crashes | Known pyarrow fork issue on macOS | KiteOS always runs in the foreground; keep T1 open |
