# KiteOS

KiteOS is Kite's robot runtime for teleoperation, data collection and policy rollout on the
dual OpenYAM station. It is one Python package (`kiteos`) that installs into a pinned base runtime
(robot drivers, control loop, recorder and teleop server), which `deploy/install.sh` sets up next to
it. KiteOS never modifies the base runtime; it extends it through entry points, subclasses and
command-line flags.

What KiteOS provides:

- **KiteOS setups.** Four setups (`kiteos.dual-openyam-teleop`, `-collection`, `-rollout` and
  `-rollout-recorded`)
  and a `kiteos` CLI that fills every flag from one station file (`config/station.yaml`): CAN
  selectors and camera devices per OS, task, dataset repo, policy.
- **Safety fence.** A keep-out fence (for example the customer's glass window) enforced in the
  control coordinator right before the motors, so it covers Quest teleop, policy trajectories, the
  grippers and every other command source. It includes joint limits, a speed limit, E-stop
  latching and a teach workflow. See [docs/FENCE.md](docs/FENCE.md).
- **One stack per host, mock runs isolated.** Start commands refuse while another robot stack runs.
  Mock runs get a private bus and their own ports, and their coordinator refuses real arms.
- **Branded Quest teleop.** A KiteOS page served to a USB-tethered Meta Quest through a local proxy
  that keeps `adb reverse` applied. It works offline, with no Wi-Fi.
- **Collection, upload and rollout tooling.** Episode controls, graceful stop (SIGTERM only; the
  recorder gets 100 s to finalize), private-by-default Hugging Face upload, and a dual-arm, 3-camera
  policy runtime with start-jump and speed guards.
- **Deploy.** A single idempotent installer for macOS and Linux, udev rules for stable CAN names, a
  pinned base runtime, `kiteos doctor`, and a morning [runbook](docs/RUNBOOK.md).

## Quickstart

```bash
# 1. Get KiteOS
git clone <kiteos repo> ~/teleop_stack/kiteos   # or copy the directory

# 2. Install (preview first; it is safe to re-run)
~/teleop_stack/kiteos/deploy/install.sh --dry-run
~/teleop_stack/kiteos/deploy/install.sh              # Linux: add --with-udev

# 3. Check the station, then try it with mock arms (no hardware)
kiteos doctor
kiteos teleop            # add --real only after the RUNBOOK checklist
```

`install.sh` installs system packages (Homebrew or apt), uv, and Rust (only for the recorder).
It sets up the base runtime at the pinned commit, builds the collection recorder and installs
KiteOS into the base runtime's virtualenv. Then it fetches the robot model and pre-builds the
policy environment. Run `deploy/install.sh --help` for the flags. It never touches hardware and
never overwrites an existing station config.

Daily use (details in [docs/RUNBOOK.md](docs/RUNBOOK.md)):

| Task | Command |
|---|---|
| Health check | `kiteos doctor` |
| Cameras | `kiteos cameras list`, `kiteos cameras check` |
| CAN (Linux: after every plug-in) | `kiteos can setup`, `kiteos can status` |
| Quest over USB | `kiteos quest proxy`, then open `http://localhost:8080/teleop` in the headset |
| Teleop | `kiteos teleop [--real]` |
| Collect | `kiteos collect start [--real]` + `kiteos collect controls` |
| Upload | `kiteos upload` |
| Rollout | `kiteos rollout start [--real] [--speed 0.3] [--record]` + `kiteos rollout controls` |
| Fence | `kiteos fence show\|validate\|check\|teach\|test` |
| Stop (arms fall: support them first) | `kiteos stop [--pid PID]` |

Without `--real`, no CAN ports are passed and the arms are simulated. The run gets a private bus
and free WebXR/viser ports, and the coordinator refuses any real adapter
(`--controlcoordinator.expect-mock true`). `--real` requires a fence that passes the real-hardware
checks, and a typed confirmation. Every start command refuses while another robot stack runs on
the host. `kiteos stop` asks for a typed confirmation unless the run is proven to be mock.

> **Keep the KiteOS install:** a plain `uv sync` inside the base runtime checkout is exact and
> uninstalls the editable `kiteos` package (and its `kiteos.*` setups). KiteOS always uses
> `uv run --no-sync`. To restore it, re-run `deploy/install.sh`; `kiteos doctor` also prints the
> exact reinstall command.

## Base runtime

- **Pinned, never modified.** KiteOS runs against one pinned commit of the base runtime, recorded
  in the `PIN` file under `deploy/`. That commit ships with KiteOS as a small git bundle next to the
  `PIN` file. The installer creates the checkout, applies the bundle and verifies `HEAD`. An
  existing checkout is only verified, never reset, and `kiteos doctor` reports drift.
- **KiteOS lives in the base runtime's virtualenv.** The installer registers the `kiteos` command
  and the KiteOS setups there. The policy environment sees KiteOS through `PYTHONPATH=<repo>/src`,
  which the CLI sets, so `import kiteos` stays dependency-free.
- **Keep `.git` in the base runtime checkout.** Without it, the runtime downloads unpinned code
  when it first needs its project files, and the policy environment would run that code instead of
  the pinned commit.
- **Updating the pin:** check out and test the new commit, regenerate the bundle, update `PIN`,
  re-vendor `src/kiteos/data/dual_openyam.urdf` if it changed, and run the tests.

## Repository layout

```
pyproject.toml             package "kiteos": console script + setup entry points
CONTRACTS.md               internal build contracts (APIs, names)
README.md  NOTICE          this file, attributions
src/kiteos/
  __init__.py  brand.py    version and brand constants (import-light)
  config.py                StationConfig (config/station.yaml)
  safety/                  kinematics, covering spheres, zones, fence checker, teach helpers,
                           FencedWholeBodyAdapter + KiteDualOpenYamCoordinator (control-loop side)
  blueprints/              dual_openyam_teleop / _collection / _rollout (+ _rollout_recorded),
                           KiteArmTeleopModule (WebXR server on 127.0.0.1)
  rollout/                 DualArmPolicyModule + runtime (runs in the policy environment)
  quest/                   branded Quest proxy, static assets, vendored ES modules
  cli/                     the `kiteos` Typer app (runs.py: stack detection, run-state files, private bus)
  data/dual_openyam.urdf   vendored robot model used by the fence
config/                    station.yaml, fence.yaml and their *.example.yaml templates
deploy/
  install.sh  lib/         installer and its helpers
  udev/                    80-kiteos-can.rules (can_left/can_right), 70-kiteos-quest.rules (uaccess)
  limits/                  99-kiteos-memlock.conf (Zenoh shared memory)
                           plus the pinned base runtime (PIN file and git bundle)
docs/                      RUNBOOK.md (daily checklist), FENCE.md (fence reference)
tests/                     pytest suite (mock hardware only)
legacy/  kite_rollout/     the pre-KiteOS tools and rollout prototype, kept unchanged for reference
```

## Development

```bash
cd ~/teleop_stack/kiteos
PYTHONPATH=src <python> -m pytest -q
```

`<python>` is the interpreter KiteOS is installed in; `kiteos doctor` shows it on its `python`
line.

Tests never touch real hardware. Some tests need the base runtime's virtualenv. Tests marked
`mock_e2e` launch the runtime with mock arms (`tests/e2e/test_mock_fence_e2e.py`, about 40 s; skip
it with `-m "not mock_e2e"`). `tests/conftest.py` puts the whole suite on a private, per-process
LCM/Zenoh bus, so a test can never reach a stack running on the default bus. The e2e test also
refuses to run if that private bus is not in effect. Unit tests never scan the host for stacks or
listen on a bus (the conftest stubs those probes), and KiteOS run-state files go to a temporary
directory. Every finding from the pre-release safety review has a regression test in the main
suite (for example the gripper-toggle ratchet replay in `tests/fixtures/`).

## License

Copyright 2026 Kite ML. Third-party components and their licenses are listed in [NOTICE](NOTICE).
