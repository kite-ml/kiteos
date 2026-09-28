# KiteOS POC — build contracts (read first)

KiteOS is a Kite-branded extension of dimOS. It is one Python distribution (`kiteos`, src layout)
installed *into the dimOS virtualenv* (`uv pip install --no-deps -e <repo>`). It never modifies dimOS
source. dimOS is pinned to `dual-collection @ 3e7cac5dc34bb766617d30486d7d9f1b2d89d93a`
(checkout: `~/teleop_stack/dimos`). Python 3.12. Everything must work on macOS and Linux.

Hard rules
- Do not edit anything under `~/teleop_stack/dimos`. Extend via subclassing, entry points, CLI flags.
- `import kiteos` must stay dependency-free (it is also imported inside dimOS's isolated LeRobot env
  via PYTHONPATH). `kiteos/__init__.py` imports nothing heavy. `kiteos.safety.*` depends only on
  numpy + pydantic + pyyaml (no dimOS, no pinocchio) — except `kiteos.safety.coordinator` and
  `kiteos.safety.adapter`, which import dimOS.
- Code copied/adapted from dimOS keeps its Apache-2.0 header plus a line
  `# Modified by Kite ML for KiteOS.` Everything else: `# Copyright 2026 Kite ML.`
- Safety code fails closed: never raise out of the control tick; on any error hold position.
- No real hardware is touched by anything in this build. Mock hardware only.
- Tests never touch the default dimOS bus: tests/conftest.py moves the suite to a private
  LCM/Zenoh bus, stubs the CLI's host probes and points KiteOS run-state files at a temp dir.

## Layout
```
pyproject.toml                  # name "kiteos", console script "kiteos", dimos.blueprints entry points
CONTRACTS.md  README.md  NOTICE
src/kiteos/__init__.py          # __version__ = "0.1.0"; BRAND constants only
src/kiteos/brand.py             # BRAND_NAME="KiteOS", colors, ascii banner
src/kiteos/config.py            # StationConfig (see below)
src/kiteos/safety/__init__.py
src/kiteos/safety/kinematics.py # numpy FK for dual OpenYAM
src/kiteos/safety/geometry.py   # covering spheres per link
src/kiteos/safety/zones.py      # zone models + signed distance
src/kiteos/safety/fence.py      # FenceConfig, FenceChecker, FenceReport
src/kiteos/safety/adapter.py    # FencedWholeBodyAdapter (imports dimOS)
src/kiteos/safety/coordinator.py# KiteDualOpenYamCoordinator (imports dimOS)
src/kiteos/safety/teach.py      # plane fitting / teach helpers (pure numpy)
src/kiteos/blueprints/teleop.py      # dual_openyam_teleop   (Blueprint object)
src/kiteos/blueprints/collection.py  # dual_openyam_collection (Blueprint object)
src/kiteos/blueprints/rollout.py     # dual_openyam_rollout, dual_openyam_rollout_recorded (Blueprint objects)
src/kiteos/blueprints/webxr.py       # KiteArmTeleopModule: dimOS ArmTeleopModule, server on 127.0.0.1
src/kiteos/yamlio.py            # load_yaml_unique (duplicate YAML keys are errors)
src/kiteos/rollout/module.py    # DualArmPolicyModule (port of kite_rollout, fixed guards)
src/kiteos/rollout/runtime.py   # DualArmPolicyRuntime (runs in dimOS isolated LeRobot env)
src/kiteos/quest/proxy.py       # branded HTTP->HTTPS Quest proxy + adb reverse keeper
src/kiteos/quest/static/        # brand.css, logo.svg, status.js
src/kiteos/quest/vendor/esm/    # vendored esm.sh modules (copied from legacy/tools/vendor/esm)
src/kiteos/cli/__init__.py      # Typer app `kiteos` (main)
src/kiteos/cli/*.py             # doctor, cameras, can, quest, teleop, collect, rollout, fence, upload, stop
src/kiteos/cli/runs.py          # host stack detection, KiteOS run-state files, private bus, bus probe
src/kiteos/data/dual_openyam.urdf    # vendored copy of dimOS data/dual_openyam_abc_box_v2/dual_openyam.urdf
config/station.example.yaml  config/station.yaml  config/fence.example.yaml  config/fence.yaml
                                # <name>.local.yaml next to the station file used is always merged on top
deploy/install.sh  deploy/udev/*.rules  deploy/dimos/dual-collection.bundle  deploy/dimos/PIN
docs/RUNBOOK.md  docs/FENCE.md
tests/...
```

## Joint vector (from dimOS `dimos.robot.manipulators.dual_openyam.joints`)
`DUAL_OPENYAM_JOINTS = [left_joint1..6, right_joint1..6, "left_arm/gripper", "right_arm/gripper"]`
Arm joints in rad (passed straight to motors). Grippers: normalized opening 0 (closed) .. 1 (open).
kiteos.safety must not import dimOS; it redefines these name lists identically (a test asserts equality).

## Kinematics (kiteos.safety.kinematics)
World frame `dual_openyam_base`: +x forward (gripper approach at home), +y toward left arm, +z up,
z=0 at the base mounting surface. Arm bases mounted at `left_base`=(0,+0.31,0), `right_base`=(0,-0.31,0),
identity rotation (URDF default; the 0.62 m spacing comes from a sim scene, so mounts are configurable).
Per-arm chain (identical per side; T = T_parent · Trans(xyz) · Rz(y)Ry(p)Rx(r) · AxisAngle(axis, q)):
```
j1 rev base->link1  xyz(-2.49778e-07,-9.174e-13,0.068) rpy(0,1.5708,3.1415963267949) axis(-1,0,0) [-2.61799,3.14159]
j2 rev link1->link2 xyz(-0.0455,-0.0339,-0.02) rpy(0,0,-3.14159) axis(0,1,0) [0,3.66519]
j3 rev link2->link3 xyz(4.08431e-07,-0.0688,0.264) rpy(0,0,-3.14159) axis(0,1,0) [0,3.14159]
j4 rev link3->link4 xyz(-0.0600003,-0.0688,-0.244999) rpy(0,0,0) axis(0,1,0) [-1.69297,1.5708]
j5 rev link4->link5 xyz(-0.0405003,0.0338995,-0.0739989) rpy(0,0,0) axis(1,0,0) [-1.5708,1.5708]
j6 rev link5->gripper xyz(0.0404996,2.39858e-07,-0.0356) rpy(0,0,0) axis(0,0,-1) [-2.0944,2.0944]
fixed gripper->tip_left xyz(-0.0239931,0.0445119,-0.0566829); gripper->tip_right xyz(0.0239931,-0.0445119,-0.0566829)
fixed gripper->grasp_frame xyz(0,0,-0.1)
```
Use exact values parsed from the vendored URDF (`kiteos/data/dual_openyam.urdf`) at import; the table
above is the reference for tests. FK test vectors (left arm, world frame):
- q=0: link3 (-0.244,0.2751,0.1135), gripper (0.1106,0.31,0.1735), grasp (0.2106,0.31,0.1735), fingertip (0.2574,0.31,0.1735)
- q=[0,1.047,1.047,0,0,0]: gripper (0.2426,0.31,0.4021), grasp (0.3426,0.31,0.4021), fingertip (0.3894,0.31,0.4021)
- q=[0.5,1.2,0.8,-0.3,0.4,1.0]: grasp (0.311,0.4197,0.1913), fingertip (0.3486,0.4195,0.1635)
Right arm = left with y - 0.62 (with default mounts).

API:
```python
ARM_SIDES = ("left", "right")
ARM_JOINTS: dict[str, list[str]]          # {"left": [left_joint1..6], "right": [...]}
DUAL_OPENYAM_JOINTS: list[str]            # 14 names as above
GRIPPER_JOINTS: dict[str, str]            # {"left": "left_arm/gripper", "right": "right_arm/gripper"}
class ArmChain:                            # parsed per side from URDF
    joint_limits: np.ndarray               # (6,2)
    def link_poses(self, q6: np.ndarray, base_T: np.ndarray) -> dict[str, np.ndarray]  # link name (no side prefix: "base","link1".."link5","gripper","tip_left","tip_right","grasp_frame") -> 4x4 in world
class DualArmKinematics:
    def __init__(self, mounts: dict[str, np.ndarray] | None = None, urdf_path: str | None = None)
    def link_poses(self, side: str, q6) -> dict[str, np.ndarray]
    def fingertip(self, side: str, q6) -> np.ndarray   # (3,) world, midpoint of tip_left/tip_right at gripper z=-0.1468 i.e. the finger end
```

## Geometry (kiteos.safety.geometry)
Covering spheres per link (centre in link frame, radius already includes +5 mm pad):
```
link1: (-0.0315,-0.003,-0.0092) r0.065
link2: (0,-0.0334,0.0235) r0.070; (0,-0.0323,0.132) r0.070; (0,-0.0334,0.2406) r0.070
link3: (-0.0276,-0.0349,-0.0044) r0.080; (-0.0624,-0.0349,-0.0572) r0.077; (-0.0608,-0.033,-0.1491) r0.061; (-0.06,-0.0344,-0.2327) r0.061
link4: (-0.0184,0.03,-0.0134) r0.066; (-0.0694,0.0339,-0.0667) r0.051
link5: (0.0345,0,-0.0011) r0.054
gripper (incl. fingers): (0,0,-0.0157) r0.044; (0,0,-0.0681) r0.068 [+stroke]; (0,0,-0.1206) r0.046 [+stroke]
```
Finger spheres (last two gripper spheres) are inflated by `gripper_stroke_m * opening` (opening 0..1,
default stroke 0.05 m; if opening unknown use 1.0). `extra_spheres` from FenceConfig are appended
(e.g. wrist cameras: link "gripper", centre in gripper frame, radius; |centre| <= 0.3 m, radius <= 0.2 m).
```python
@dataclass class Spheres: centers: np.ndarray (N,3) world; radii: np.ndarray (N,); labels: list[str] ("left.link3[1]", "left.camera", ...)
def arm_spheres(side, link_poses, gripper_opening: float | None, stroke_m: float, extra: list[ExtraSphere]) -> Spheres
```

## Zones (kiteos.safety.zones) — pydantic v2, extra="forbid"
```python
Frame = Literal["dual_openyam_base", "left_base", "right_base"]
AppliesTo = Literal["both", "left", "right"]
class HalfSpaceZone:  kind: Literal["half_space"]; name: str; frame: Frame="dual_openyam_base"; point: tuple[float,float,float]; normal: tuple[float,float,float]  # normal points INTO the forbidden side (toward the glass and beyond)
class BoxZone:        kind: Literal["box"];        name; frame; center: xyz; half_extents: xyz (>0); rpy: xyz=(0,0,0)  # forbidden solid
class KeepInBox:      kind: Literal["keep_in_box"]; name; frame; center; half_extents; rpy    # allowed volume; spheres must stay fully inside
Zone = Annotated[HalfSpaceZone | BoxZone | KeepInBox, Field(discriminator="kind")]; every zone also has applies_to: AppliesTo="both", enabled: bool=True
def clearance(zone, centers_world, radii, frame_T: dict[str, np.ndarray]) -> np.ndarray  # per-sphere signed clearance in metres: >0 free distance, <=0 penetration (margin NOT included)
```
Frames: `left_base`/`right_base` zones are expressed in that arm's base frame (mount transform from
FenceConfig), which lets you teach a zone per arm without trusting the inter-arm spacing.

## Fence (kiteos.safety.fence)
```python
class Mount(BaseModel): xyz: tuple3; rpy: tuple3 = (0,0,0)
class ExtraSphere(BaseModel): name: str; side: Literal["left","right","both"]="both"; link: str; center: tuple3; radius: float
class FenceConfig(BaseModel, extra="forbid"):
    version: Literal[1] = 1
    station: str = ""
    mode: Literal["enforce", "monitor"] = "enforce"     # monitor = log only; refused on real hardware
    margin_m: float = Field(0.05, ge=0.02)              # extra clearance added to every sphere
    gripper_stroke_m: float = Field(0.05, ge=0)
    mounts: dict[Literal["left","right"], Mount] = {"left": Mount(xyz=(0,0.31,0)), "right": Mount(xyz=(0,-0.31,0))}
    extra_spheres: list[ExtraSphere] = []
    zones: list[Zone] = []
    max_joint_speed_rad_s: float = Field(2.5, gt=0)     # final-stage rate limit; real hardware refuses > 3.0
    joint_limit_margin_rad: float = Field(0.01, ge=0)
    interpolation_step_rad: float = Field(0.01, gt=0, le=0.02)  # swept check between successive commands
    refuse_start_inside: bool = True
    max_feedforward_tau_nm: float = Field(0.0, ge=0, le=20)  # |tau| bound on arm commands
    allow_no_zones: bool = False                        # real hardware refuses an arm without a zone unless true
    def unprotected_arms(self) -> list[str]
    @classmethod def load(cls, path) -> "FenceConfig"   # YAML; empty file and duplicate keys are errors
    def dump(self, path) -> None
def real_hardware_problems(config) -> list[str]         # monitor mode, unprotected arm, speed > 3.0
def escape_path_ok(clearances (M, P), margin) -> bool   # per (zone, sphere) pair: within margin -> non-decreasing, clear -> stays clear
@dataclass class ArmReport: side: str; ok: bool; min_clearance_m: float; zone: str|None; sphere: str|None
@dataclass class FenceReport: ok: bool; arms: dict[str, ArmReport]; def to_dict(self) -> dict
class FenceChecker:
    def __init__(self, config: FenceConfig, urdf_path: str | None = None)
    config: FenceConfig; joint_limits: dict[str, np.ndarray]  # per side (6,2)
    def check(self, q14_or_by_name, gripper_openings: dict[str,float] | None = None) -> FenceReport   # accepts np.ndarray(14|12) in DUAL_OPENYAM_JOINTS order or dict name->value
    def check_arm(self, side, q6, gripper_opening=None) -> ArmReport      # ok iff every sphere has clearance > margin_m for every enabled zone that applies to the side
    def check_arm_motion(self, side, q6_from, q6_to, gripper_opening=None) -> ArmReport  # worst case over interpolated samples (max joint step <= interpolation_step_rad), includes endpoints
    def clamp_to_limits(self, side, q6) -> np.ndarray         # URDF limits minus joint_limit_margin_rad
    def pair_clearances(self, side, samples (M,6), opening) -> np.ndarray  # (M, Z*N)
    def is_escape_motion(self, side, q_from, q_to, opening) -> bool       # start within margin, escape_path_ok, some pair gains
    def check_escape_path(self, side, segments[(q_from, q_to, opening)]) -> ArmReport  # precheck when starting inside
```

## Enforcement (kiteos.safety.adapter / kiteos.safety.coordinator) — imports dimOS
`FencedWholeBodyAdapter(inner, *, joint_names: list[str], checker: FenceChecker, on_event=None, clock=time.monotonic)`
implements the dimOS WholeBodyAdapter protocol (connect, disconnect, is_connected, read_motor_states,
has_motor_states, read_imu, get_limits, write_motor_commands) + activate/deactivate (delegate if the inner
has them). write_motor_commands(cmds: list[MotorCommand] len 14), per arm independently:
1. requested q6 from cmds; if any non-finite -> hold.
2. stale target: a raw arm request bit-identical to the one last refused (hold, E-STOP or error) ->
   hold without re-checking, until a different request arrives (dimOS re-sends pre-fence targets).
3. clamp to URDF limits (minus margin); rate-limit vs last written target: step <= max_joint_speed_rad_s * clamp(dt, 0.001, 0.02).
4. accept if check_arm_motion(last_written -> candidate) ok; else accept if FenceChecker.is_escape_motion
   (per (zone, sphere) pair along the dense samples: pairs within margin never lose clearance, clear
   pairs stay clear, one pair within margin gains); else HOLD:
   q = last_written (known safe), kp/kd = max(cmd, arm defaults kp=(80,80,80,10,10,10) kd=(5,5,5,1.5,1.5,1.5)), dq=0, tau=0.
   Accepted: dq clamped to +-speed, tau clamped to +-max_feedforward_tau_nm.
5. grippers: closing always passes; an opening increase passes only if the arm plan with the wider
   fingers is clear (ok, not an escape; or, when held, check_arm(last, wider) ok); else the gripper
   is held at its last written opening (dq=0, tau=0) with a "hold" event. The finger inflation uses
   max(sent opening, measured opening).
6. estop latched -> hold both arms and both grippers (grippers frozen at the measured opening).
   mode="monitor" -> never modify, only emit events.
7. Any exception -> hold both arms and grippers at last_written (fail closed), log once per second.
Events carry new_episode=True on the first hold/error of a hold episode.
limit_clamp is only reported when the target exceeds the URDF limit (routine clamps are silent).
Tracks: last_written (per arm, initialised from measured q on first read/activate), measured q cache from
read_motor_states (never re-reads the bus), event counters, last event.
`FenceEvent(t, side, kind: "hold"|"escape"|"limit_clamp"|"rate_limit"|"estop"|"error"|"start_inside", zone, sphere, clearance_m, detail, enforced, new_episode)`.
Extras: `latch_estop(bool)`, `status() -> dict` (mode, zones, holds count, per-arm last report, estop),
`check_positions(dict name->q) -> dict`, `rewrite_applied(joint_state_msg) -> msg` (replaces arm positions by
what was actually written so recorded actions are truthful), `reload(config: FenceConfig)` (atomic swap).
activate(): delegate, then read measured q; if refuse_start_inside and any arm violates -> deactivate inner
and return False (coordinator raises). Initialise last_written = measured.

`KiteDualOpenYamCoordinator(DualOpenYamCoordinator)` in `kiteos.safety.coordinator`, config
`KiteDualOpenYamCoordinatorConfig(DualOpenYamCoordinatorConfig)` adds:
`fence_path: str | None = None`, `require_fence_on_hardware: bool = True`,
`expect_mock: bool = False` (the CLI passes true without --real: any real adapter is refused).
- `_create_whole_body_adapter(component)`: inner = super()...; if fence_path is None: raise if
  adapter_type != "mock_whole_body" and require_fence_on_hardware; with mock and no fence, still wrap with an
  empty FenceConfig (limits/rate still enforced). Loads FenceConfig from fence_path (absolute or relative to cwd).
  On real hardware: raise on real_hardware_problems(config) (monitor, unprotected arm without
  allow_no_zones, speed > 3.0). An empty fence file is a load error.
- `start()`: shadow `applied_joint_position_command.publish` with the rewriting publisher BEFORE
  super().start() (dimOS's TickLoop captures port.publish), then verify the tick loop uses it
  (fallback: wrap `TickLoop._publish_command_callback`, assert attr exists).
- `set_estop(estopped)`: fenced.latch_estop, cancel_trajectory() when estopped, then super().
- `execute_trajectory(trajectory)` and `task_invoke(<JointTrajectoryTask>, "execute"|"on_joint_command", ...)`:
  reject (INVALID_TRAJECTORY; on_joint_command returns False) if estop latched or the path violates the
  fence (check_waypoints: routes current->wp0->... and current->wp1->...; when an arm starts within the
  margin only an escape path that ends clear passes); else super().
- RPCs: `fence_status() -> dict` (recent_events coalesced by (side, kind, zone) with "repeats",
  plus recent_holds), `fence_check(positions: dict[str,float]) -> dict`,
  `fence_reload(path: str | None = None) -> dict` (real hardware: refuses another station, logs the diff).
- Fence events on the tick thread must never call coordinator RPCs (non re-entrant locks); a daemon
  supervisor thread cancels an executing trajectory at most once per hold episode.

## Blueprints (dimos.blueprints entry points; names must match ^[a-z0-9]+(-[a-z0-9]+)*$)
```
[project.entry-points."dimos.blueprints"]
dual-openyam-teleop     = "kiteos.blueprints.teleop:dual_openyam_teleop"
dual-openyam-collection = "kiteos.blueprints.collection:dual_openyam_collection"
dual-openyam-rollout    = "kiteos.blueprints.rollout:dual_openyam_rollout"
dual-openyam-rollout-recorded = "kiteos.blueprints.rollout:dual_openyam_rollout_recorded"
```
Run as `dimos run kiteos.dual-openyam-teleop ...`. All use `KiteDualOpenYamCoordinator` with
instance_name "ControlCoordinator" (so flags are `--controlcoordinator.left-can-port`,
`--controlcoordinator.right-can-port`, `--controlcoordinator.fence-path`). Cameras: instance names
`left_wrist`, `right_wrist`, `top`; flags `--left-wrist.hardware.camera-index <int|/dev path>` etc.
Viser binds 127.0.0.1 by default (flag to open it); the Quest WebXR server too (KiteArmTeleopModule,
instance "armteleopmodule", `--armteleopmodule.server-host`). Camera defaults are the lab Mac's
mapping (left_wrist 0, top 1, right_wrist 2); the CLI always passes the station's devices.
Collection = teleop + 3 cameras + collection_recorder
(DUAL_OPENYAM_COLLECTION) + EpisodeMonitorModule(instance "episodes"), mirroring dimOS
learning_collection.py. Rollout = port of kite_rollout with the KiteOS coordinator; the trajectory
task's velocity_limits (1.0 per joint) are explicit; `-rollout-recorded` adds the recorder.
With no CAN ports given, the coordinator uses dimOS's mock adapter (safe for testing).

## Station config (kiteos.config) — YAML, `KITEOS_STATION` env or `config/station.yaml`
```yaml
station: kite-lab-dual-openyam
task: "close the box"
can: {left: {darwin: 006C003E4845500F20303750, linux: can_left}, right: {darwin: 003900374845500F20303750, linux: can_right}}
cameras: {left_wrist: {darwin: 0, linux: /dev/v4l/by-path/REPLACE}, right_wrist: {darwin: 2, linux: ...}, top: {darwin: 1, linux: ...}}
fence: config/fence.yaml        # relative to the station file
recordings_dir: ~/teleop_stack/dimos/recordings
datasets_dir: ~/teleop_stack/dimos/datasets
hub: {repo_id: kiteml/dual-openyam-close-box, private: true}
policy: {checkpoint: ~/teleop_stack/policies/act-close-box-step30000, device: auto,   # auto -> cuda|mps|cpu
         max_start_jump: 0.3, max_joint_speed: 3.0}                                    # rollout chunk guards
dimos_root: ~/teleop_stack/dimos
viser_host: 127.0.0.1
quest: {port: 8080, upstream: https://127.0.0.1:8443, adb: auto}
```
`StationConfig.load(path=None)`, `.can_port(side)`, `.camera(role)` resolve by `sys.platform`
("darwin"/"linux"; other -> "linux" values), expanding `~`. `.fence_path()` absolute.

## CLI (`kiteos`, Typer) — prints a KiteOS banner; all commands support --station PATH
- `kiteos doctor` — environment, pinned dimOS commit, entry points visible (`dimos list`), recorder binary,
  URDF, station+fence validation, FK self-test, adb, CAN (darwin: `dimos hardware can list`; linux: `ip link`),
  cameras enumerate, HF auth. Exit non-zero on blocking failures.
- `kiteos teleop [--real] [--no-cameras]`, `kiteos collect start [--real] [--session NAME]`,
  `kiteos collect controls`, `kiteos rollout start --checkpoint ... [--real] [--speed]`, `kiteos rollout controls`
  -> `dimos run kiteos.<blueprint> --config <empty> <flags from station>` in the dimOS root with
  `uv run --no-sync` and PYTHONPATH=<repo>/src. Always pinned: `--controlcoordinator.require-fence-on-hardware true`,
  `--controlcoordinator.expect-mock true|false`. Refused: CONTROLCOORDINATOR__/G__/TRANSPORTS__ keys in
  the environment or <dimos_root>/.env; another dimOS stack on the host (registry, KiteOS run-state,
  process scan, 8443/8095 listeners, and for --real a passive bus listen). `--real` requires a fence
  passing real_hardware_problems + interactive safety confirmation. Without `--real` no CAN ports are
  passed (mock arms), and the run gets a private bus (LCM port + zenoh scouting group), free
  WebXR/viser ports and `--viewer none` (`--station-ports`, `--allow-concurrent` for mock only).
  A run-state file (pid, real, bus, ports, recording) is written before exec; client commands
  (collect/rollout controls, fence check/teach, stop, quest proxy) join that run's bus.
  Mock collections record into `mock-<prefix>-NNN` (never uploaded). `rollout start --record` uses
  `kiteos.dual-openyam-rollout-recorded` and records into `rollout-<prefix>-NNN`.
- `kiteos quest proxy` (branded proxy + adb reverse), `kiteos cameras list|check`, `kiteos can setup|status`,
  `kiteos upload` (refuses mock-* sessions and a non-station prefix without --repo-id),
  `kiteos stop [--pid]` (graceful SIGTERM, KiteOS never kills; typed confirmation unless the run is
  proven mock; the recorder gets `--recorder.shutdown-timeout 100`).
- `kiteos fence show|validate|check|teach|test` (see docs/FENCE.md). `check` connects to a running stack
  (`Dimos.connect()` -> ControlCoordinator RPC `fence_status`/`fence_check`); `teach` records fingertip points
  from live joint positions (RPC `get_joint_state` or the coordinator_joint_state stream) and fits a plane;
  `test` runs an offline sweep and replays LeRobot datasets through the fence.
