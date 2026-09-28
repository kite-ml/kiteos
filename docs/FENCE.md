# KiteOS safety fence

The fence keeps a robot's arms out of the space you declare forbidden, such as a customer's glass
window, a monitor or a shelf. It protects the two robots KiteOS runs with its own setups: the dual
OpenYAM and the OpenArm (the fence file's `robot:` says which). It is a software guard that runs on the KiteOS control
coordinator. It does **not** replace physical safeguards: an emergency stop within reach, an
operator watching the arms, and reduced speed near people and fragile objects.

## What the fence does

Every command that reaches the arms goes through one call per control tick (100 Hz):
`FencedWholeBodyAdapter.write_motor_commands`, which runs just before the CAN write. Quest
teleop IK, policy rollouts (`execute_trajectory`), gripper tasks and direct joint commands all
take this path. For each arm, independently, the fence then does the following:

1. It rejects non-finite joint targets and holds.
2. It clamps the target to the URDF joint limits minus `joint_limit_margin_rad`.
3. It limits the rate, so a target moves at most `max_joint_speed_rad_s` × dt from the last
   written target.
4. It runs a swept check: the straight joint-space motion from the last written target to the
   new one is sampled so that no joint moves more than `interpolation_step_rad` between samples.
   Both endpoints are included. At every sample, the fence computes forward kinematics from the
   vendored URDF and places covering spheres on the arm. The motion is accepted only if every
   sphere stays more than `margin_m` from every enabled zone that applies to that arm.
5. If the check fails, the arm holds its last written target, which is known to be safe. There is
   one exception, so an arm that is already too close (after a hot reload or a newly taught zone)
   can back out: an **escape**. Every (zone, sphere) pair is followed along the densely sampled
   motion. A pair that is within the margin must never lose clearance, a pair that is clear must
   stay clear, and at least one pair within the margin must gain clearance. So no part of the arm
   ever gets closer to any zone while it is too close. The arm cannot trade one sphere or one
   zone against another.
6. A **stale target stays refused.** The control loop re-sends the last pre-fence target of every joint that
   a command does not mention; for example, the gripper task sends only the gripper. An arm
   request that is bit-for-bit the one the fence last refused is held without being re-checked,
   until a different request arrives. So a held arm never drives off later by itself: not when the
   gripper closes, not after a fence reload, and not after an E-STOP is cleared.
7. Accepted commands have their velocity feed-forward bounded by the speed limit, and their torque
   feed-forward by `max_feedforward_tau_nm` (default 0: the Damiao adapter adds gravity
   compensation itself).

Some other rules apply:

- **Grippers are fenced too.** Closing is always allowed. An opening increase is allowed only if
  the arm, where it will be this tick, stays clear of the fence with the wider finger spheres.
  Otherwise the gripper keeps its last written opening and a `hold` event is reported. So a
  gripper never opens into a zone, and it cannot open at all while its arm is inside one.
- A latched E-STOP holds both arms and both grippers (each gripper at its measured opening).
- `mode: monitor` only logs events. It is refused on real hardware.
- Any internal error holds both arms and both grippers at their last written targets, so the
  fence fails closed.
- If an arm already violates the fence when the stack starts, the coordinator refuses to start
  when `refuse_start_inside: true`.
- Recorded actions are rewritten to what was actually sent to the motors, grippers included, so a
  dataset never contains a fenced-out action. The rewrite is installed before the runtime starts its
  control loop, so this covers the first tick too.
- On real hardware the coordinator refuses (at start and on `fence_reload`):
  - a fence that leaves an arm with no enabled zone, unless the file says `allow_no_zones: true`;
  - an empty or comment-only fence file (never "a fence with no zones");
  - `mode: monitor`, and `max_joint_speed_rad_s` above 3 rad/s;
  - on reload, a file for another `station`.

  A reload logs every setting it changes (margin, speed, mounts, extra spheres, zones).

### Geometry

**Dual OpenYAM.** The URDF has no collision shapes, so KiteOS covers each arm with 14 spheres.
They were fitted to the visual meshes and each radius includes a 5 mm pad. From base to gripper: 1
on link1, 3 on link2, 4 on link3, 2 on link4, 1 on link5 and 3 on the gripper including the closed
fingers. The last two gripper spheres cover the fingers.

**OpenArm.** 23 spheres per arm, fitted by `tools/fit_link_spheres.py` to the collision and visual
meshes of `enactic/openarm_description` (every vertex and 40,000 surface samples per mesh are
inside, each radius includes a 5 mm pad): 1 on base_link, 2 each on link1, link2, link4, link5 and
link6, 3 on link3, 1 on the hand (ee_base_link) and 4 on each finger (ee_link1, ee_link2). The left
arm's meshes are mirrored, so each side has its own table (`src/kiteos/data/openarm_v20_spheres.json`).
The fingertip is the closed fingers' distal end, 3.2 cm beyond the grasp frame.

The fingers are modelled closed. The finger spheres therefore grow by `gripper_stroke_m ×
opening`, where opening is between 0 (closed) and 1 (open). An unknown opening counts as fully
open. This growth is the same in every direction, so it is conservative. The OpenArm's pinch
fingers swing 45°; its default stroke, 0.081 m, bounds how far any finger point moves at any
opening.

Anything else attached to the arm must be added under `extra_spheres`. That includes wrist
cameras, cables and held tools.

With the fingers pointing straight at a plane, the closed-finger sphere reaches 2 cm beyond the
fingertip. The arm therefore stops when the fingertip is about `margin_m + 0.02 m` from the zone,
plus `gripper_stroke_m × opening` when the gripper is open.

### Zones and frames

| kind | forbidden region | fields |
| --- | --- | --- |
| `half_space` | everything on the side `normal` points to | `point`, `normal` (need not be unit length) |
| `box` | the solid box | `center`, `half_extents` (> 0, half the size), `rpy` |
| `keep_in_box` | everything outside the box (spheres must stay fully inside) | same as `box` |

Every zone also has these fields:

- `name` (unique)
- `frame`: default `dual_openyam_base` on the dual OpenYAM, `world` on the OpenArm
- `applies_to`: `both`, `left` or `right`
- `enabled`: default `true`

The frames are:

- `world`: the robot's world frame, also called by its URDF name:
  - `dual_openyam_base` (dual OpenYAM): +x points forward (the direction the grippers face at
    home), +y points toward the left arm, +z points up, and z = 0 is the mounting surface of the
    arm bases.
  - `openarm_body_link0` (OpenArm): +x forward, +y toward the left arm, +z up, z = 0 at the bottom
    of the body. The arm bases (shoulders) are 0.698 m up, 3.1 cm either side of the centre.
- `left_base` and `right_base`: one arm's own base frame, placed by `mounts`. Use these frames when
  the spacing between the arms has not been measured. Zones taught per arm stay correct whatever
  the spacing.

Box orientation uses URDF roll-pitch-yaw in radians, relative to the zone frame
(R = Rz(yaw)·Ry(pitch)·Rx(roll)). All lengths are in metres.

### In the viewer

KiteOS setups draw the enabled zones in the viser viewer, next to the robot, and add them to the
planner's world, so planned motions keep out of them too. A half-space is drawn as a 10 cm pane on
its forbidden side, centred on the arms and wide enough to cover their reach; a box as itself; a
keep-in box is not drawn. The viewer shows the zones the stack started with. The planner and the
viewer are aids only: the coordinator enforces the fence.

## What the fence does not guarantee

- **It checks commands, not the measured arm.** The swept check covers the commanded motion
  between ticks. However, the real arm lags behind its command (by several centimetres during fast
  motion) and can overshoot under PD control. `margin_m` must cover this plus calibration error.
  Never set it below the enforced floor of 2 cm, and prefer 3–5 cm.
- **It only covers commands that go through the KiteOS coordinator.** It does not cover other
  tools that open the CAN bus directly, the base runtime's own setups (anything not named
  `kiteos.*`, such as the legacy `start_session.sh` flow), or firmware-level motion.
- **It cannot stop a person.** Someone pushing or back-driving an arm can move it into a zone. The
  fence reacts only to commands.
- **Motors off means the arms fall.** Stopping the stack, a motor fault latch or a power loss
  disables the motors, and the arms drop under gravity with no fence at all. Park both arms away
  from the glass before you stop, and keep the space below the arms clear.
- **It does not detect arm-vs-arm or self collisions.**
- **It is not a certified safety function.** It is not rated to ISO 10218 or ISO 13849. It is
  only as accurate as the model, the mounts and the taught zones (see Limitations).
- **It holds the arm rather than sliding it.** At the boundary, a teleoperated arm freezes instead
  of sliding along the wall. Move back to release it.
- **The trajectory precheck is only an early rejection.** `execute_trajectory` refuses a
  trajectory that would cross the fence before it starts. So do the trajectory task's
  `task_invoke("joint_trajectory", "execute" | "on_joint_command", ...)` commands, and all three
  are refused while the E-STOP is latched. The `joint_command` stream still skips the precheck.
  Its commands are held at the adapter one tick at a time (the mock end-to-end test proves this).
  After a hold, the executing trajectory is cancelled once per hold episode, not once per tick.
- **Backing out of a zone.** When an arm already starts within the margin, the precheck accepts
  only a trajectory that is an escape along its whole densely sampled path (see rule 5) and that
  ends clear. The straight path home usually qualifies; anything that gets closer anywhere is
  rejected. Quest teleop and the `joint_command` stream get the same rule tick by tick.

## Configuration reference (`config/fence.yaml`)

| field | default | meaning |
| --- | --- | --- |
| `version` | `1` | schema version |
| `robot` | `dual-openyam` | the robot the fence protects: `dual-openyam` or `openarm`. A stack refuses a fence written for another robot |
| `station` | `""` | free-text station name |
| `mode` | `enforce` | `enforce` or `monitor` (log only; refused on real hardware) |
| `margin_m` | `0.05` | minimum distance between any sphere surface and any zone; **must be ≥ 0.02** |
| `gripper_stroke_m` | `0.05` (OpenArm `0.081`) | how far ONE finger travels from closed to fully open; the finger spheres grow by this at opening 1.0. Replace the dual OpenYAM default with the measured value |
| `mounts.left` / `mounts.right` | dual OpenYAM `xyz (0, ±0.31, 0)`; OpenArm `xyz (0, ±0.031, 0.698)`; `rpy (0,0,0)` | pose of each arm base in the world frame; both sides are required; at most 2 m from the origin |
| `extra_spheres[]` | `[]` | `name`, `side` (`left`, `right` or `both`), `link` (the robot's links without the side prefix: dual OpenYAM `base`, `link1`–`link5`, `gripper`, `tip_left`, `tip_right`, `grasp_frame`; OpenArm `base_link`, `link1`–`link6`, `ee_base_link`, `ee_link1`, `ee_link2`, `grasp_frame`), `center` (in the link frame, at most 0.3 m from its origin), `radius` (> 0, at most 0.2 m) |
| `zones[]` | `[]` | see Zones and frames |
| `max_joint_speed_rad_s` | `2.5` | final rate limit on commanded arm targets; real hardware refuses more than 3.0 |
| `joint_limit_margin_rad` | `0.01` | distance kept inside the URDF joint limits |
| `interpolation_step_rad` | `0.01` | largest per-joint step between swept-check samples; at most 0.02 |
| `refuse_start_inside` | `true` | refuse to start if an arm already violates the fence |
| `max_feedforward_tau_nm` | `0.0` | largest torque feed-forward (Nm) passed to an arm motor; at most 20 |
| `allow_no_zones` | `false` | allow real arms with an arm that no enabled zone covers. Only for teaching the first zone; `kiteos fence teach` clears it once both arms are covered |

Unknown keys are rejected, and so are duplicated keys anywhere in the file (YAML would otherwise
silently keep the last one). An empty or comment-only file is an error. `kiteos fence validate`
reports every problem at once. For example:

```text
config/fence.yaml: invalid fence config:
  - margin_m: Input should be greater than or equal to 0.02 (KiteOS never allows less than 2 cm of margin)
  - zones[0].normal: normal must be a non-zero vector pointing INTO the forbidden side (from the robot toward the glass)
  - zones[1].half_extents: half_extents must all be > 0 metres (got (0.1, 0.0, 0.1)); they are HALF the box size
  - zones[2]: missing 'kind' (one of: half_space, box, keep_in_box)
```

### Example: glass window in front of the robot

The glass was measured about 0.57 m in front of the arm bases. The taught plane below already
includes a 2 cm standoff, so it sits at x = 0.55 m. `kiteos fence validate` passes for it.

```yaml
# KiteOS fence configuration. Reference: docs/FENCE.md
version: 1
station: customer-glass-cell
mode: enforce
margin_m: 0.03
gripper_stroke_m: 0.05          # replace with the measured per-finger travel
mounts:
  left:  {xyz: [0.0, 0.31, 0.0], rpy: [0.0, 0.0, 0.0]}
  right: {xyz: [0.0, -0.31, 0.0], rpy: [0.0, 0.0, 0.0]}
extra_spheres:
  # Example values: measure your camera mount. Gripper frame: +z points back toward the wrist,
  # +y is the side of the URDF's tip_left finger.
  - {name: wrist_camera, side: both, link: gripper, center: [0.0, -0.055, -0.02], radius: 0.035}
zones:
  - kind: half_space
    name: glass_window
    frame: dual_openyam_base
    point: [0.55, 0.0, 0.0]     # any point on the (standoff-shifted) plane
    normal: [1.0, 0.0, 0.0]     # points INTO the glass, away from the robot
  - kind: box
    name: table_top               # table surface in front of the bases; top face at z = 0
    center: [0.35, 0.0, -0.05]
    half_extents: [0.20, 0.80, 0.05]
max_joint_speed_rad_s: 2.5
joint_limit_margin_rad: 0.01
interpolation_step_rad: 0.01
refuse_start_inside: true
max_feedforward_tau_nm: 0.0
```

What to expect with this file:

- At home (`[0, 1.047, 1.047, 0, 0, 0]`) the fingertip is at x = 0.389 m.
- At home the nearest sphere surface is 14.1 cm from the plane with the gripper closed, and 9.1 cm
  with it fully open (assumed 5 cm stroke). Both are above `margin_m: 0.03`, so the arm can start
  there and the gripper can open there.
- Had the glass been measured at 0.47 m (plane at 0.45 m), home would be clear only with the
  gripper closed (+4.1 cm), not open (-0.9 cm). `kiteos fence validate` reports that as a FAIL,
  because the fence would then refuse every gripper opening at home.
- The table is protected with a box that starts 15 cm in front of the bases. Do not use a
  half-space at the mounting surface: link1's sphere always sits 3.45 cm above that surface, so
  such a plane would leave almost no room for a margin.

Place zones with the robot's real working poses in mind, and replay your recorded episodes
through `kiteos fence test` before a customer visit.

## Teaching a glass plane

Measuring the glass with a tape is error-prone. Instead, touch it with the fingertip and let
KiteOS fit the plane from the arm's own kinematics. This also absorbs any error in the base
spacing.

1. **Prepare.**
   - Start teleop at low speed, for example `max_joint_speed_rad_s: 0.5`.
   - Leave the glass zone out of the fence file, or set `enabled: false`. Keep any other zones.
     While an arm has no enabled zone, a `--real` run needs `allow_no_zones: true` in the file.
   - Have an operator's hand on the E-STOP.
   - Close the gripper. The fingertip point is defined for closed fingers, 14.68 cm along the
     gripper axis.
2. **Touch at least 4 points, 5 or more is better,** on the glass with the fingertip. Spread them
   as widely as the arm comfortably reaches, for example the corners of a quadrilateral 20–30 cm
   across plus its centre. Touch lightly and hold still while each point is recorded.
   `kiteos fence teach` reads the live joint positions and records the fingertip,
   `DualArmKinematics.fingertip(side, q)`. Run `kiteos fence teach --help` for the options.
3. **Fit.** `plane_zone_from_points(points, robot_ref, name, frame, applies_to, standoff_m)`
   does the following:
   - It fits the plane by SVD. It needs at least 4 points: with 3 the residual is always zero, so
     a missed touch would go unnoticed. The points must spread at least 5 cm (RMS) along both
     directions of the plane.
   - It rejects the fit if the RMS residual is above 1 cm. A large residual means the points are
     not on one plane: the glass flexed, a point was missed, or the model is off.
   - It rejects the fit if its tilt uncertainty could move the plane by more than the standoff
     (at least 1 cm) 50 cm away from the touched points (2 sigma). Away from the touched patch,
     the zone could otherwise lie behind the real glass.
   - `--force` overrides both checks. Re-touch rather than forcing.
   - It orients the normal away from `robot_ref`, a point well inside the workspace such as the
     arm base.

   The preview prints the per-point residuals and the tilt uncertainty. It also prints whether the
   home and zero poses stay clear with the gripper open, and warns when an arm would be left
   without any zone.
4. **Add a standoff.** `standoff_m` (typically 0.02) moves the plane toward the robot to absorb
   FK and calibration error. `margin_m` still applies on top.
5. **Per-arm option.** If the spacing between the arm bases has not been measured, teach the glass
   separately for each arm: `kiteos fence teach --arms left`, then `kiteos fence teach --arms
   right`.
   - Each zone is taught in that arm's own base frame (`left_base` / `right_base`), applies to that
     arm, and gets its own default name (`glass_window_left` / `glass_window_right`), so the second
     run never replaces the first arm's zone.
   - Replacing an existing zone with one in another frame, or for other arms, is refused unless
     you pass `--replace`.
   - `--arms both` with a per-arm `--frame` needs an explicit `--applies-to`.
   - Once both arms have an enabled zone, teach switches `allow_no_zones` off.
6. **Save and check.** Run `kiteos fence validate` and `kiteos fence show`, then restore the normal
   speed. `validate` fails when an arm has no zone (unless `allow_no_zones`), and when home or
   zero is blocked with the gripper closed or open. It warns when no extra sphere models a wrist
   camera.

For boxes, `box_zone_from_corners(corners, name, pad_m=..., keep_in=False)` builds an
axis-aligned box in the chosen frame from touched corner points. `pad_m` grows a keep-out box
and shrinks a keep-in box.

## Validation before relying on the fence

1. **Schema.** `kiteos fence validate`.
2. **Calibration.** Command home on the real arms and tape-measure the fingertip against FK:
   about (0.389, ±0.31, 0.402) m in `dual_openyam_base`. This confirms that motor zero matches URDF
   zero. A zeroing error shifts the whole fence.
3. **Offline replay.** `kiteos fence test` runs an offline sweep and replays recorded LeRobot
   datasets through the fence. Normal task motion should produce no violations. If it does, the
   zone is too tight or the task needs to change.
4. **Mock run.** `kiteos teleop` without `--real` enforces the same fence on mock arms. Drive into
   the zone and confirm the arm holds, and `kiteos fence check` reports the violation.
   The automated version is `tests/e2e/test_mock_fence_e2e.py` (see "Mock end-to-end test" below).
5. **Low-speed approach on hardware.**
   - Use a reduced speed limit.
   - For the first test, use a cardboard panel standing where the glass will be.
   - Approach the glass slowly at several points across the pane:
     - with the gripper closed and open;
     - with each wrist camera facing the glass;
     - with the elbow leading.
   - Measure the gap at which each arm stops, and confirm that it is at least `margin_m` plus the
     standoff.
6. **Re-validate after any change** to mounts, cameras, cables, tooling, grippers, the URDF, motor
   zeroing or the zone file.

## Mock end-to-end test

`tests/e2e/test_mock_fence_e2e.py` (marker `mock_e2e`, about 45 s) launches the
`kiteos.dual-openyam-teleop` setup with `--controlcoordinator.fence-path tests/fixtures/fence_e2e.yaml`
in the foreground with mock arms. It uses the flags `kiteos teleop` pins for a mock run
(`--config <empty>`, `--controlcoordinator.expect-mock true`, `--controlcoordinator.require-fence-on-hardware true`).
It then drives the running `ControlCoordinator` over RPC:

- The WebXR server listens on 127.0.0.1 only. KiteOS's passive bus probe sees the stack's
  traffic, and its process scan finds the stack.
- The fixture fence is a plane at x = 0.50 m with a 3 cm margin, between the home fingertip and a
  reach pose whose fingertip is 15 cm past it.
- `execute_trajectory` to the reach pose is rejected by the precheck, and so is the same
  trajectory sent through `task_invoke("joint_trajectory", "execute" | "on_joint_command")`. The
  arm does not move.
- A trajectory accepted under a zone-less fence meets the glass fence, hot-reloaded while it
  runs. The adapter holds the arm about 3 cm from the plane, and the supervisor cancels the
  trajectory exactly once.
- Small streamed steps on `joint_command` (no precheck) are held again. The stream re-executes the
  trajectory task about 50 times a second, but it is cancelled once per hold episode.
- A hot-reloaded plane 4 cm closer puts the held arm inside a zone. A trajectory that goes deeper
  is rejected. The straight trajectory home is accepted as an escape and executed: between
  measured samples no zone/sphere pair within the margin loses clearance, and it ends clear.
- `kiteos fence check` reads the running fence (exit 0 at home, exit 1 for the reach pose).
- Every published joint state (about 1,500 ticks) and every RPC reading of the measured mock
  positions is re-checked by an independent `FenceChecker`. None comes closer than the margin.
- SIGTERM stops the run cleanly (exit 0).

What it does **not** prove: anything about the real arms. The mock adapter tracks commands
exactly, so tracking lag, overshoot, CAN timing and calibration are untested (see "What the fence
does not guarantee"). Quest teleop IK is not driven either. The test runs on a private bus and
free ports, so it cannot reach a stack running on the default bus. Set
`KITEOS_E2E_ARTIFACTS=<dir>` to keep the runtime log and a JSON summary.

## Limitations

- **Model accuracy.** The kinematics come from the vendored URDF (the dual OpenYAM's has visual
  meshes only; the OpenArm's is the expanded `enactic/openarm_description` model). The spheres
  cover the meshes with a 5 mm pad. Any mismatch between the model and the real arm (link
  lengths, motor zero, flex) moves the real arm relative to the fence.
- **Mount spacing.** The URDF's 0.62 m spacing between the bases comes from a simulation scene.
  Measure the spacing and set `mounts`, or teach zones per arm in `left_base`/`right_base`.
- **Wrist cameras and cables are not in the URDF.** Add each one as an `extra_spheres` entry on
  `gripper` or `link5`. Measure its centre in the link frame, and use a radius that covers the
  camera body and its cable loop.
- **Held objects.** An object in the gripper is not modelled. Add an extra sphere on `gripper`
  (below the fingertips) sized for the largest object handled, or increase the margin.
- **Finger stroke.** `gripper_stroke_m` defaults to an unmeasured 5 cm, and the growth is the same
  in every direction. Measure the real per-finger travel. Expect open grippers to be stopped
  earlier than closed ones, and a gripper near a zone to refuse opening.
- **Wrist cameras.** Until they are measured, the shipped `config/fence.yaml` puts a 3.5 cm sphere
  on both sides of each gripper. That is conservative, but it only covers a camera body that
  sticks out no more than those spheres. `kiteos fence validate` and `--real` warn when no extra
  sphere models a camera.
- **Discrete checks.** The swept check samples straight joint-space segments between successive
  commands. The physical path between ticks, tracking lag and overshoot are covered only by the
  margin.
- **Human pushing the arm, or motors off.** Neither is covered (see above). Park away from the
  glass before stopping, and never leave the arms powered off above anything fragile.
- **Static base.** The base itself is not checked. A zone that cuts through an arm base is a
  configuration error.
