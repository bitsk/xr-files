// SPDX-License-Identifier: GPL-3.0-or-later
//! OpenXR 6DoF Dual-Hand Controller Action Set, Schmitt Trigger Hysteresis,
//! Quadratic Thumbstick Curve, held secondary buttons, and Graded Haptics.

use crate::{
    app::{HandControllerInput, HapticEvent},
    spatial::{quat_forward, Pose3D},
};
use anyhow::{Context, Result};
use openxr as xr;

const TRIGGER_PRESS_TH: f32 = 0.65;
const TRIGGER_RELEASE_TH: f32 = 0.35;
const SQUEEZE_PRESS_TH: f32 = 0.60;
const SQUEEZE_RELEASE_TH: f32 = 0.35;

#[derive(Default, Clone, Copy)]
struct SchmittGate {
    armed: bool,
    latched: bool,
}

impl SchmittGate {
    fn reset(&mut self) {
        *self = Self::default();
    }

    fn update(&mut self, active: bool, value: f32, press_th: f32, release_th: f32) -> bool {
        if !active || !value.is_finite() {
            self.reset();
            return false;
        }
        if !self.armed {
            if value <= release_th {
                self.armed = true;
            }
            return false;
        }
        if self.latched {
            if value <= release_th {
                self.latched = false;
            }
        } else if value >= press_th {
            self.latched = true;
        }
        self.latched
    }
}

#[derive(Default, Clone, Copy)]
struct EdgeButton {
    armed: bool,
}

impl EdgeButton {
    fn reset(&mut self) {
        *self = Self::default();
    }

    fn just_pressed(&mut self, active: bool, down: bool) -> bool {
        if !active {
            self.reset();
            return false;
        }
        let fired = down && self.armed;
        self.armed = !down;
        fired
    }
}

pub struct XrControllers {
    action_set: xr::ActionSet,
    subpaths: [xr::Path; 2],
    grip_action: xr::Action<xr::Posef>,
    aim_action: xr::Action<xr::Posef>,
    grip_spaces: [xr::Space; 2],
    aim_spaces: [xr::Space; 2],
    trigger_val: xr::Action<f32>,
    trigger_click: xr::Action<bool>,
    squeeze_val: xr::Action<f32>,
    squeeze_click: xr::Action<bool>,
    secondary_click: xr::Action<bool>,
    back_click: xr::Action<bool>,
    thumb_click: xr::Action<bool>,
    thumbstick: xr::Action<xr::Vector2f>,
    haptic: xr::Action<xr::Haptic>,

    trig_gates: [SchmittGate; 2],
    sqz_gates: [SchmittGate; 2],
    sec_gates: [SchmittGate; 2],
    back_btns: [EdgeButton; 2],
    thumb_btns: [EdgeButton; 2],
}

impl XrControllers {
    pub fn new(
        instance: &xr::Instance,
        session: &xr::Session<xr::Vulkan>,
        has_bd_controller: bool,
    ) -> Result<Self> {
        let action_set = instance.create_action_set("xr_files", "openKylin XR Files", 0)?;
        let left = instance.string_to_path("/user/hand/left")?;
        let right = instance.string_to_path("/user/hand/right")?;
        let subpaths = [left, right];

        let grip_action =
            action_set.create_action::<xr::Posef>("grip_pose", "Grip Pose", &subpaths)?;
        let aim_action =
            action_set.create_action::<xr::Posef>("aim_pose", "Aim Pose", &subpaths)?;
        let trigger_val =
            action_set.create_action::<f32>("trigger_val", "Trigger Value", &subpaths)?;
        let trigger_click =
            action_set.create_action::<bool>("trigger_click", "Trigger Click", &subpaths)?;
        let squeeze_val =
            action_set.create_action::<f32>("squeeze_val", "Squeeze Value", &subpaths)?;
        let squeeze_click =
            action_set.create_action::<bool>("squeeze_click", "Squeeze Click", &subpaths)?;
        let secondary_click =
            action_set.create_action::<bool>("secondary_click", "Context Menu", &subpaths)?;
        let back_click = action_set.create_action::<bool>("back_click", "Back / Up", &subpaths)?;
        let thumb_click =
            action_set.create_action::<bool>("thumb_click", "Thumbstick Click", &subpaths)?;
        let thumbstick =
            action_set.create_action::<xr::Vector2f>("thumbstick", "Thumbstick", &subpaths)?;
        let haptic = action_set.create_action::<xr::Haptic>("haptic", "Haptic Pulse", &subpaths)?;

        // 1. Khronos Simple Controller
        suggest_profile(
            instance,
            "/interaction_profiles/khr/simple_controller",
            &[
                (&grip_action, "/user/hand/left/input/grip/pose"),
                (&grip_action, "/user/hand/right/input/grip/pose"),
                (&aim_action, "/user/hand/left/input/aim/pose"),
                (&aim_action, "/user/hand/right/input/aim/pose"),
                (&trigger_click, "/user/hand/left/input/select/click"),
                (&trigger_click, "/user/hand/right/input/select/click"),
                (&back_click, "/user/hand/left/input/menu/click"),
                (&back_click, "/user/hand/right/input/menu/click"),
                (&haptic, "/user/hand/left/output/haptic"),
                (&haptic, "/user/hand/right/output/haptic"),
            ],
        )?;

        // 2. Oculus Touch (also used by A081 SteamVR / Monado / WiVRn)
        suggest_profile(
            instance,
            "/interaction_profiles/oculus/touch_controller",
            &[
                (&grip_action, "/user/hand/left/input/grip/pose"),
                (&grip_action, "/user/hand/right/input/grip/pose"),
                (&aim_action, "/user/hand/left/input/aim/pose"),
                (&aim_action, "/user/hand/right/input/aim/pose"),
                (&trigger_val, "/user/hand/left/input/trigger/value"),
                (&trigger_val, "/user/hand/right/input/trigger/value"),
                (&squeeze_val, "/user/hand/left/input/squeeze/value"),
                (&squeeze_val, "/user/hand/right/input/squeeze/value"),
                (&secondary_click, "/user/hand/left/input/x/click"),
                (&secondary_click, "/user/hand/right/input/a/click"),
                (&back_click, "/user/hand/left/input/y/click"),
                (&back_click, "/user/hand/right/input/b/click"),
                (&back_click, "/user/hand/left/input/menu/click"),
                (&thumb_click, "/user/hand/left/input/thumbstick/click"),
                (&thumb_click, "/user/hand/right/input/thumbstick/click"),
                (&thumbstick, "/user/hand/left/input/thumbstick"),
                (&thumbstick, "/user/hand/right/input/thumbstick"),
                (&haptic, "/user/hand/left/output/haptic"),
                (&haptic, "/user/hand/right/output/haptic"),
            ],
        )?;

        // 3. Valve Index Controller
        let _ = suggest_profile(
            instance,
            "/interaction_profiles/valve/index_controller",
            &[
                (&grip_action, "/user/hand/left/input/grip/pose"),
                (&grip_action, "/user/hand/right/input/grip/pose"),
                (&aim_action, "/user/hand/left/input/aim/pose"),
                (&aim_action, "/user/hand/right/input/aim/pose"),
                (&trigger_val, "/user/hand/left/input/trigger/value"),
                (&trigger_val, "/user/hand/right/input/trigger/value"),
                (&squeeze_val, "/user/hand/left/input/squeeze/value"),
                (&squeeze_val, "/user/hand/right/input/squeeze/value"),
                (&secondary_click, "/user/hand/left/input/a/click"),
                (&secondary_click, "/user/hand/right/input/a/click"),
                (&back_click, "/user/hand/left/input/b/click"),
                (&back_click, "/user/hand/right/input/b/click"),
                (&thumb_click, "/user/hand/left/input/thumbstick/click"),
                (&thumb_click, "/user/hand/right/input/thumbstick/click"),
                (&thumbstick, "/user/hand/left/input/thumbstick"),
                (&thumbstick, "/user/hand/right/input/thumbstick"),
                (&haptic, "/user/hand/left/output/haptic"),
                (&haptic, "/user/hand/right/output/haptic"),
            ],
        );

        // 4. HTC Vive Wand
        let _ = suggest_profile(
            instance,
            "/interaction_profiles/htc/vive_controller",
            &[
                (&grip_action, "/user/hand/left/input/grip/pose"),
                (&grip_action, "/user/hand/right/input/grip/pose"),
                (&aim_action, "/user/hand/left/input/aim/pose"),
                (&aim_action, "/user/hand/right/input/aim/pose"),
                (&trigger_val, "/user/hand/left/input/trigger/value"),
                (&trigger_val, "/user/hand/right/input/trigger/value"),
                (&squeeze_click, "/user/hand/left/input/squeeze/click"),
                (&squeeze_click, "/user/hand/right/input/squeeze/click"),
                (&secondary_click, "/user/hand/left/input/trackpad/click"),
                (&secondary_click, "/user/hand/right/input/trackpad/click"),
                (&back_click, "/user/hand/left/input/menu/click"),
                (&back_click, "/user/hand/right/input/menu/click"),
                (&thumbstick, "/user/hand/left/input/trackpad"),
                (&thumbstick, "/user/hand/right/input/trackpad"),
                (&haptic, "/user/hand/left/output/haptic"),
                (&haptic, "/user/hand/right/output/haptic"),
            ],
        );

        // 5. ByteDance PICO 4 / Neo3 when extension is advertised
        if has_bd_controller {
            for profile in [
                "/interaction_profiles/bytedance/pico4_controller",
                "/interaction_profiles/bytedance/pico_neo3_controller",
            ] {
                let _ = suggest_profile(
                    instance,
                    profile,
                    &[
                        (&grip_action, "/user/hand/left/input/grip/pose"),
                        (&grip_action, "/user/hand/right/input/grip/pose"),
                        (&aim_action, "/user/hand/left/input/aim/pose"),
                        (&aim_action, "/user/hand/right/input/aim/pose"),
                        (&trigger_val, "/user/hand/left/input/trigger/value"),
                        (&trigger_val, "/user/hand/right/input/trigger/value"),
                        (&squeeze_val, "/user/hand/left/input/squeeze/value"),
                        (&squeeze_val, "/user/hand/right/input/squeeze/value"),
                        (&secondary_click, "/user/hand/left/input/x/click"),
                        (&secondary_click, "/user/hand/right/input/a/click"),
                        (&back_click, "/user/hand/left/input/y/click"),
                        (&back_click, "/user/hand/right/input/b/click"),
                        (&thumb_click, "/user/hand/left/input/thumbstick/click"),
                        (&thumb_click, "/user/hand/right/input/thumbstick/click"),
                        (&thumbstick, "/user/hand/left/input/thumbstick"),
                        (&thumbstick, "/user/hand/right/input/thumbstick"),
                        (&haptic, "/user/hand/left/output/haptic"),
                        (&haptic, "/user/hand/right/output/haptic"),
                    ],
                );
            }
        }

        let grip_spaces = [
            grip_action.create_space(session, left, xr::Posef::IDENTITY)?,
            grip_action.create_space(session, right, xr::Posef::IDENTITY)?,
        ];
        let aim_spaces = [
            aim_action.create_space(session, left, xr::Posef::IDENTITY)?,
            aim_action.create_space(session, right, xr::Posef::IDENTITY)?,
        ];

        session
            .attach_action_sets(&[&action_set])
            .context("attach OpenXR controller action set")?;

        Ok(Self {
            action_set,
            subpaths,
            grip_action,
            aim_action,
            grip_spaces,
            aim_spaces,
            trigger_val,
            trigger_click,
            squeeze_val,
            squeeze_click,
            secondary_click,
            back_click,
            thumb_click,
            thumbstick,
            haptic,
            trig_gates: [SchmittGate::default(); 2],
            sqz_gates: [SchmittGate::default(); 2],
            sec_gates: [SchmittGate::default(); 2],
            back_btns: [EdgeButton::default(); 2],
            thumb_btns: [EdgeButton::default(); 2],
        })
    }

    pub fn reset(&mut self) {
        self.trig_gates = [SchmittGate::default(); 2];
        self.sqz_gates = [SchmittGate::default(); 2];
        self.sec_gates = [SchmittGate::default(); 2];
        self.back_btns = [EdgeButton::default(); 2];
        self.thumb_btns = [EdgeButton::default(); 2];
    }

    pub fn poll(
        &mut self,
        session: &xr::Session<xr::Vulkan>,
        base_space: &xr::Space,
        time: xr::Time,
    ) -> Result<[HandControllerInput; 2]> {
        session.sync_actions(&[xr::ActiveActionSet::new(&self.action_set)])?;
        let mut out = [HandControllerInput::default(); 2];
        let _ = (&self.grip_action, &self.aim_action);

        for h in 0..2 {
            let sp = self.subpaths[h];
            let aim_loc = self.aim_spaces[h].locate(base_space, time)?;
            let grip_loc = self.grip_spaces[h].locate(base_space, time)?;
            let valid_flags =
                xr::SpaceLocationFlags::ORIENTATION_VALID | xr::SpaceLocationFlags::POSITION_VALID;
            if !aim_loc.location_flags.contains(valid_flags) {
                self.trig_gates[h].reset();
                self.sqz_gates[h].reset();
                self.sec_gates[h].reset();
                self.back_btns[h].reset();
                self.thumb_btns[h].reset();
                continue;
            }

            let aim_pose = xr_pose_to_pose3d(aim_loc.pose);
            let grip_pose = if grip_loc.location_flags.contains(valid_flags) {
                xr_pose_to_pose3d(grip_loc.pose)
            } else {
                aim_pose
            };
            let aim_dir = quat_forward(aim_pose.orientation);

            let tv = self.trigger_val.state(session, sp)?;
            let tc = self.trigger_click.state(session, sp)?;
            let trig_raw = if tv.is_active {
                tv.current_state
            } else if tc.is_active && tc.current_state {
                1.0
            } else {
                0.0
            };
            let trig_active = tv.is_active || tc.is_active;
            let trigger_pressed = self.trig_gates[h].update(
                trig_active,
                trig_raw,
                TRIGGER_PRESS_TH,
                TRIGGER_RELEASE_TH,
            );

            let sv = self.squeeze_val.state(session, sp)?;
            let sc = self.squeeze_click.state(session, sp)?;
            let sqz_raw = if sv.is_active {
                sv.current_state
            } else if sc.is_active && sc.current_state {
                1.0
            } else {
                0.0
            };
            let sqz_active = sv.is_active || sc.is_active;
            let squeeze_pressed =
                self.sqz_gates[h].update(sqz_active, sqz_raw, SQUEEZE_PRESS_TH, SQUEEZE_RELEASE_TH);

            let sec = self.secondary_click.state(session, sp)?;
            let secondary_pressed = self.sec_gates[h].update(
                sec.is_active,
                if sec.current_state { 1.0 } else { 0.0 },
                0.5,
                0.5,
            );

            let bk = self.back_click.state(session, sp)?;
            let back_just_pressed = self.back_btns[h].just_pressed(bk.is_active, bk.current_state);

            let th = self.thumb_click.state(session, sp)?;
            let thumb_click_just_pressed =
                self.thumb_btns[h].just_pressed(th.is_active, th.current_state);

            let st = self.thumbstick.state(session, sp)?;
            let stick = if st.is_active {
                [
                    quadratic_stick_curve(st.current_state.x),
                    quadratic_stick_curve(st.current_state.y),
                ]
            } else {
                [0.0, 0.0]
            };

            out[h] = HandControllerInput {
                active: true,
                grip_pose,
                aim_origin: aim_pose.position,
                aim_dir,
                trigger_pressed,
                squeeze_pressed,
                secondary_pressed,
                back_just_pressed,
                thumb_click_just_pressed,
                stick,
            };
        }

        Ok(out)
    }

    pub fn apply_haptics(
        &self,
        session: &xr::Session<xr::Vulkan>,
        events: [Option<HapticEvent>; 2],
    ) {
        for (h, ev) in events.into_iter().enumerate() {
            let Some(ev) = ev else { continue };
            let (amp, dur_ms) = match ev {
                HapticEvent::HoverTick => (0.12, 6),
                HapticEvent::PrimaryClick => (0.35, 12),
                HapticEvent::ContextMenu => (0.45, 18),
                HapticEvent::GrabStart => (0.55, 22),
            };
            let vib = xr::HapticVibration::new()
                .amplitude(amp)
                .duration(xr::Duration::from_nanos(dur_ms * 1_000_000))
                .frequency(xr::FREQUENCY_UNSPECIFIED);
            let _ = self.haptic.apply_feedback(session, self.subpaths[h], &vib);
        }
    }
}

trait BindableAction {
    fn as_binding(&self, path: xr::Path) -> xr::Binding<'_>;
}

impl<T: xr::ActionTy> BindableAction for xr::Action<T> {
    fn as_binding(&self, path: xr::Path) -> xr::Binding<'_> {
        xr::Binding::new(self, path)
    }
}

fn suggest_profile(
    instance: &xr::Instance,
    profile_str: &str,
    bindings: &[(&dyn BindableAction, &str)],
) -> Result<()> {
    let profile = instance.string_to_path(profile_str)?;
    let mut resolved = Vec::with_capacity(bindings.len());
    for (act, path_str) in bindings {
        let p = instance.string_to_path(path_str)?;
        resolved.push(act.as_binding(p));
    }
    instance.suggest_interaction_profile_bindings(profile, &resolved)?;
    Ok(())
}

fn xr_pose_to_pose3d(pose: xr::Posef) -> Pose3D {
    Pose3D {
        position: [pose.position.x, pose.position.y, pose.position.z],
        orientation: [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ],
    }
}

/// Quadratic precision curve with `0.18` radial deadzone (`Constraint 2.2`).
pub fn quadratic_stick_curve(v: f32) -> f32 {
    const DEADZONE: f32 = 0.18;
    if !v.is_finite() || v.abs() <= DEADZONE {
        0.0
    } else {
        let norm = (v.abs() - DEADZONE) / (1.0 - DEADZONE);
        v.signum() * (0.35 * norm + 0.65 * norm * norm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schmitt_trigger_prevents_chatter_and_requires_initial_release() {
        let mut g = SchmittGate::default();
        // Held during initial focus -> must NOT fire until released below 0.35
        assert!(!g.update(true, 0.90, TRIGGER_PRESS_TH, TRIGGER_RELEASE_TH));
        assert!(!g.update(true, 0.30, TRIGGER_PRESS_TH, TRIGGER_RELEASE_TH));
        // Now armed: 0.52 (<0.65) stays false; 0.68 (>=0.65) latches; 0.48 (>0.35) stays latched; 0.30 releases
        assert!(!g.update(true, 0.52, TRIGGER_PRESS_TH, TRIGGER_RELEASE_TH));
        assert!(g.update(true, 0.68, TRIGGER_PRESS_TH, TRIGGER_RELEASE_TH));
        assert!(g.update(true, 0.48, TRIGGER_PRESS_TH, TRIGGER_RELEASE_TH));
        assert!(!g.update(true, 0.30, TRIGGER_PRESS_TH, TRIGGER_RELEASE_TH));
    }
}
