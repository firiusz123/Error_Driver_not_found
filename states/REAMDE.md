# Finite State Machine for controler selection

The general idea of this finite state machine is to 

## Inputs

**`incapacitated_flag`** *(bool)*
DMS classifier output — true when the driver-monitoring model (PERCLOS-dominant, fused with gaze/head-pose/torque) considers the driver unable to drive. Migrate later to `dms_confidence: double ∈ [0,1]` and apply hysteresis at the chart boundary.

**`cancel_button`** *(bool, event-like)*
Debounced driver-intent-to-override pulse from the wheel button. Hold-time and edge logic live in an upstream debouncer; the FSM treats it as a clean rising-edge event.

**`driver_torque`** *(double, Nm)*
Steering-wheel torque applied by the driver. Used as evidence of conscious override during `TransitionDemand`; compared against `TORQUE_OVERRIDE`.

**`ttc`** *(double, s)*
Time-to-collision estimate from forward perception (camera/radar fusion). Drives the EM transition.

**`ego_speed`** *(double, m/s)*
Longitudinal speed from wheel odometry / GNSS. Used only by `Stopping → Stopped` and `Fault → Stopped` transitions.

**`road_class`** *(uint8 enum)*
0 motorway, 1 trunk, 2 primary, 3 residential. Determined upstream by fusing OpenDRIVE/HD-map lookup with GPS.

**`shoulder_available`** *(bool)*
Perception confirms a paved shoulder of sufficient width is reachable within the planning horizon. Only consulted on motorway/trunk classes.

**`planner_feasible`** *(bool)*
The MRM trajectory planner can produce a kinematically-feasible pull-over path under current dynamic constraints. False triggers `Fault`.

**`sensor_health_ok`** *(bool)*
Aggregate BIST flag from camera/radar/lidar/GPS.

**`on_target_lane`** *(bool)*
Lateral controller has settled within the target lane (shoulder for highway, curbside for urban) — typically `|lateral_offset| < 0.3 m` sustained for 1 s.

**`curb_distance`** *(double, m)*
Lateral distance to the detected curb. Used only in urban MRM to confirm the vehicle has approached close enough (`< 0.5 m`) before final braking.

# Outputs

**`incapacitated_flag`** *(bool)*
DMS classifier output — true when the driver-monitoring model (PERCLOS-dominant, fused with gaze/head-pose/torque) considers the driver unable to drive. Migrate later to `dms_confidence: double ∈ [0,1]` and apply hysteresis at the chart boundary.

**`cancel_button`** *(bool, event-like)*
Debounced driver-intent-to-override pulse from the wheel button. Hold-time and edge logic live in an upstream debouncer; the FSM treats it as a clean rising-edge event.

**`driver_torque`** *(double, Nm)*
Steering-wheel torque applied by the driver. Used as evidence of conscious override during `TransitionDemand`; compared against `TORQUE_OVERRIDE`.

**`ttc`** *(double, s)*
Time-to-collision estimate from forward perception (camera/radar fusion). Drives the EM transition.

**`ego_speed`** *(double, m/s)*
Longitudinal speed from wheel odometry / GNSS. Used only by `Stopping → Stopped` and `Fault → Stopped` transitions.

**`road_class`** *(uint8 enum)*
0 motorway, 1 trunk, 2 primary, 3 residential. Determined upstream by fusing OpenDRIVE/HD-map lookup with GPS.

**`shoulder_available`** *(bool)*
Perception confirms a paved shoulder of sufficient width is reachable within the planning horizon. Only consulted on motorway/trunk classes.

**`planner_feasible`** *(bool)*
The MRM trajectory planner can produce a kinematically-feasible pull-over path under current dynamic constraints. False triggers `Fault`.

**`sensor_health_ok`** *(bool)*
Aggregate BIST flag from camera/radar/lidar/GPS.

**`on_target_lane`** *(bool)*
Lateral controller has settled within the target lane (shoulder for highway, curbside for urban) — typically `|lateral_offset| < 0.3 m` sustained for 1 s.

**`curb_distance`** *(double, m)*
Lateral distance to the detected curb. Used only in urban MRM to confirm the vehicle has approached close enough (`< 0.5 m`) before final braking.