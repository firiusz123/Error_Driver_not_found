function [out, st] = mrm_step(in, st, dt)
%MRM_STEP MATLAB port of the mrm_controller Stateflow chart.
%
% Mirrors build_mrm_stateflow.m one-for-one: same state IDs, same transition
% guards, same output assignments. Use this when a script-driven loop would
% rather not pay the cost of sim('mrm_controller', ...) per step.
%
%   [out, st] = mrm_step(in, st, dt)
%
%   in : struct with all 11 chart inputs
%        .incapacitated_flag (bool)   .cancel_button     (bool)
%        .driver_torque      (Nm)     .ttc               (s)
%        .ego_speed          (m/s)    .road_class        (uint8 0..3)
%        .shoulder_available (bool)   .planner_feasible  (bool)
%        .sensor_health_ok   (bool)   .on_target_lane    (bool)
%        .curb_distance      (m)
%   st : persistent state — pass [] on first call to initialize
%   dt : step size in seconds
%
%   out: struct(state_id, lat_assist_gain, long_decel_cmd,
%               hmi_alert_level, hazards_on)
%   st : updated state, feed back next call
%
% Differences from the .slx chart that we add deliberately:
%   * AttentionWarning ramps lat_assist_gain 0 -> AW_PEAK_GAIN linearly so
%     the driver feels a small haptic "nudge" before TransitionDemand.
%   * lat_assist_gain is slew-rate-limited (LAT_GAIN_SLEW per second) so the
%     reset on TD->DFC is not a discontinuous snap. The chart was free to
%     do this in the controller; here we bake it into the FSM output.

    % --- Tunable constants (kept identical to chart) ---
    TD_DURATION     = 10;   % s, R157 standard
    AW_DURATION     = 4;    % s before TD escalates
    TTC_EM          = 2.0;  % s, EM threshold
    DECEL_EM        = 5.0;  % m/s^2
    DECEL_MRM       = 2.5;  % m/s^2
    TORQUE_OVERRIDE = 3.0;  % Nm

    % --- Local additions ---
    LAT_GAIN_SLEW   = 2.0;  % per-second slew rate (both directions)
    AW_PEAK_GAIN    = 0.2;  % AW pre-ramp peak

    % --- State IDs (1..10) ---
    DFC=1; AW=2; TD=3; PLAN=4; HWY=5; URB=6; STOPPING=7; EM=8; FAULT=9; STOPPED=10;

    if isempty(st)
        st.state_id    = DFC;
        st.t_in_state  = 0;
        st.lat_gain    = 0;
        st.decel       = 0;
        st.alert       = uint8(0);
        st.hazards     = false;
    end

    next = st.state_id;

    % ===== Transitions =====
    % Stateflow gives parent (group) transitions priority over substate
    % transitions, so we evaluate them first.

    nominal_set = [DFC AW TD];
    mrm_set     = [PLAN HWY URB STOPPING];

    if any(st.state_id == [nominal_set mrm_set]) && in.ttc < TTC_EM
        next = EM;
    elseif st.state_id == TD && ~in.sensor_health_ok
        next = FAULT;
    elseif any(st.state_id == mrm_set) && ...
           (~in.sensor_health_ok || ~in.planner_feasible)
        next = FAULT;
    elseif any(st.state_id == mrm_set) && in.cancel_button
        next = DFC;
    else
        switch st.state_id
        case DFC
            if in.incapacitated_flag, next = AW; end
        case AW
            if (~in.incapacitated_flag || in.cancel_button)
                next = DFC;
            elseif st.t_in_state >= AW_DURATION && in.incapacitated_flag
                next = TD;
            end
        case TD
            if abs(in.driver_torque) > TORQUE_OVERRIDE || in.cancel_button
                next = DFC;
            elseif st.t_in_state >= TD_DURATION && in.incapacitated_flag
                next = PLAN;
            end
        case PLAN
            if (in.road_class==0 || in.road_class==1) && ...
               in.shoulder_available && in.planner_feasible
                next = HWY;
            elseif (in.road_class==2 || in.road_class==3) && in.planner_feasible
                next = URB;
            end
        case HWY
            if in.on_target_lane, next = STOPPING; end
        case URB
            if in.on_target_lane && in.curb_distance < 0.5, next = STOPPING; end
        case STOPPING
            if in.ego_speed < 0.1, next = STOPPED; end
        case EM
            if in.ttc > TTC_EM*2 &&  in.incapacitated_flag, next = PLAN;
            elseif in.ttc > TTC_EM*2 && ~in.incapacitated_flag, next = DFC;
            end
        case FAULT
            if in.ego_speed < 0.1, next = STOPPED; end
        case STOPPED
            % terminal
        end
    end

    if next ~= st.state_id
        st.t_in_state = 0;
    else
        st.t_in_state = st.t_in_state + dt;
    end
    st.state_id = next;

    % ===== Output computation =====
    % Each state declares a target lat_assist_gain; the actual output is
    % slew-limited so handover is smooth.
    switch st.state_id
    case DFC
        target_gain = 0;
        st.decel    = 0;
        st.alert    = uint8(0);
        st.hazards  = false;
    case AW
        target_gain = AW_PEAK_GAIN * min(1, st.t_in_state / AW_DURATION);
        st.decel    = 0;
        st.alert    = uint8(1);
        st.hazards  = false;
    case TD
        % Sigmoid 0 -> 1 over TD_DURATION (matches chart's during action).
        target_gain = 1.0 / (1.0 + exp(-1.0*(st.t_in_state - TD_DURATION/2)));
        st.decel    = DECEL_MRM * 0.5;
        st.alert    = uint8(3);
        st.hazards  = false;
    case PLAN
        target_gain = 1.0;
        st.decel    = DECEL_MRM;
        st.alert    = uint8(3);
        st.hazards  = true;
    case HWY
        target_gain = 1.0;
        st.decel    = DECEL_MRM;
        st.alert    = uint8(3);
        st.hazards  = true;
    case URB
        target_gain = 1.0;
        st.decel    = DECEL_MRM;
        st.alert    = uint8(3);
        st.hazards  = true;
    case STOPPING
        target_gain = 1.0;
        st.decel    = DECEL_MRM;
        st.alert    = uint8(3);
        st.hazards  = true;
    case EM
        target_gain = 1.0;
        st.decel    = DECEL_EM;
        st.alert    = uint8(3);
        st.hazards  = true;
    case FAULT
        target_gain = 1.0;
        st.decel    = DECEL_MRM;
        st.alert    = uint8(3);
        st.hazards  = true;
    case STOPPED
        % Vehicle parked. Release lateral assist; longitudinal hold handled
        % by the orchestrator (it forces v_cmd=0 in this state).
        target_gain = 0;
        st.decel    = 0;
        st.alert    = uint8(0);
        st.hazards  = true;
    end

    % Slew-limit lat_assist_gain
    delta = target_gain - st.lat_gain;
    max_step = LAT_GAIN_SLEW * dt;
    if abs(delta) > max_step
        st.lat_gain = st.lat_gain + sign(delta)*max_step;
    else
        st.lat_gain = target_gain;
    end

    out.state_id        = uint8(st.state_id);
    out.lat_assist_gain = st.lat_gain;
    out.long_decel_cmd  = st.decel;
    out.hmi_alert_level = st.alert;
    out.hazards_on      = st.hazards;
end
