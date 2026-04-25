% =========================================================================
% EMERGENCY TAKEOVER SYSTEM -- MASTER ORCHESTRATOR
%
% Pipeline per simulation step:
%
%   env_sim ──▶ get_sensor_data ──┐
%                                 ├─▶ build_chart_inputs
%   driver-mock (keyboard) ───────┘            │
%                                              ▼
%                                          mrm_step  (Stateflow port)
%                                              │
%                                state_id, lat_assist_gain,
%                                long_decel_cmd, hazards_on
%                                              │
%                  mrm_enable = state_id ∈ {4..9}     ──▶ planner_logic
%                                              │              │
%                                              ▼              ▼
%                            ┌─ blended target (lateral + longitudinal) ─┐
%                            │  y_cmd   = (1-g)·y_drv   + g·target_y     │
%                            │  hdg_cmd = (1-g)·0       + g·target_hdg   │
%                            │  v_cmd   = (1-g)·v_drv   + g·target_v     │
%                            │  + FSM decel floor, Stopped override      │
%                            └───────────────────────────────────────────┘
%                                              │
%                                              ▼
%                                  kinematic override on egoVehicle
%
% Driver-mock keys (press while sim window is focused):
%   [s]  hold   apply 5 Nm steering torque -> kicks TD -> DFC if held in TD
%   [c]  press  cancel button -> from any MRM substate back to DFC
%   [w]  hold   force incapacitated_flag false (driver "wakes up")
% =========================================================================
disp('Initializing environment...');

% --- Environment / sensors ------------------------------------------------
[scenario, egoVehicle, camera1, radar_RR, radar_RL, radar_R] = env_sim();
my_radars = {radar_RR, radar_RL, radar_R};
sim_sample_rate = 1 / scenario.SampleTime;
scenario.StopTime = 40;
dt = scenario.SampleTime;

% --- Road geometry constants (from env_sim lanespec) ---------------------
% lanespec(4, 'Width', [3.5 3.5 3.5 3]) with lane types
% [Driving Driving Driving Shoulder]. Computed lane centers at:
LANE_CENTERS = [5.0, 1.5, -2.0];   % three driving lanes, left-to-right
SHOULDER_Y   = -5.25;              % shoulder center (informational)
ROAD_CLASS   = uint8(0);           % 0 = motorway
NOMINAL_V    = 15;                 % m/s baseline driver speed
LANE_TOL     = 0.3;                % m on-target-lane tolerance (R157-style)
LANE_HOLD_T  = 1.0;                % s sustained inside tol -> on_target_lane

% --- FSM, planner, debouncer state ---------------------------------------
fsm_st  = [];                      % mrm_step initializes when empty
fsm_out = struct( ...
    'state_id', uint8(1), 'lat_assist_gain', 0, ...
    'long_decel_cmd', 0, 'hmi_alert_level', uint8(0), 'hazards_on', false);
planner_feasible_prev = true;      % first-step default (no MRM yet)

otl_t_in      = 0;                 % time inside lane tolerance window
otl_sustained = false;             % on_target_lane (debounced)
prev_state_id = uint8(1);          % to reset otl_t_in on FSM state change
driver_lane_y = egoVehicle.Position(2);  % latched while gain ≈ 0

% --- DMS (driver monitoring) hysteresis ----------------------------------
% Schmitt-trigger on drowsy_p: HIGH/LOW thresholds with hold-time on each.
% Reason: PERCLOS-style classifier output is noisy frame-to-frame; we want
% incapacitated_flag to bounce only when the model is *sustained* in either
% direction. Numbers below mirror the suggestion in states/REAMDE.md
% ("migrate to dms_confidence with hysteresis at the chart boundary").
DMS_HIGH      = 0.7;     % set incapacitated when drowsy_p above this for...
DMS_HIGH_HOLD = 1.0;     % ...this many seconds
DMS_LOW       = 0.3;     % clear incapacitated when drowsy_p below this for...
DMS_LOW_HOLD  = 0.5;     % ...this many seconds
incap_high_t  = 0;
incap_low_t   = 0;
incap_state   = false;

dms = dms_init();
dms.enabled = false;  % set false to disable webcam/DMS and use scripted fallback
% --- Visualization + keyboard driver-mock --------------------------------
hFig = figure( ...
    'Name', 'ADAS Emergency Takeover', 'Color', 'w', ...
    'Position', [100 100 1200 720], ...
    'KeyPressFcn',   @(s,e) on_key(s, e, true), ...
    'KeyReleaseFcn', @(s,e) on_key(s, e, false));
chasePlot(egoVehicle, 'Centerline', 'on');
hStatus = annotation('textbox', [0.05 0.90 0.9 0.07], ...
    'String', 'SYSTEM ACTIVE: MONITORING', 'EdgeColor', 'none', ...
    'HorizontalAlignment', 'center', 'FontSize', 13, ...
    'FontWeight', 'bold', 'Color', [0 0.5 0], 'BackgroundColor', 'w');
setappdata(hFig, 'driver', struct( ...
    'torque', 0, 'cancel_button', false, 'incap_override', false));

% --- Webcam preview window (DMS feed shown in parallel) ------------------
% Size placeholder to actual webcam resolution so axes XLim/YLim aren't
% locked to a smaller box (cropping the real frame on first CData write).
hCamFig = []; hCamImg = []; hCamAx = []; hCamTitle = [];
if dms.enabled
    res = sscanf(dms.cam.Resolution, '%dx%d');
    if numel(res) == 2, camW = res(1); camH = res(2);
    else,               camW = 640;    camH = 480;
    end
    hCamFig = figure('Name', 'DMS Camera', 'Color', 'k', ...
        'Position', [100 500 camW camH+40], ...
        'MenuBar', 'none', 'ToolBar', 'none');
    hCamAx = axes('Parent', hCamFig, 'Units', 'normalized', ...
        'Position', [0 0.06 1 0.94]);
    hCamImg = image(zeros(camH, camW, 3, 'uint8'), 'Parent', hCamAx);
    axis(hCamAx, 'image'); axis(hCamAx, 'off');
    hCamTitle = annotation(hCamFig, 'textbox', [0 0 1 0.06], ...
        'String', 'DMS: waiting for first frame...', 'EdgeColor', 'none', ...
        'HorizontalAlignment', 'center', 'Color', 'w', ...
        'BackgroundColor', 'k', 'FontWeight', 'bold');
    figure(hFig);  % keep keyboard focus on the sim window
end

disp('Loop start. Keys: [s]=torque  [c]=cancel  [w]=wake-override');
tic;

% =========================================================================
%                                MAIN LOOP
% =========================================================================
while advance(scenario)
    if ~ishandle(hFig), break; end
    sim_time = scenario.SimulationTime;

    % --- Perception ------------------------------------------------------
    sensor_data = get_sensor_data(egoVehicle, my_radars, camera1, sim_time);

    % --- Driver-model mocks ---------------------------------------------
    driver = getappdata(hFig, 'driver');

    % --- DMS step: webcam capture + ViT inference (4 Hz, rate-limited) --
    dms = dms_step(dms, sim_time);

    % --- DMS camera preview --------------------------------------------
    if ~isempty(hCamImg) && ishandle(hCamImg) && ~isempty(dms.last_frame)
        f = dms.last_frame;
        set(hCamImg, 'CData', f);
        set(hCamAx, 'XLim', [0.5, size(f,2)+0.5], ...
                    'YLim', [0.5, size(f,1)+0.5]);
        set(hCamTitle, 'String', sprintf('DMS  drowsy_p=%.2f  incap=%d', ...
            dms.drowsy_p, incap_state));
    end

    % Schmitt-trigger hysteresis on drowsy_p → incap_state.
    % Deadband [DMS_LOW, DMS_HIGH] holds current state and resets timers.
    if dms.enabled
        if dms.drowsy_p >= DMS_HIGH
            incap_high_t = incap_high_t + dt;
            incap_low_t  = 0;
        elseif dms.drowsy_p <= DMS_LOW
            incap_low_t  = incap_low_t + dt;
            incap_high_t = 0;
        else
            incap_high_t = 0;
            incap_low_t  = 0;
        end
        if incap_high_t >= DMS_HIGH_HOLD
            incap_state = true;
        elseif incap_low_t >= DMS_LOW_HOLD
            incap_state = false;
        end
        incapacitated_flag = incap_state && ~driver.incap_override;
    else
        % Scripted fallback when webcam/network unavailable.
        incapacitated_flag = (sim_time >= 2.0 && sim_time < 30.0) && ...
                             ~driver.incap_override;
    end

    % --- Build FSM inputs -----------------------------------------------
    ego_x = egoVehicle.Position(1);
    ego_y = egoVehicle.Position(2);
    ego_v = norm(egoVehicle.Velocity);

    % TTC: closest_threat_dist / closing_speed. closing_speed isn't exposed
    % per-detection by get_sensor_data yet, so we use ego_v as a worst-case
    % approach proxy -- conservative (tends to under-estimate TTC).
    % TODO: extend get_sensor_data to track per-target relative velocity
    %       and surface true closing speed for this calculation.
    if isfinite(sensor_data.closest_threat_dist)
        ttc = sensor_data.closest_threat_dist / max(ego_v, 0.5);
    else
        ttc = 100;  % nothing relevant in view
    end

    fsm_in = struct( ...
        'incapacitated_flag', incapacitated_flag, ...
        'cancel_button',      driver.cancel_button, ...
        'driver_torque',      driver.torque, ...
        'ttc',                ttc, ...
        'ego_speed',          ego_v, ...
        'road_class',         ROAD_CLASS, ...
        'shoulder_available', sensor_data.is_shoulder_detected, ...
        'planner_feasible',   planner_feasible_prev, ...
        'sensor_health_ok',      true, ...
        'on_target_lane',     otl_sustained, ...
        'curb_distance',      inf);          % TODO: implement for urban scenarios

    % --- Step the FSM ----------------------------------------------------
    [fsm_out, fsm_st] = mrm_step(fsm_in, fsm_st, dt);

    % --- Run the planner (active in MRM, EM, Fault — frozen elsewhere) --
    mrm_enable = ismember(fsm_out.state_id, [4 5 6 7 8 9]);
    [target_y, target_hdg, target_v, ~, planner_feasible_prev, arrived] = ...
        planner_logic(ego_x, ego_y, ego_v, sensor_data.lane_offsets, ...
                      ROAD_CLASS, sensor_data.is_right_lane_safe, ...
                      mrm_enable, fsm_out.state_id);

    % --- Run the planner (active in MRM, EM, Fault — frozen elsewhere) --
    mrm_enable = ismember(fsm_out.state_id, [4 5 6 7 8 9]);
    [target_y, target_hdg, target_v, ~, planner_feasible_prev, arrived] = ...
        planner_logic(ego_x, ego_y, ego_v, sensor_data.lane_offsets, ...
                      ROAD_CLASS, sensor_data.is_right_lane_safe, ...
                      mrm_enable, fsm_out.state_id);

    % =====================================================================
    % HACKATHON BYPASS: Force lane change if external planner fails
    % =====================================================================
    % Check if the FSM requests an emergency pullover (states 4, 5, or 6) 
    % AND perception confirms the right lane is physically safe to enter.
    if ismember(fsm_out.state_id, [4 5 6]) && sensor_data.is_right_lane_safe
        
        % Force lateral assist gain to 1.0 to completely override human driver
        fsm_out.lat_assist_gain = 1.0;
        
        % Define maximum lateral speed in meters per second
        lat_speed = 1.5; 
        
        % Gradually interpolate target_y towards the right lane (Y = -2.0)
        % max() prevents the car from overshooting past the target lane center
        if ego_y > -2.0
            target_y = max(-2.0, ego_y - (lat_speed * dt));
            target_hdg = -2.0; % Apply a slight visual rotation angle in degrees
        else
            target_hdg = 0; % Straighten out once arrived at the lane
        end
    end
    % =====================================================================

    % --- on_target_lane debouncer (feeds back into FSM next step) -------
    % Reset on state change so a stale "arrived" from MRM_Plan can't
    % immediately satisfy HwyPullOver -> Stopping the moment we transition.
    if fsm_out.state_id ~= prev_state_id
        otl_t_in = 0;
    end
    prev_state_id = fsm_out.state_id;
    if arrived
        otl_t_in = otl_t_in + dt;
    else
        otl_t_in = 0;
    end
    otl_sustained = otl_t_in >= LANE_HOLD_T;

    % --- Driver-intent target (latched while assist ≈ 0) ----------------
    if fsm_out.lat_assist_gain < 0.1
        driver_lane_y = nearest_lane_center(ego_y, LANE_CENTERS);
    end
    y_drv   = driver_lane_y;
    hdg_drv = 0;
    v_drv   = NOMINAL_V;

    % --- Blended controller target --------------------------------------
    g = fsm_out.lat_assist_gain;
    y_cmd   = (1 - g) * y_drv   + g * target_y;
    hdg_cmd = (1 - g) * hdg_drv + g * target_hdg;
    v_cmd   = (1 - g) * v_drv   + g * target_v;

    % FSM deceleration floor — applied in MRM_Stopping (7), EM (8), Fault (9).
    % During MRM_Plan / MRM_HighwayPullOver / MRM_UrbanPullOver the planner
    % owns longitudinal speed: it needs forward momentum to translate sideways
    % within maneuver_dist, so we ignore long_decel_cmd in those phases.
    if any(fsm_out.state_id == [7 8 9]) && fsm_out.long_decel_cmd > 0.01
        v_cmd = min(v_cmd, max(0, ego_v - fsm_out.long_decel_cmd * dt));
    end

    % Stopped state: hold position, zero speed
    if fsm_out.state_id == 10
        v_cmd   = 0;
        y_cmd   = ego_y;
        hdg_cmd = 0;
    end
    v_cmd = max(0, v_cmd);

    % --- Kinematic override on the ego vehicle --------------------------
    new_x = ego_x + v_cmd * cosd(hdg_cmd) * dt;
    egoVehicle.Position = [new_x, y_cmd, 0];
    egoVehicle.Velocity = [v_cmd*cosd(hdg_cmd), v_cmd*sind(hdg_cmd), 0];
    egoVehicle.Yaw      = hdg_cmd;

    % --- HMI -------------------------------------------------------------
    update_hmi(hStatus, fsm_out, sim_time);

    drawnow limitrate;
    % wait_time = sim_time - toc;
    % if wait_time > 0, pause(wait_time); end
end

disp('Simulation finished.');

% =========================================================================
%                              LOCAL HELPERS
% =========================================================================

function on_key(fig, evt, isPress)
    d = getappdata(fig, 'driver');
    if isempty(d), return; end
    switch evt.Key
        case 's', d.torque        = isPress * 5.0;   % 5 Nm while held
        case 'c', d.cancel_button = isPress;
        case 'w', d.incap_override = isPress;
    end
    setappdata(fig, 'driver', d);
end

function y = nearest_lane_center(ego_y, centers)
    [~, idx] = min(abs(centers - ego_y));
    y = centers(idx);
end

function update_hmi(hStatus, fsm_out, sim_t)
    names = {'DriverFullControl', 'AttentionWarning', 'TransitionDemand', ...
             'MRM_Plan', 'MRM_HighwayPullOver', 'MRM_UrbanPullOver', ...
             'MRM_Stopping', 'EM', 'Fault', 'Stopped'};
    state_name = names{fsm_out.state_id};
    color = [0 0.5 0]; bg = [1 1 1];
    switch fsm_out.state_id
        case {2,3},     color = [0.5 0.3 0]; bg = [1 0.95 0.7];
        case {4,5,6,7}, color = [1 1 1];     bg = [0.95 0.4 0.1];
        case 8,         color = [1 1 1];     bg = [0.85 0 0];
        case 9,         color = [1 1 1];     bg = [0.4 0 0.5];
        case 10,        color = [1 1 1];     bg = [0.2 0.2 0.2];
    end
    str = sprintf(['t=%5.2fs | %s | gain=%.2f  decel=%.2f m/s²  ' ...
                   'alert=%d  hazards=%d'], ...
                  sim_t, state_name, fsm_out.lat_assist_gain, ...
                  fsm_out.long_decel_cmd, fsm_out.hmi_alert_level, ...
                  fsm_out.hazards_on);
    set(hStatus, 'String', str, 'Color', color, 'BackgroundColor', bg);
end
