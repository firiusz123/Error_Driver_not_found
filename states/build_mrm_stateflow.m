function build_mrm_stateflow(modelName)
%BUILD_MRM_STATEFLOW Programmatically construct an L3 MRM controller chart.
%
%   build_mrm_stateflow()              % uses default name 'mrm_controller'
%   build_mrm_stateflow('my_model')    % custom model name
%
% Creates a Simulink model containing a single Stateflow chart that
% implements the Minimum Risk Maneuver (MRM) state logic for a driver-
% incapacitation scenario, broadly aligned with UNECE R157 (ALKS).
%
% State hierarchy:
%
%   NOMINAL                                MRM
%     +-- DriverFullControl  (default)       +-- MRM_Plan
%     +-- AttentionWarning                   +-- MRM_HighwayPullOver
%     +-- TransitionDemand                   +-- MRM_UrbanPullOver
%                                            +-- MRM_Stopping
%
%   EM       (top-level, preempts NOMINAL and MRM on imminent collision)
%   Fault    (top-level, degraded fallback when sensors/planner fail)
%   Stopped  (terminal hazard state)
%
% Outputs feed downstream lateral/longitudinal controllers and HMI:
%   lat_assist_gain  in [0,1] -- sigmoid ramp during TransitionDemand
%   long_decel_cmd   in m/s^2 -- 0 nominal, ~2.5 MRM, 5.0 EM
%   hmi_alert_level  in {0,1,2,3}
%   hazards_on       boolean
%   state_id         uint8 -- for DSSAD logging and downstream dispatch
%
% Action language: MATLAB. Switch ch.ActionLanguage to 'C' for code-gen
% targets that don't accept the MATLAB action subset.
%
% Example:
%   build_mrm_stateflow();
%   open_system('mrm_controller');

    if nargin < 1, modelName = 'mrm_controller'; end

    %% --- Create empty model + chart ----------------------------------
    if bdIsLoaded(modelName)
        close_system(modelName, 0);
    end
    sfnew(modelName);
    rt = sfroot;
    m  = find(rt, '-isa', 'Stateflow.Machine', 'Name', modelName);
    ch = find(m,  '-isa', 'Stateflow.Chart');
    ch.Name = 'MRMController';
    ch.ActionLanguage = 'MATLAB';

    %% --- Data declarations ------------------------------------------
    % Inputs from perception, DMS, GPS/HD-map, planner, vehicle bus
    declareData(ch, 'incapacitated_flag', 'Input',  'boolean');
    declareData(ch, 'cancel_button',      'Input',  'boolean');
    declareData(ch, 'driver_torque',      'Input',  'double');
    declareData(ch, 'ttc',                'Input',  'double');
    declareData(ch, 'ego_speed',          'Input',  'double');
    declareData(ch, 'road_class',         'Input',  'uint8');   % 0=highway 1=trunk 2=primary 3=residential
    declareData(ch, 'shoulder_available', 'Input',  'boolean');
    declareData(ch, 'planner_feasible',   'Input',  'boolean');
    declareData(ch, 'sensor_health_ok',   'Input',  'boolean');
    declareData(ch, 'on_target_lane',     'Input',  'boolean');
    declareData(ch, 'curb_distance',      'Input',  'double');

    % Outputs to controllers and HMI
    declareData(ch, 'lat_assist_gain',    'Output', 'double');
    declareData(ch, 'long_decel_cmd',     'Output', 'double');
    declareData(ch, 'hmi_alert_level',    'Output', 'uint8');
    declareData(ch, 'hazards_on',         'Output', 'boolean');
    declareData(ch, 'state_id',           'Output', 'uint8');

    % Tunable constants in the chart and can be tweaked from the model workspace.
    declareConstant(ch, 'TD_DURATION',     '10');     % seconds, R157 standard
    declareConstant(ch, 'AW_DURATION',     '4');      % seconds before TD escalates
    declareConstant(ch, 'TTC_EM',          '2.0');    % seconds-to-collision threshold
    declareConstant(ch, 'DECEL_EM',        '5.0');    % m/s^2 emergency decel
    declareConstant(ch, 'DECEL_MRM',       '2.5');    % m/s^2 nominal MRM decel
    declareConstant(ch, 'TORQUE_OVERRIDE', '3.0');    % Nm threshold during TD

    %% --- States: NOMINAL superstate ---------------------------------
    sNom = Stateflow.State(ch);
    sNom.Name = 'NOMINAL';
    sNom.Position = [30 30 480 360];

    sDFC = Stateflow.State(sNom);
    sDFC.Name = 'DriverFullControl';
    sDFC.LabelString = sprintf([ ...
        'DriverFullControl\n' ...
        'entry:\n' ...
        '  lat_assist_gain = 0;\n' ...
        '  long_decel_cmd  = 0;\n' ...
        '  hmi_alert_level = uint8(0);\n' ...
        '  hazards_on      = false;\n' ...
        '  state_id        = uint8(1);']);
    sDFC.Position = [50 70 200 100];

    sAW = Stateflow.State(sNom);
    sAW.Name = 'AttentionWarning';
    sAW.LabelString = sprintf([ ...
        'AttentionWarning\n' ...
        'entry:\n' ...
        '  hmi_alert_level = uint8(1);\n' ...
        '  state_id        = uint8(2);']);
    sAW.Position = [280 70 200 100];

    sTD = Stateflow.State(sNom);
    sTD.Name = 'TransitionDemand';
    % Sigmoid lateral-assist ramp: 1 / (1 + exp(-k*(t - t0)))
    % with k = 1.0 and t0 = TD_DURATION/2 => ~0 at entry, ~1 at TD timeout.
    % `temporalCount(sec)` returns elapsed time in current state.
    sTD.LabelString = sprintf([ ...
        'TransitionDemand\n' ...
        'entry:\n' ...
        '  hmi_alert_level = uint8(3);\n' ...
        '  long_decel_cmd  = DECEL_MRM * 0.5;\n' ...
        '  state_id        = uint8(3);\n' ...
        'during:\n' ...
        '  lat_assist_gain = 1.0/(1.0 + exp(-1.0*(temporalCount(sec) - TD_DURATION/2)));']);
    sTD.Position = [165 200 200 130];

    %% --- States: MRM superstate -------------------------------------
    sMRM = Stateflow.State(ch);
    sMRM.Name = 'MRM';
    sMRM.Position = [540 30 480 360];

    sPlan = Stateflow.State(sMRM);
    sPlan.Name = 'MRM_Plan';
    sPlan.LabelString = sprintf([ ...
        'MRM_Plan\n' ...
        'entry:\n' ...
        '  lat_assist_gain = 1.0;\n' ...
        '  long_decel_cmd  = DECEL_MRM;\n' ...
        '  hazards_on      = true;\n' ...
        '  state_id        = uint8(4);']);
    sPlan.Position = [560 70 200 80];

    sHwy = Stateflow.State(sMRM);
    sHwy.Name = 'MRM_HighwayPullOver';
    sHwy.LabelString = sprintf([ ...
        'MRM_HighwayPullOver\n' ...
        'entry: state_id = uint8(5);']);
    sHwy.Position = [800 70 200 80];

    sUrb = Stateflow.State(sMRM);
    sUrb.Name = 'MRM_UrbanPullOver';
    sUrb.LabelString = sprintf([ ...
        'MRM_UrbanPullOver\n' ...
        'entry: state_id = uint8(6);']);
    sUrb.Position = [800 180 200 80];

    sStopping = Stateflow.State(sMRM);
    sStopping.Name = 'MRM_Stopping';
    sStopping.LabelString = sprintf([ ...
        'MRM_Stopping\n' ...
        'entry:\n' ...
        '  long_decel_cmd = DECEL_MRM;\n' ...
        '  state_id       = uint8(7);']);
    sStopping.Position = [560 290 200 80];

    %% --- States: top-level EM, Fault, Stopped -----------------------
    sEM = Stateflow.State(ch);
    sEM.Name = 'EM';
    sEM.LabelString = sprintf([ ...
        'EM\n' ...
        'entry:\n' ...
        '  long_decel_cmd  = DECEL_EM;\n' ...
        '  hmi_alert_level = uint8(3);\n' ...
        '  hazards_on      = true;\n' ...
        '  state_id        = uint8(8);']);
    sEM.Position = [30 420 280 100];

    sFault = Stateflow.State(ch);
    sFault.Name = 'Fault';
    sFault.LabelString = sprintf([ ...
        'Fault\n' ...
        'entry:\n' ...
        '  long_decel_cmd = DECEL_MRM;\n' ...
        '  hazards_on     = true;\n' ...
        '  state_id       = uint8(9);']);
    sFault.Position = [340 420 280 100];

    sStopped = Stateflow.State(ch);
    sStopped.Name = 'Stopped';
    sStopped.LabelString = sprintf([ ...
        'Stopped\n' ...
        'entry:\n' ...
        '  long_decel_cmd = 0;\n' ...
        '  hazards_on     = true;\n' ...
        '  state_id       = uint8(10);']);
    sStopped.Position = [650 420 280 100];

    %% --- Default transition into DriverFullControl ------------------
    dt = Stateflow.Transition(ch);
    dt.Destination       = sDFC;
    dt.DestinationOClock = 0;
    dt.SourceEndPoint    = dt.DestinationEndpoint - [0 30];
    dt.MidPoint          = dt.DestinationEndpoint - [0 15];

    %% --- Transitions inside NOMINAL ---------------------------------
    addTrans(ch, sDFC, sAW,  '[incapacitated_flag]');
    addTrans(ch, sAW,  sDFC, '[!incapacitated_flag || cancel_button]');
    addTrans(ch, sAW,  sTD,  '[after(AW_DURATION,sec) && incapacitated_flag]');
    addTrans(ch, sTD,  sDFC, '[abs(driver_torque) > TORQUE_OVERRIDE || cancel_button]');
    addTrans(ch, sTD,  sPlan,'[after(TD_DURATION,sec) && incapacitated_flag]');

    %% --- Transitions inside MRM -------------------------------------
    addTrans(ch, sPlan, sHwy, ...
        '[(road_class==uint8(0)||road_class==uint8(1)) && shoulder_available && planner_feasible]');
    addTrans(ch, sPlan, sUrb, ...
        '[(road_class==uint8(2)||road_class==uint8(3)) && planner_feasible]');
    addTrans(ch, sHwy,      sStopping, '[on_target_lane]');
    addTrans(ch, sUrb,      sStopping, '[on_target_lane && curb_distance < 0.5]');
    addTrans(ch, sStopping, sStopped,  '[ego_speed < 0.1]');

    % Group transition: any MRM substate -> DFC if button pressed.
    % Per your Q3 answer, button alone is enough (no liveness check).
    addTrans(ch, sMRM, sDFC, '[cancel_button]');

    %% --- Top-level transitions: EM and Fault ------------------------
    % Group transitions on superstates -- fire from any active substate.
    addTrans(ch, sNom, sEM, '[ttc < TTC_EM]');
    addTrans(ch, sMRM, sEM, '[ttc < TTC_EM]');

    % EM recovery: re-enter MRM if still incapacitated, else manual.
    addTrans(ch, sEM, sPlan, '[ttc > TTC_EM*2 &&  incapacitated_flag]');
    addTrans(ch, sEM, sDFC,  '[ttc > TTC_EM*2 && !incapacitated_flag]');

    % Fault paths -- only entered from automated phases.
    addTrans(ch, sTD,  sFault, '[!sensor_health_ok]');
    addTrans(ch, sMRM, sFault, '[!sensor_health_ok || !planner_feasible]');
    addTrans(ch, sFault, sStopped, '[ego_speed < 0.1]');

    %% --- Save and notify --------------------------------------------
    save_system(modelName);
    fprintf('[OK] Built Stateflow chart "%s" in model "%s.slx"\n', ...
            ch.Name, modelName);
    fprintf('Open the chart with:\n');
    fprintf('    open_system(''%s''); sfopen %s\n', modelName, modelName);
end

%% =====================================================================
%% Helpers
%% =====================================================================

function declareData(ch, name, scope, dtype)
    d = Stateflow.Data(ch);
    d.Name              = name;
    d.Scope             = scope;
    d.Props.Type.Method = 'Built-in';
    d.DataType          = dtype;
end

function declareConstant(ch, name, initVal)
    d = Stateflow.Data(ch);
    d.Name                  = name;
    d.Scope                 = 'Constant';
    d.Props.InitialValue    = initVal;
    d.Props.Type.Method     = 'Built-in';
    d.DataType              = 'double';
end

function t = addTrans(ch, src, dst, label)
    t = Stateflow.Transition(ch);
    t.Source      = src;
    t.Destination = dst;
    t.LabelString = label;
end