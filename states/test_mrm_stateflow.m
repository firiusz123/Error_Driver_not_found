function results = test_mrm_stateflow(scenarioName)
%TEST_MRM_STATEFLOW Drive the MRM controller chart through scripted scenarios.
%
%   results = test_mrm_stateflow()              % runs all scenarios
%   results = test_mrm_stateflow('highway_mrm') % runs one scenario
%
% Builds a wrapper Simulink model around the MRMController chart, drives
% it with timeseries inputs from a chosen scenario, runs the simulation,
% and prints the sequence of (time, state_id) transitions plus a verdict
% against the expected sequence.
%
% State IDs (must match build_mrm_stateflow.m):
%    1 DriverFullControl    6 MRM_UrbanPullOver
%    2 AttentionWarning     7 MRM_Stopping
%    3 TransitionDemand     8 EM
%    4 MRM_Plan             9 Fault
%    5 MRM_HighwayPullOver 10 Stopped
%
% Scenarios:
%   highway_mrm     -- driver passes out on motorway, full pull-over to shoulder
%   urban_mrm       -- driver passes out on residential street, curbside stop
%   driver_recovers -- driver wakes during AttentionWarning, returns to manual
%   button_override -- driver cancels mid-MRM via button press
%   em_during_mrm   -- imminent collision occurs during MRM execution
%   fault_during_mrm-- sensor health drops during MRM, degraded fallback

    if nargin < 1, scenarioName = 'all'; end

    modelName = 'mrm_test_harness';
    sourceModel = 'mrm_controller';

    % --- Make sure the controller chart exists ---
    if ~exist([sourceModel '.slx'], 'file')
        fprintf('[INFO] %s.slx not found, building it now...\n', sourceModel);
        build_mrm_stateflow(sourceModel);
    end

    % --- Build (or rebuild) the test harness wrapper ---
    buildHarness(modelName, sourceModel);

    % --- Run scenarios ---
    allScenarios = {'highway_mrm','urban_mrm','driver_recovers', ...
                    'button_override','em_during_mrm','fault_during_mrm'};
    if strcmp(scenarioName,'all')
        toRun = allScenarios;
    else
        toRun = {scenarioName};
    end

    results = struct();
    for k = 1:numel(toRun)
        name = toRun{k};
        fprintf('\n========== Scenario: %s ==========\n', name);
        [inputs, expected, tStop] = makeScenario(name);
        out = runOnce(modelName, inputs, tStop);
        verdict = compareSequence(out.stateSeq, expected);
        printSequence(out.stateSeq);
        fprintf('Expected: %s\n', mat2str(expected));
        fprintf('Verdict : %s\n', verdict);
        results.(name) = struct('observed',out.stateSeq, ...
                                'expected',expected, ...
                                'verdict',verdict);
    end
end

%% =====================================================================
%% Harness construction
%% =====================================================================

function buildHarness(modelName, sourceModel)
    if bdIsLoaded(modelName), close_system(modelName, 0); end
    new_system(modelName);
    open_system(modelName);

    % Reference the controller chart's parent model via Model block,
    % or just drop a copy of the chart in. Easiest: copy the Chart block.
    add_block([sourceModel '/Chart'], [modelName '/MRMController']);

    % Inputs: From Workspace blocks, one per chart input.
    inputs = {'incapacitated_flag','cancel_button','driver_torque','ttc', ...
              'ego_speed','road_class','shoulder_available','planner_feasible', ...
              'sensor_health_ok','on_target_lane','curb_distance'};
    for k = 1:numel(inputs)
        bname = inputs{k};
        b = add_block('simulink/Sources/From Workspace', ...
                      [modelName '/' bname], ...
                      'VariableName', bname, ...
                      'Position', [30 30+50*k 130 60+50*k]);
        add_line(modelName, [bname '/1'], ['MRMController/' num2str(k)]);
    end

    % Outputs: Out blocks for state_id (the only one we assert on),
    % plus scopes if you want them. Keep it minimal.
    add_block('simulink/Sinks/To Workspace', ...
              [modelName '/state_log'], ...
              'VariableName', 'state_log', ...
              'SaveFormat', 'Timeseries', ...
              'Position', [600 220 700 250]);
    % state_id is output port 5 in the order declared in build_mrm_stateflow.
    add_line(modelName, 'MRMController/5', 'state_log/1');

    % Solver: discrete, fixed-step, sample time 0.01s.
    set_param(modelName, 'Solver','FixedStepDiscrete', ...
                          'FixedStep','0.01', ...
                          'StopTime','30');
    save_system(modelName);
end

%% =====================================================================
%% Scenario factory
%% =====================================================================

function [inputs, expectedSeq, tStop] = makeScenario(name)
% Returns timeseries for every chart input and the expected state sequence.

    tStop = 30;
    t = (0:0.01:tStop)';
    N = numel(t);

    % Defaults: nominal everything.
    inputs.incapacitated_flag = ts(t, false(N,1));
    inputs.cancel_button      = ts(t, false(N,1));
    inputs.driver_torque      = ts(t, zeros(N,1));
    inputs.ttc                = ts(t, 100*ones(N,1));     % far away
    inputs.ego_speed          = ts(t, 25*ones(N,1));      % 90 km/h
    inputs.road_class         = ts(t, uint8(0)*ones(N,1));% motorway
    inputs.shoulder_available = ts(t, true(N,1));
    inputs.planner_feasible   = ts(t, true(N,1));
    inputs.sensor_health_ok   = ts(t, true(N,1));
    inputs.on_target_lane     = ts(t, false(N,1));
    inputs.curb_distance      = ts(t, 5*ones(N,1));

    switch name
    case 'highway_mrm'
        % Driver passes out at t=2; never recovers. Reaches shoulder at t=20.
        inputs.incapacitated_flag = ts(t, t >= 2);
        inputs.on_target_lane     = ts(t, t >= 20);
        inputs.ego_speed          = ts(t, max(0, 25 - 1.5*max(0,t-16)));
        expectedSeq = [1 2 3 4 5 7 10];

    case 'urban_mrm'
        inputs.road_class         = ts(t, uint8(3)*ones(N,1));   % residential
        inputs.shoulder_available = ts(t, false(N,1));           % no shoulder
        inputs.ego_speed          = ts(t, max(0, 14 - 1.5*max(0,t-18)));
        inputs.incapacitated_flag = ts(t, t >= 2);
        inputs.on_target_lane     = ts(t, t >= 20);
        inputs.curb_distance      = ts(t, max(0.2, 5 - 0.3*max(0,t-18)));
        expectedSeq = [1 2 3 4 6 7 10];

    case 'driver_recovers'
        % Brief incapacitation, driver wakes during AttentionWarning.
        flag = false(N,1);
        flag(t>=2 & t<5) = true;
        inputs.incapacitated_flag = ts(t, flag);
        expectedSeq = [1 2 1];

    case 'button_override'
        % Driver passes out, MRM commits, driver presses cancel mid-MRM.
        inputs.incapacitated_flag = ts(t, t >= 2);
        cancel = false(N,1);
        cancel(t>=18 & t<19) = true;
        inputs.cancel_button = ts(t, cancel);
        % After cancel, model assumes driver alert -- clear flag.
        flag = t >= 2 & t < 18;
        inputs.incapacitated_flag = ts(t, flag);
        expectedSeq = [1 2 3 4 5 1];

    case 'em_during_mrm'
        % MRM in progress, sudden imminent collision at t=15.
        inputs.incapacitated_flag = ts(t, t >= 2);
        ttc = 100*ones(N,1);
        ttc(t>=15 & t<17) = 1.5;     % below TTC_EM
        inputs.ttc = ts(t, ttc);
        % After EM clears, still incapacitated -> back to MRM_Plan.
        expectedSeq = [1 2 3 4 5 8 4];

    case 'fault_during_mrm'
        inputs.incapacitated_flag = ts(t, t >= 2);
        health = true(N,1);
        health(t >= 16) = false;
        inputs.sensor_health_ok = ts(t, health);
        inputs.ego_speed        = ts(t, max(0, 25 - 1.5*max(0,t-16)));
        expectedSeq = [1 2 3 4 5 9 10];

    otherwise
        error('Unknown scenario: %s', name);
    end
end

%% =====================================================================
%% Simulation runner & sequence comparison
%% =====================================================================

function out = runOnce(modelName, inputs, tStop)
    % Push every input into base workspace as a timeseries.
    fields = fieldnames(inputs);
    for k = 1:numel(fields)
        assignin('base', fields{k}, inputs.(fields{k}));
    end
    set_param(modelName, 'StopTime', num2str(tStop));
    simOut = sim(modelName, 'ReturnWorkspaceOutputs','on');

    log = simOut.get('state_log');
    [vals, ia] = unique(log.Data, 'stable');
    times = log.Time(ia);
    out.stateSeq = vals(:)';
    out.times    = times(:)';
end

function tsObj = ts(t, data)
    tsObj = timeseries(double(data), t);
end

function v = compareSequence(observed, expected)
    if isequal(observed, expected)
        v = 'PASS';
    else
        v = 'FAIL (sequence mismatch)';
    end
end

function printSequence(seq)
    names = {'DFC','AW','TD','Plan','HwyPull','UrbPull','Stopping', ...
             'EM','Fault','Stopped'};
    parts = arrayfun(@(s) sprintf('%d:%s', s, names{s}), seq, ...
                     'UniformOutput', false);
    fprintf('Observed: %s\n', strjoin(parts, ' -> '));
end