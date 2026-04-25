% =========================================================================
% EMERGENCY TAKEOVER SYSTEM - KINEMATIC OVERRIDE (V1.3)
% =========================================================================

clear; close all; clc;
disp('Initializing environment...');
% 1. LOAD ENVIRONMENT & FAILSAFE ACTOR EXTRACTION
[scenario, tempVar] = env_sim();
if isa(tempVar, 'drivingScenario')
    scenario = tempVar;
end
egoVehicle = scenario.Actors(1);


% HARD FIX: Force the scenario to run for exactly 15 seconds. 
scenario.StopTime = 15;
% Get the simulation sample rate to synchronize the toolboxes
sim_sample_rate = 1 / scenario.SampleTime;
% 2. CREATE THE NAV TOOLBOX TRAJECTORY (THE BRAIN)
normal_waypoints = [0 1.5 0; 50 1.5 0; 100 1.5 0; 150 1.5 0; 200 1.5 0];
speed = 20; % Constant speed in m/s (54 km/h)
% Calculate Time of Arrival starting from 0
distances = [0; 50; 100; 150; 200];
toa_normal = distances / speed;
% Create the standalone trajectory generator (The Brain)
planner = waypointTrajectory(normal_waypoints, toa_normal, 'SampleRate', sim_sample_rate);

% 3. VISUALIZATION SETUP
hFig = figure('Name', 'ADAS Emergency Takeover', 'Color', 'w');
set(hFig, 'Position', [10, 10, 50, 80]);
plot(scenario);
grid on; axis equal;
hTitle = title('SYSTEM STATUS: MONITORING', 'Color', [0 0.5 0], 'FontSize', 14);
is_emergency_active = false;

% 4. MAIN SIMULATION LOOP
disp('Starting simulation...');

while advance(scenario)
    sim_time = scenario.SimulationTime;
    
    % --- LOGIC MODULE: EMERGENCY DETECTION --- 
    % Trigger emergency protocol after 3 seconds of simulation
    if sim_time >= 3.0 && ~is_emergency_active
        is_emergency_active = true;
        disp('CRITICAL ALERT: Driver unresponsive!');
        set(hTitle, 'String', 'CRITICAL ALERT: PULLING OVER', 'Color', 'r');
        
        % Get exact position at the moment of the event
        start_pos = egoVehicle.Position;
        
        % Define the evasion route (pulling to the right shoulder)
        emergency_path = [
            start_pos;                                     
            start_pos(1)+30, start_pos(2)-1.8, 0;          
            start_pos(1)+70, start_pos(2)-3.6, 0;          
            start_pos(1)+100, start_pos(2)-3.6, 0          
        ];
        
        % Calculate realistic times for the emergency maneuver
        segment_dist = sqrt(sum(diff(emergency_path).^2, 2));
        segment_times = segment_dist / 10; % Average maneuver speed of 10 m/s
        
        % FIX: The new planner's internal clock starts at 0! 
        % We MUST start the TimeOfArrival array at 0, not sim_time.
        toa_emergency = [0; cumsum(segment_times)];
        
        % Overwrite the Brain with the new emergency route
        planner = waypointTrajectory(emergency_path, toa_emergency, 'SampleRate', sim_sample_rate);
    end
    
    % --- KINEMATIC OVERRIDE (Brain controls Body) ---
    % Read coordinates from the Navigation Toolbox
    if ~isDone(planner)
        [pos, orient, vel] = planner();
        
        % Check if the generated position is valid (not NaN) just to be absolutely safe
        if all(isfinite(pos))
            % Manually push the coordinates to the car in the Driving Scenario
            egoVehicle.Position = pos;
            egoVehicle.Velocity = vel;
            
            % Convert quaternion orientation to Euler angles (Yaw, Pitch, Roll)
            angles = eulerd(orient, 'ZYX', 'frame');
            egoVehicle.Yaw = angles(1);
        end
    end
    
    % --- VISUALIZATION: CHASE CAMERA ---
    current_pos = egoVehicle.Position;
    % Make sure we don't try to set invalid limits
    if all(isfinite(current_pos))
        xlim([current_pos(1)-40, current_pos(1)+40]);
        ylim([current_pos(2)-15, current_pos(2)+15]);
    end
    
    drawnow limitrate;
    pause(0.02); 
end

disp('Safety protocol complete.');