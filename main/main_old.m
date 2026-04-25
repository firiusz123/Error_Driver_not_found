% =========================================================================
% EMERGENCY TAKEOVER SYSTEM - SYNCHRONIZED REAL-TIME VERSION
% =========================================================================
clear; close all; clc;
disp('Initializing environment...');

% 1. LOAD ENVIRONMENT
[scenario, egoVehicle, camera1, radar_RR, radar_RL, radar_R] = env_sim();
my_radars = {radar_RR, radar_RL, radar_R};
sim_sample_rate = 1 / scenario.SampleTime;
scenario.StopTime = 30; % Wydłużony czas na manewry

% 2. INITIAL PLANNER
speed = 15;
waypoints = [0 1.5 0; 500 1.5 0]; 
planner = waypointTrajectory(waypoints, [0; 500/speed], 'SampleRate', sim_sample_rate);

% 3. VISUALIZATION
hFig = figure('Name', 'ADAS 3D Takeover', 'Color', 'w', 'Position', [100 100 1200 700]);
chasePlot(egoVehicle, 'Centerline', 'on');
hStatus = annotation('textbox', [0.1 0.88 0.8 0.1], 'String', 'SYSTEM ACTIVE: MONITORING', ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 16, 'FontWeight', 'bold', 'Color', [0 0.5 0]);

% State Machine variables
system_state = 0; % 0:Normal, 1:ChangingLane, 2:StoppingOnShoulder

% 4. MAIN LOOP
disp('Starting simulation...');
tic; % Uruchomienie stopera czasu rzeczywistego

while advance(scenario)
    sim_time = scenario.SimulationTime;
    
    % --- PERCEPTION ---
    sensor_data = get_sensor_data(egoVehicle, my_radars, camera1, sim_time);
    
    % MOCK: Zasłabnięcie w 4 sekundzie
    driver_fainted = sim_time >= 4.0;
    
    % --- LOGIC / STATE MACHINE ---
    if driver_fainted
        % Efekt świateł awaryjnych (miganie napisu co 0.5s)
        if mod(floor(sim_time * 2), 2) == 0
            set(hStatus, 'BackgroundColor', [1 0.5 0], 'Color', 'w');
        else
            set(hStatus, 'BackgroundColor', 'r', 'Color', 'w');
        end

        if sensor_data.is_right_lane_safe
            % KROK 1: Jeśli nie widzę pobocza, ale muszę uciekać -> Zmień pas o 3.5m
            if system_state == 0 && ~sensor_data.is_shoulder_detected
                system_state = 1;
                set(hStatus, 'String', 'ZASŁABNIĘCIE: SZUKAM POBOCZA (ZMIANA PASA)');
                pos = egoVehicle.Position;
                % Przesunięcie o jeden pas w prawo (1.5 -> -2.0)
                new_path = [pos; pos(1)+60, -2.0, 0];
                planner = waypointTrajectory(new_path, [0; 60/speed], 'SampleRate', sim_sample_rate);
            
            % KROK 2: Jeśli wykryto linię ciągłą (pobocze) -> Zjedź głębiej i STOP
            elseif (system_state == 0 || system_state == 1) && sensor_data.is_shoulder_detected
                system_state = 2;
                set(hStatus, 'String', 'WYKRYTO POBOCZE: ZATRZYMYWANIE');
                pos = egoVehicle.Position;
                % Zjazd na środek pobocza (Y = -8.5) i hamowanie do zera
                stop_path = [pos; pos(1)+40, -8.5, 0; pos(1)+100, -8.5, 0];
                planner = waypointTrajectory(stop_path, [0; 5; 15], 'SampleRate', sim_sample_rate);
            end
        else
            if system_state < 2
                set(hStatus, 'String', 'PAS ZAJĘTY: OCZEKIWANIE NA LUKĘ');
            end
        end
    end
    
    % --- KINEMATIC OVERRIDE ---
    if ~isDone(planner)
        [p, o, v] = planner();
        if all(isfinite(p))
            egoVehicle.Position = p;
            egoVehicle.Velocity = v;
            ang = eulerd(o, 'ZYX', 'frame');
            egoVehicle.Yaw = ang(1);
        end
    end
    
    % --- SYNCHRONIZACJA CZASU (Płynność) ---
    drawnow limitrate;
    
    % Czekaj, jeśli symulacja idzie szybciej niż czas rzeczywisty
    wait_time = sim_time - toc;
    if wait_time > 0
        pause(wait_time);
    end
end
disp('Simulation finished.');