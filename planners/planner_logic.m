function [target_y, target_heading, target_v, hazard_lights] = mrm_planner(ego_x, ego_y, ego_v, lane_offsets, road_class, is_lane_safe, mrm_enable)
    % MRM_PLANNER: Generates spatial path and speed targets for Stanley and PI controllers
    
    % Persistent variables hold state between Simulink time steps
    persistent start_x start_y final_target_y initial_v maneuver_dist locked_safe prev_mrm_enable
    
    % Initialization for the very first step
    if isempty(prev_mrm_enable)
        prev_mrm_enable = false;
        start_x = 0; start_y = 0; final_target_y = 0; 
        initial_v = 0; maneuver_dist = 100; locked_safe = true;
    end

    % Default outputs (passthrough if MRM is not active)
    target_y = ego_y;
    target_heading = 0;
    target_v = ego_v;
    hazard_lights = false;

    % --- 1. RISING EDGE DETECTION: Lock in parameters when MRM starts ---
    if mrm_enable && ~prev_mrm_enable
        start_x = ego_x;
        start_y = ego_y;
        initial_v = ego_v;
        
        % Latch the safety decision. We don't want to change our minds mid-maneuver.
        locked_safe = is_lane_safe; 
        
        if locked_safe
            % Lane is safe: Calculate pull-over destination offset
            y_offset = determine_target_offset(lane_offsets, road_class);
            final_target_y = ego_y + y_offset;
            
            % Lookahead distance based on initial speed (~6 seconds to change lanes)
            maneuver_dist = max(initial_v, 5.0) * 6.0; 
        else
            % Lane NOT safe: Target our current exact lateral position
            final_target_y = ego_y;
            maneuver_dist = 10.0; % Arbitrary small distance, as we just hold straight
        end
    end
    
    % --- 2. EXECUTION: Generate targets every frame MRM is active ---
    if mrm_enable
        if locked_safe
            % PULL-OVER MANEUVER (Safe)
            % Calculate how far forward we've driven (0.0 to 1.0)
            spatial_progress = max(0.0, min(1.0, (ego_x - start_x) / maneuver_dist));
            
            % Smooth S-Curve for Stanley to track
            smooth_progress = (1 - cos(pi * spatial_progress)) / 2;
            target_y = start_y + (final_target_y - start_y) * smooth_progress;
            
            % Heading derivative for Stanley target heading
            dy_dx = ((final_target_y - start_y) * (pi / 2 / maneuver_dist)) * sin(pi * spatial_progress);
            target_heading = atand(dy_dx);
            
            % Target speed smoothly decreases over the distance
            target_v = initial_v - (initial_v * spatial_progress);
            
        else
            % IN-LANE EMERGENCY STOP (Not Safe)
            target_y = start_y;      % Hold the wheel dead straight
            target_heading = 0;      % No heading angle changes
            target_v = 0;            % Command 0 m/s immediately (max braking)
        end
        
        % --- 3. EMERGENCY LIGHTS LOGIC ---
        % Light up if the velocity difference from MRM start is large (> 5 m/s or ~18 km/h),
        % OR if we are forced to do a dangerous in-lane stop.
        vel_difference = initial_v - ego_v;
        if vel_difference > 5.0 || ~locked_safe
            hazard_lights = true;
        end
    end
    
    % Save state for next step
    prev_mrm_enable = mrm_enable;
end

% --- Helper Function ---
function y_offset = determine_target_offset(offsets, road_class)
    % Updated for strict 3.0m lane widths
    half_lane = 1.5; 
    y_offset = 0; % Fallback if sensors fail
    
    if isempty(offsets)
        return;
    end
    
    if road_class == 0 % Motorway (Target the shoulder)
        % Find the rightmost line (most negative offset)
        rightmost = min(offsets);
        % Center of the shoulder is exactly 1.5m right of the rightmost line
        y_offset = rightmost - half_lane; 
        
    else % Urban (Target the rightmost driving lane)
        right_lines = offsets(offsets < 0);
        
        if length(right_lines) >= 2
            % If we see both boundaries of the target lane, the midpoint is safest
            right_lines = sort(right_lines, 'descend');
            y_offset = (right_lines(1) + right_lines(2)) / 2;
        elseif length(right_lines) == 1
            % If we only see the left boundary of the target lane, offset by 1.5m
            y_offset = right_lines(1) - half_lane; 
        end
    end
end