function sensor_data = get_sensor_data(egoVehicle, radars_list, camera_sensor, sim_time)
    % =====================================================================
    % FULL PERCEPTION MODULE (Fixed Output Indexing)
    % =====================================================================
    
    sensor_data = struct();
    sensor_data.is_right_lane_safe = true;           
    sensor_data.is_shoulder_detected = false; 
    sensor_data.closest_threat_dist = inf;

    % ---------------------------------------------------------------------
    % PART 1: CAMERA VISION (SHOULDER DETECTION)
    % ---------------------------------------------------------------------
    if ~isempty(camera_sensor)
        % Create ground truth for the camera (50m ahead)
        ground_truth_lanes = laneBoundaries(egoVehicle, 'XDistance', linspace(0, 50, 51));
        
        % FIX: Camera in 'Lanes only' mode returns exactly 2 outputs:
        % [detections, isValidTime]
        [lane_dets, ~] = camera_sensor(ground_truth_lanes, sim_time);
        
        % Check how many lanes we ACTUALLY detected using length()
        num_actual_lanes = length(lane_dets);
        
        right_line_offset = -inf;
        right_line_type = 'Unmarked';
        
        for i = 1:num_actual_lanes
            % Failsafe: check if the field exists before accessing
            if isfield(lane_dets(i), 'LateralOffset') || isprop(lane_dets(i), 'LateralOffset')
                offset = lane_dets(i).LateralOffset;
                
                % Find the closest line on the right side (negative Y)
                if offset < 0 && offset > right_line_offset
                    right_line_offset = offset;
                    right_line_type = lane_dets(i).BoundaryType;
                end
            end
        end
        
        % If the closest right boundary is a solid line -> it's a shoulder
        if strcmp(right_line_type, 'Solid')
            sensor_data.is_shoulder_detected = true;
        end
    end

    % ---------------------------------------------------------------------
    % PART 2: RADARS (COLLISION & GAP ACCEPTANCE)
    % ---------------------------------------------------------------------
    targets = targetPoses(egoVehicle);
    if isempty(targets)
        return; 
    end

    % Your Lane Specs: Ego is at Y=1.5. Right lane is at Y = [-2, 1.5]
    right_lane_y_min = -3.5; % Adjusted for 3.5m lanes
    right_lane_y_max = 0.5;  
    safe_ttc_threshold = 4.0; 
    critical_bubble_radius = 2.0; 

    for r = 1:length(radars_list)
        [detections, ~] = radars_list{r}(targets, sim_time);
        numDets = length(detections);

        for i = 1:numDets
            meas = detections{i}.Measurement;
            X = meas(1); Y = meas(2); Vx = meas(4); 
            
            % Euclidean distance for the 2m bubble
            true_dist = sqrt(X^2 + Y^2);
            if true_dist < critical_bubble_radius
                sensor_data.is_right_lane_safe = false;
                if true_dist < sensor_data.closest_threat_dist
                    sensor_data.closest_threat_dist = true_dist;
                end
                continue; 
            end
            
            % Right lane check
            if Y >= right_lane_y_min && Y <= right_lane_y_max
                if abs(X) < sensor_data.closest_threat_dist
                    sensor_data.closest_threat_dist = abs(X);
                end
                if X < 0 && Vx > 0 % Dogania nas
                    ttc = abs(X) / Vx;
                    if ttc < safe_ttc_threshold
                        sensor_data.is_right_lane_safe = false;
                    end
                end
            end
        end
    end
end