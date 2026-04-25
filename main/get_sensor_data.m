function sensor_data = get_sensor_data(egoVehicle, radars_list, camera_sensor, sim_time)
    % =====================================================================
    % FULL PERCEPTION MODULE (Radars + Camera Lane Detection)
    % =====================================================================
    
    % 1. INITIALIZE OUTPUT STRUCTURE
    sensor_data = struct();
    sensor_data.is_right_lane_safe = true;           
    sensor_data.is_shoulder_detected = false; % Default assumption: no shoulder
    sensor_data.closest_threat_dist = inf;

    % ---------------------------------------------------------------------
    % PART 1: CAMERA VISION (SHOULDER DETECTION)
    % ---------------------------------------------------------------------
    if ~isempty(camera_sensor)
        % To feed the camera, we must first extract the mathematical "ground truth" 
        % of the road from the scenario, up to 40 meters ahead.
        ground_truth_lanes = laneBoundaries(egoVehicle, 'XDistance', linspace(0, 40, 40));
        
        % We also need target poses (even if we ignore objects from camera here)
        targets = targetPoses(egoVehicle);
        
        % Step the camera. It returns objects (which we ignore using '~') and lanes.
        [~, ~, lane_dets, num_lanes] = camera_sensor(targets, ground_truth_lanes, sim_time);
        
        % Algorithm to find the immediate right lane line
        right_line_offset = -inf;
        right_line_type = 'Unmarked';
        
        % Loop through all detected lines
        for i = 1:num_lanes
            offset = lane_dets(i).LateralOffset;
            
            % If the line is to our right (offset < 0) and is the closest one
            if offset < 0 && offset > right_line_offset
                right_line_offset = offset;
                right_line_type = lane_dets(i).BoundaryType;
            end
        end
        
        % Check if the closest right line is Solid (indicating a shoulder)
        if strcmp(right_line_type, 'Solid')
            sensor_data.is_shoulder_detected = true;
        end
    end

    % ---------------------------------------------------------------------
    % PART 2: RADARS (COLLISION & GAP ACCEPTANCE)
    % ---------------------------------------------------------------------
    targets = targetPoses(egoVehicle);
    if isempty(targets)
        return; % Road is completely empty
    end

    % Logic Parameters for 3-meter lanes
    right_lane_y_min = -4.6;  
    right_lane_y_max = -1.4;  
    safe_ttc_threshold = 4.0; 
    critical_bubble_radius = 2.0; % Absolute 2-meter dead zone

    for r = 1:length(radars_list)
        [detections, numDets] = radars_list{r}(targets, sim_time);

        for i = 1:numDets
            meas = detections{i}.Measurement;
            X = meas(1);  
            Y = meas(2);  
            Vx = meas(4); 
            
            % RULE 1: THE CRITICAL 2-METER BUBBLE
            true_distance = sqrt(X^2 + Y^2);
            if true_distance < critical_bubble_radius
                sensor_data.is_right_lane_safe = false;
                if true_distance < sensor_data.closest_threat_dist
                    sensor_data.closest_threat_dist = true_distance;
                end
                continue; 
            end
            
            % RULE 2: RIGHT LANE GAP ACCEPTANCE
            if Y >= right_lane_y_min && Y <= right_lane_y_max
                if abs(X) < sensor_data.closest_threat_dist
                    sensor_data.closest_threat_dist = abs(X);
                end
                
                % If vehicle is behind us and catching up
                if X < 0 && Vx > 0
                    ttc = abs(X) / Vx;
                    if ttc < safe_ttc_threshold
                        sensor_data.is_right_lane_safe = false;
                    end
                end
            end
        end
    end
end