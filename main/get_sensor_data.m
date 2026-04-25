function sensor_data = get_sensor_data(egoVehicle, radars_list, camera_sensor, sim_time)
    % =====================================================================
    % PERCEPTION MODULE
    % Returns a struct with:
    %   .is_right_lane_safe   bool  no closing traffic in right-adjacent lane
    %   .is_shoulder_detected bool  closest right boundary is Solid
    %   .closest_threat_dist  m     distance to closest relevant threat
    %   .lane_offsets         vec   lateral offsets (m) of every visible
    %                               lane boundary, ego frame, ascending
    % =====================================================================

    sensor_data = struct();
    sensor_data.is_right_lane_safe   = true;
    sensor_data.is_shoulder_detected = false;
    sensor_data.closest_threat_dist  = inf;
    sensor_data.lane_offsets         = [];

    % ---------------------------------------------------------------------
    % PART 1: CAMERA -- lane boundary geometry & shoulder-line detection
    % ---------------------------------------------------------------------
    if ~isempty(camera_sensor)
        ground_truth_lanes = laneBoundaries(egoVehicle, ...
            'XDistance', linspace(0, 50, 51));

        % visionDetectionGenerator in 'Lanes only' mode returns 3 outputs.
        % env_sim.m line 56 confirms this signature.
        [lane_dets, ~, ~] = camera_sensor(ground_truth_lanes, sim_time);

        right_line_offset = -inf;
        right_line_type   = 'Unmarked';
        offsets = [];

        for i = 1:length(lane_dets)
            d = lane_dets(i);
            if ~(isprop(d,'LateralOffset') || isfield(d,'LateralOffset'))
                continue;
            end
            offset = d.LateralOffset;
            offsets(end+1) = offset; %#ok<AGROW>

            % Closest line on the right side (largest negative offset)
            if offset < 0 && offset > right_line_offset
                right_line_offset = offset;
                if isprop(d,'BoundaryType') || isfield(d,'BoundaryType')
                    right_line_type = char(d.BoundaryType);
                end
            end
        end

        sensor_data.lane_offsets = sort(offsets, 'descend');
        sensor_data.is_shoulder_detected = strcmp(right_line_type, 'Solid');
    end

    % ---------------------------------------------------------------------
    % PART 2: RADARS -- collision and gap acceptance
    % ---------------------------------------------------------------------
    targets = targetPoses(egoVehicle);
    if isempty(targets)
        return;
    end

    % Right-adjacent lane Y bounds (ego at Y=1.5, right lane Y in [-3.5, 0.5])
    right_lane_y_min       = -3.5;
    right_lane_y_max       =  0.5;
    safe_ttc_threshold     =  4.0;
    critical_bubble_radius =  2.0;

    for r = 1:length(radars_list)
        [detections, ~] = radars_list{r}(targets, sim_time);
        % main.m passes radars in order {radar_RR, radar_RL, radar_R}.
        % Only the first two are rear-facing — their X axis points astern,
        % so the "X<0 && Vx>0 means approaching from behind" predicate
        % is only meaningful for them.
        % TODO: pass radar mounting metadata so we don't rely on list order.
        is_rear = (r <= 2);

        for i = 1:length(detections)
            meas = detections{i}.Measurement;
            X = meas(1); Y = meas(2); Vx = meas(4);

            true_dist = sqrt(X^2 + Y^2);
            if true_dist < critical_bubble_radius
                sensor_data.is_right_lane_safe = false;
                if true_dist < sensor_data.closest_threat_dist
                    sensor_data.closest_threat_dist = true_dist;
                end
                continue;
            end

            if is_rear && Y >= right_lane_y_min && Y <= right_lane_y_max
                if abs(X) < sensor_data.closest_threat_dist
                    sensor_data.closest_threat_dist = abs(X);
                end
                if X < 0 && Vx > 0  % approaching from behind
                    ttc = abs(X) / Vx;
                    if ttc < safe_ttc_threshold
                        sensor_data.is_right_lane_safe = false;
                    end
                end
            end
        end
    end
end
