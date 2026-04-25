function [target_y, target_heading, target_v, hazard_lights, planner_feasible, arrived] = ...
        planner_logic(ego_x, ego_y, ego_v, lane_offsets, road_class, ...
                      is_lane_safe, mrm_enable, fsm_state)
    % PLANNER_LOGIC: lateral / longitudinal reference for the controller.
    %
    % Multi-phase pull-over:
    %   * MRM_Plan (state 4)             -> hop one lane right at a time
    %                                       until the FSM sees a Solid line
    %                                       on the right and advances us.
    %   * MRM_HighwayPullOver (state 5)  -> S-curve onto the shoulder.
    %   * MRM_UrbanPullOver  (state 6)   -> S-curve to rightmost driving lane.
    %   * MRM_Stopping       (state 7)   -> hold lateral position, decelerate.
    %
    % Re-latch (capture new start_x / start_y / final_target_y) happens on:
    %   - the rising edge of mrm_enable
    %   - any FSM state change while mrm_enable stays true
    %   - in MRM_Plan, when the previous lane-change phase finishes but the
    %     shoulder still isn't visible (we need another hop right)
    %
    % Outputs:
    %   target_y, target_heading, target_v   reference for the controller
    %   hazard_lights                        hazard flashers on
    %   planner_feasible                     latched at first MRM frame
    %   arrived                              S-curve complete and ego is
    %                                        within tolerance of the latched
    %                                        target — feeds on_target_lane.

    persistent start_x start_y final_target_y initial_v maneuver_dist ...
               locked_safe locked_feasible prev_mrm_enable prev_fsm_state ...
               phase_complete

    if isempty(prev_mrm_enable)
        prev_mrm_enable = false; prev_fsm_state = uint8(0);
        start_x = 0; start_y = 0; final_target_y = 0;
        initial_v = 0; maneuver_dist = 100;
        locked_safe = true; locked_feasible = true; phase_complete = false;
    end

    target_y         = ego_y;
    target_heading   = 0;
    target_v         = ego_v;
    hazard_lights    = false;
    planner_feasible = locked_feasible;
    arrived          = false;

    rising_edge   = mrm_enable && ~prev_mrm_enable;
    state_changed = mrm_enable && (fsm_state ~= prev_fsm_state);
    % Stay-in-state re-latch: only meaningful inside MRM_Plan, where the
    % FSM keeps us until shoulder_available flips true. Each completed
    % lane-change makes the next boundary visible to the camera, so the
    % NEXT re-latch picks a new midpoint that hops one lane further right.
    relatch_phase = mrm_enable && fsm_state == 4 && phase_complete;

    if rising_edge || state_changed || relatch_phase
        start_x         = ego_x;
        start_y         = ego_y;
        initial_v       = ego_v;
        locked_safe     = is_lane_safe;
        locked_feasible = ~isempty(lane_offsets);
        phase_complete  = false;

        if locked_safe && locked_feasible
            y_offset = pick_target_offset(lane_offsets, road_class, fsm_state);
            final_target_y = ego_y + y_offset;
            % 4 s lateral move budget at start speed; floor of 5 m/s keeps
            % the maneuver short enough to demo even after TD pre-braking.
            maneuver_dist = max(initial_v, 5.0) * 4.0;
        else
            final_target_y = ego_y;
            maneuver_dist  = 10.0;
        end
        planner_feasible = locked_feasible;
    end

    if mrm_enable
        if locked_safe && locked_feasible
            spatial_progress = max(0.0, min(1.0, (ego_x - start_x) / maneuver_dist));
            smooth_progress  = (1 - cos(pi * spatial_progress)) / 2;
            target_y = start_y + (final_target_y - start_y) * smooth_progress;

            dy_dx = ((final_target_y - start_y) * (pi / 2 / maneuver_dist)) ...
                    * sin(pi * spatial_progress);
            target_heading = atand(dy_dx);

            if fsm_state == 7  % MRM_Stopping
                target_v = initial_v * (1 - spatial_progress);
            else
                % Maintain speed while translating laterally; the FSM's
                % long_decel_cmd is treated as an upper-bound floor only in
                % EM / Fault by the orchestrator, not during lane changes.
                target_v = initial_v;
            end

            if spatial_progress >= 0.95 && abs(ego_y - final_target_y) < 0.3
                arrived = true;
                phase_complete = true;
            end
        else
            target_y       = start_y;
            target_heading = 0;
            target_v       = 0;
        end

        if (initial_v - ego_v) > 5.0 || ~locked_safe || ~locked_feasible
            hazard_lights = true;
        end
    end

    prev_mrm_enable = mrm_enable;
    prev_fsm_state  = fsm_state;
end

% =========================================================================
% Pick a lateral offset based on road class and current FSM phase.
% Same midpoint formula handles both progressive lane hops (MRM_Plan) and
% the final shoulder S-curve (MRM_HighwayPullOver) — only the visible set
% of boundaries differs between the two phases.
% =========================================================================
function y_offset = pick_target_offset(offsets, road_class, fsm_state)
    y_offset = 0;
    if isempty(offsets), return; end

    if road_class == 0  % motorway
        switch fsm_state
        case {4, 5}  % MRM_Plan or MRM_HighwayPullOver -> next-right midpoint
            neg = sort(offsets(offsets < 0), 'descend');
            if numel(neg) >= 2
                y_offset = (neg(1) + neg(2)) / 2;
            elseif numel(neg) == 1
                y_offset = neg(1) - 1.5;   % single-line fallback
            end
        case 7  % MRM_Stopping -> hold lateral position
            y_offset = 0;
        end

    else  % urban (road_class 2 or 3)
        right_lines = sort(offsets(offsets < 0), 'descend');
        if length(right_lines) >= 2
            y_offset = (right_lines(1) + right_lines(2)) / 2;
        elseif length(right_lines) == 1
            y_offset = right_lines(1) - 1.5;
        end
    end
end
