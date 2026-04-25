function fd = face_detect_step(fd, frame, sim_time)
%FACE_DETECT_STEP BlazeFace detection + MediaPipe landmarks + EAR inference.
%
%   fd = face_detect_step(fd, frame, sim_time)
%
% frame  — uint8 RGB image (any resolution; shared from dms.last_frame)
% Updates:
%   fd.ear_p          0..1 drowsiness probability from EAR (1 = eyes closed)
%   fd.face_detected  true if BlazeFace confidence >= CONF_THRESH
%
% Rate-limited to FD_PERIOD seconds. Falls back to last ear_p on any error.
% ear_p = 0.5 (neutral) when no face detected — avoids false incap triggers.

FD_PERIOD = 0.25;   % 4 Hz, matches dms_step rate

if ~fd.enabled || isempty(frame)
    return;
end
if (sim_time - fd.last_inference_t) < FD_PERIOD
    return;
end

try
    [origH, origW, ~] = size(frame);

    % ── Stage 1: Preprocess for BlazeFace (256×256, normalised to [-1,1]) ──
    img256   = imresize(frame, [256 256]);
    img256   = im2single(img256) * 2 - 1;
    detInput = dlarray(img256, 'SSCB');

    % ── Stage 2: Run face detector ─────────────────────────────────────────
    [box_coords_1, box_coords_2, box_scores_1, box_scores_2] = ...
        predict(fd.det_net, detInput);

    all_scores = [extractdata(sigmoid(box_scores_1))(:); ...
                  extractdata(sigmoid(box_scores_2))(:)];
    c1 = reshape(extractdata(box_coords_1), 512, 16);
    c2 = reshape(extractdata(box_coords_2), 384, 16);
    all_coords = [c1; c2];

    % ── Stage 3: Decode anchor-relative box offsets ────────────────────────
    SCALE      = 256.0;
    decoded_cx = fd.anchors(:,1) + all_coords(:,2) / SCALE;
    decoded_cy = fd.anchors(:,2) + all_coords(:,1) / SCALE;
    decoded_w  = all_coords(:,4) / SCALE;
    decoded_h  = all_coords(:,3) / SCALE;

    [best_score, best_idx] = max(all_scores);
    fd.last_inference_t = sim_time;

    if best_score < fd.CONF_THRESH
        fd.face_detected = false;
        fd.ear_p = 0.5;   % neutral — don't penalise incap_high_t when off-camera
        return;
    end

    fd.face_detected = true;
    cx = decoded_cx(best_idx);
    cy = decoded_cy(best_idx);
    bw = decoded_w(best_idx);
    bh = decoded_h(best_idx);

    % ── Stage 4: Crop face ROI with padding ───────────────────────────────
    PAD = 0.25;
    x1 = max(0, cx - bw/2 - PAD*bw);
    y1 = max(0, cy - bh/2 - PAD*bh);
    x2 = min(1, cx + bw/2 + PAD*bw);
    y2 = min(1, cy + bh/2 + PAD*bh);

    px1 = max(1,     round(x1 * origW));
    py1 = max(1,     round(y1 * origH));
    px2 = min(origW, round(x2 * origW));
    py2 = min(origH, round(y2 * origH));

    face_crop = frame(py1:py2, px1:px2, :);

    % ── Stage 5: Landmark detection (192×192, [-1,1]) ─────────────────────
    face192 = imresize(face_crop, [192 192]);
    face192 = im2single(face192) * 2 - 1;
    lmInput = dlarray(face192, 'SSCB');

    [~, lm_landmarks] = predict(fd.lm_net, lmInput);
    landmarks = squeeze(extractdata(lm_landmarks));   % [468, 3]

    % ── Stage 6: EAR (Eye Aspect Ratio) → drowsiness probability ──────────
    % MediaPipe canonical face mesh indices (0-based → +1 for MATLAB)
    LEFT_EYE  = [33, 160, 158, 133, 153, 144] + 1;
    RIGHT_EYE = [362, 385, 387, 263, 373, 380] + 1;

    left_EAR  = fd_ear(landmarks, LEFT_EYE);
    right_EAR = fd_ear(landmarks, RIGHT_EYE);
    avg_EAR   = (left_EAR + right_EAR) / 2;

    % Linear map: EAR_ALERT → ear_p=0 (alert), EAR_DROWSY → ear_p=1 (drowsy)
    fd.ear_p = 1 - max(0, min(1, ...
        (avg_EAR - fd.EAR_DROWSY) / (fd.EAR_ALERT - fd.EAR_DROWSY)));

catch ME
    warning('FD:StepFail', 'Face detect error at t=%.2fs: %s', sim_time, ME.message);
end
end

% ── helpers (file-local) ──────────────────────────────────────────────────────

function ear = fd_ear(lm, indices)
    pts = lm(indices, 1:2);
    A   = norm(pts(2,:) - pts(6,:));
    B   = norm(pts(3,:) - pts(5,:));
    C   = norm(pts(1,:) - pts(4,:));
    ear = (A + B) / (2.0 * C);
end
