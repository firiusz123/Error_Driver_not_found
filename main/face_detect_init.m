function fd = face_detect_init(detector_path, landmark_path)
%FACE_DETECT_INIT Load BlazeFace + MediaPipe landmark networks for EAR-based DMS.
%
%   fd = face_detect_init()
%   fd = face_detect_init(detector_path, landmark_path)
%
% Returns struct used by face_detect_step():
%   .enabled            true if both ONNX networks loaded successfully
%   .det_net            BlazeFace dlnetwork (256×256 back model)
%   .lm_net             MediaPipe face landmark dlnetwork (192×192)
%   .anchors            [896,2] BlazeFace anchor grid (cached)
%   .ear_p              latest EAR drowsiness probability (0=alert, 1=drowsy)
%   .face_detected      bool — last inference found a face above CONF_THRESH
%   .last_inference_t   sim_time of last inference call
%   .CONF_THRESH        BlazeFace score threshold (default 0.5)
%   .EAR_ALERT          EAR value mapped to ear_p = 0 (alert)
%   .EAR_DROWSY         EAR value mapped to ear_p = 1 (drowsy)

ROOT = '/Users/hubertm/Documents/College/Events/VASC/Error_Driver_not_found';

fd = struct( ...
    'enabled',          false, ...
    'det_net',          [], ...
    'lm_net',           [], ...
    'anchors',          [], ...
    'ear_p',            0.5, ...
    'face_detected',    false, ...
    'last_inference_t', -inf, ...
    'CONF_THRESH',      0.5,  ...
    'EAR_ALERT',        0.30, ...
    'EAR_DROWSY',       0.18);

if nargin < 1 || isempty(detector_path)
    detector_path = fullfile(ROOT, 'face_detector_merged.onnx');
end
if nargin < 2 || isempty(landmark_path)
    landmark_path = fullfile(ROOT, 'face_landmark_detector_merged.onnx');
end

if ~isfile(detector_path)
    warning('FD:NoDetNet', 'face_detector_merged.onnx not found — FD disabled.');
    return;
end
if ~isfile(landmark_path)
    warning('FD:NoLmNet', 'face_landmark_detector_merged.onnx not found — FD disabled.');
    return;
end

try
    fd.det_net = importNetworkFromONNX(detector_path);
    fd.lm_net  = importNetworkFromONNX(landmark_path);
catch ME
    warning('FD:LoadFail', 'Failed to load face nets: %s — FD disabled.', ME.message);
    return;
end

fd.anchors = fd_generate_anchors();
fd.enabled = true;
fprintf('Face detector enabled (BlazeFace + MediaPipe landmarks).\n');
end

% ── helpers (file-local) ──────────────────────────────────────────────────────

function anchors = fd_generate_anchors()
% BlazeFace back model (256×256): 16×16×2 + 8×8×6 = 896 anchors
    anchors = zeros(896, 2, 'single');
    idx = 1;
    for row = 0:15
        for col = 0:15
            for i = 1:2  %#ok<FXSET>
                anchors(idx, 1) = (col + 0.5) / 16;
                anchors(idx, 2) = (row + 0.5) / 16;
                idx = idx + 1;
            end
        end
    end
    for row = 0:7
        for col = 0:7
            for i = 1:6  %#ok<FXSET>
                anchors(idx, 1) = (col + 0.5) / 8;
                anchors(idx, 2) = (row + 0.5) / 8;
                idx = idx + 1;
            end
        end
    end
end
