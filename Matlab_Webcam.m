% =========================================================================
% VASC_simu_code.m
% Webcam + ViT Drowsiness Detection — 100% Native MATLAB
% No Python required
%
% Requirements:
%   - MATLAB R2023b or newer
%   - Deep Learning Toolbox
%   - drowsiness_net.mat  (created by convert_model.m)
%   - USB Webcam Support Package
%
% Controls:
%   Q  — quit
% =========================================================================

% ── CONFIGURATION — edit if your folder is different ─────────────────
PROJECT_DIR = '/Users/hubertm/Documents/College/Events/VASC/Error_Driver_not_found';
NET_PATH    = fullfile(PROJECT_DIR, 'drowsiness_net.mat');

% ImageNet normalisation constants (ViT was trained with these)
IMG_MEAN = reshape([0.485 0.456 0.406], 1, 1, 3);
IMG_STD  = reshape([0.229 0.224 0.225], 1, 1, 3);

% How often to run inference (1 = every frame, 3 = every 3rd frame)
% Higher = faster display, lower = more accurate
RUN_EVERY = 3;

% PERCLOS-style drowsy threshold — flag if drowsy probability exceeds this
DROWSY_THRESHOLD = 0.5;

% ── STEP 1: Load network ──────────────────────────────────────────────
fprintf('Loading drowsiness network...\n');
if ~isfile(NET_PATH)
    error(['drowsiness_net.mat not found in:\n  %s\n\n' ...
           'Run convert_model.m first to create it.'], PROJECT_DIR);
end
data = load(NET_PATH, 'net');
net  = data.net;
fprintf('  Network loaded (85.5M parameters).\n');

% ── STEP 2: Webcam setup ──────────────────────────────────────────────
fprintf('Connecting to webcam...\n');
if isempty(webcamlist())
    error('No webcam detected. Check camera is not in use by another app.');
end

cam = webcam(1);
try, cam.Brightness = 0;   catch, end
try, cam.Contrast   = 128; catch, end
try, cam.Saturation = 128; catch, end
fprintf('  Camera : %s\n', cam.Name);
fprintf('  Resolution: %s\n\n', cam.Resolution);

% ── STEP 3: Figure layout ─────────────────────────────────────────────
fig = figure( ...
    'Name',     'VASC — Drowsiness Detector', ...
    'Position', [80 80 1100 620], ...
    'Color',    [0.08 0.08 0.08], ...
    'NumberTitle', 'off');

% Left panel — live camera feed
ax_cam = axes('Parent', fig, ...
              'Position', [0.01 0.01 0.62 0.98]);
axis(ax_cam, 'off');

% Top right — awake vs drowsy bar chart
ax_bar = axes('Parent', fig, ...
              'Position', [0.66 0.54 0.32 0.40]);
set(ax_bar, 'Color',  [0.13 0.13 0.13], ...
            'XColor', 'w', ...
            'YColor', 'w', ...
            'FontSize', 11);
title(ax_bar, 'Drowsiness Score', 'Color', 'w', 'FontSize', 12);
ylabel(ax_bar, 'Probability', 'Color', 'w');
ylim(ax_bar, [0 1]);
hold(ax_bar, 'on');
yline(ax_bar, DROWSY_THRESHOLD, 'r--', 'LineWidth', 1.5);

% Bottom right — rolling probability history
ax_hist = axes('Parent', fig, ...
               'Position', [0.66 0.07 0.32 0.38]);
set(ax_hist, 'Color',  [0.13 0.13 0.13], ...
             'XColor', 'w', ...
             'YColor', 'w', ...
             'FontSize', 10);
title(ax_hist, 'Drowsy Probability History', 'Color', 'w', 'FontSize', 12);
ylabel(ax_hist, 'P(drowsy)', 'Color', 'w');
xlabel(ax_hist, 'Frames',    'Color', 'w');
ylim(ax_hist, [0 1]);
hold(ax_hist, 'on');
yline(ax_hist, DROWSY_THRESHOLD, 'r--', 'LineWidth', 1.5);

% ── STEP 4: Initialise state variables ───────────────────────────────
HIST_LEN   = 200;
drowsyHist = nan(1, HIST_LEN);
histPos    = 0;
frameCount = 0;

% Last known inference result (shown while waiting for next inference)
awake_p    = 0.5;
drowsy_p   = 0.5;
is_drowsy  = false;
confidence = 0.5;

% Q key sets stop flag
set(fig, 'KeyPressFcn', @(~,e) setappdata(fig,'stop',strcmp(e.Key,'q')));
setappdata(fig, 'stop', false);

fprintf('Running — press Q in the figure window to quit.\n');
fprintf('%-6s  %-24s  %-10s  %-10s\n', 'Frame', 'Status', 'Awake', 'Drowsy');
fprintf('%s\n', repmat('-', 1, 56));

% ── STEP 5: Main loop ─────────────────────────────────────────────────
while ishandle(fig) && ~getappdata(fig, 'stop')

    % 5a. Capture frame from webcam
    frame_rgb  = snapshot(cam);        % H×W×3  uint8  RGB
    frameCount = frameCount + 1;

    % 5b. Run inference every RUN_EVERY frames
    if mod(frameCount, RUN_EVERY) == 0
        try
            % Preprocess — resize, normalise, reformat for ViT
            img  = imresize(frame_rgb, [224 224]);          % 224×224×3 uint8
            img  = single(img) / 255.0;                     % → [0,1] single
            img  = (img - IMG_MEAN) ./ IMG_STD;             % ImageNet normalise
            img  = permute(img, [3 1 2]);                   % HWC → CHW [3,224,224]
            img  = reshape(img, [1 3 224 224]);             % add batch dim
            dlImg = dlarray(img, 'BCSS');                   % label as BCSS

            % Forward pass
            dlOut  = predict(net, dlImg);                   % [1,2] dlarray
            logits = double(extractdata(dlOut));            % [1,2] double

            % Softmax
            e      = exp(logits - max(logits));
            probs  = e / sum(e);

            awake_p    = probs(1);
            drowsy_p   = probs(2);
            is_drowsy  = drowsy_p > DROWSY_THRESHOLD;
            confidence = max(probs);

        catch ME
            fprintf('  Inference error (frame %d): %s\n', frameCount, ME.message);
        end
    end

    % 5c. Update rolling history
    histPos             = mod(frameCount-1, HIST_LEN) + 1;
    drowsyHist(histPos) = drowsy_p;

    % 5d. Pick status colour and label
    if is_drowsy
        sColor = [0.90 0.10 0.10];   % red
        sLabel = sprintf('DROWSY   %.0f%%', confidence * 100);
    else
        sColor = [0.10 0.80 0.20];   % green
        sLabel = sprintf('AWAKE    %.0f%%', confidence * 100);
    end

    % 5e. Draw coloured status bar onto camera frame
    dispFrame             = frame_rgb;
    px                    = uint8(sColor * 255);
    dispFrame(1:70, :, 1) = px(1);
    dispFrame(1:70, :, 2) = px(2);
    dispFrame(1:70, :, 3) = px(3);

    % Display annotated frame
    imshow(dispFrame, 'Parent', ax_cam);
    hold(ax_cam, 'on');
    text(ax_cam, 16, 42, sLabel, ...
         'Color', 'white', 'FontSize', 20, 'FontWeight', 'bold');
    text(ax_cam, 16, 78, ...
         sprintf('Awake: %.3f     Drowsy: %.3f', awake_p, drowsy_p), ...
         'Color', 'white', 'FontSize', 13);
    text(ax_cam, 16, size(dispFrame,1) - 12, ...
         sprintf('Frame: %d   Press Q to quit', frameCount), ...
         'Color', [0.7 0.7 0.7], 'FontSize', 10);
    hold(ax_cam, 'off');

    % 5f. Update bar chart
    cla(ax_bar);
    b = bar(ax_bar, [awake_p, drowsy_p]);
    b.FaceColor = 'flat';
    b.CData     = [0.10 0.80 0.20;   % green for Awake
                   0.90 0.10 0.10];  % red for Drowsy
    set(ax_bar, 'XTickLabel', {'Awake', 'Drowsy'}, ...
                'XColor', 'w', 'YColor', 'w', ...
                'Color',  [0.13 0.13 0.13], ...
                'FontSize', 11);
    ylim(ax_bar, [0 1]);
    yline(ax_bar, DROWSY_THRESHOLD, 'r--', 'LineWidth', 1.5);
    title(ax_bar, 'Drowsiness Score', 'Color', 'w', 'FontSize', 12);
    ylabel(ax_bar, 'Probability', 'Color', 'w');

    % 5g. Update history plot
    ordered = circshift(drowsyHist, -(histPos));
    cla(ax_hist);
    hold(ax_hist, 'on');
    area(ax_hist, 1:HIST_LEN, ordered, ...
         'FaceColor', [0.9 0.5 0.1], ...
         'FaceAlpha', 0.35, ...
         'EdgeColor', 'none');
    plot(ax_hist, 1:HIST_LEN, ordered, ...
         'Color', [1.0 0.75 0.2], 'LineWidth', 1.8);
    yline(ax_hist, DROWSY_THRESHOLD, 'r--', 'LineWidth', 1.5);
    ylim(ax_hist, [0 1]);
    set(ax_hist, 'Color',  [0.13 0.13 0.13], ...
                 'XColor', 'w', 'YColor', 'w', 'FontSize', 10);
    title(ax_hist, 'Drowsy Probability History', 'Color', 'w', 'FontSize', 12);
    ylabel(ax_hist, 'P(drowsy)', 'Color', 'w');
    xlabel(ax_hist, 'Frames',    'Color', 'w');

    % 5h. Console output every 30 frames
    if mod(frameCount, 10) == 0
        fprintf('%-6d  %-24s  %-10.3f  %-10.3f\n', ...
                frameCount, sLabel, awake_p, drowsy_p);
    end

    drawnow limitrate;
end

% ── STEP 6: Clean up ──────────────────────────────────────────────────
fprintf('\n%s\n', repmat('-', 1, 56));
fprintf('Stopped at frame %d.\n', frameCount);
clear cam;
fprintf('Camera released. Done.\n');
