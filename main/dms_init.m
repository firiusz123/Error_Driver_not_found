function dms = dms_init(net_path)
    %DMS_INIT Initialize the driver-monitoring system (webcam + ViT inference).
    %
    %   dms = dms_init(net_path)
    %
    % Returns a struct with fields used by dms_step():
    %   .enabled            true if webcam + network are both available
    %   .net                loaded dlnetwork (empty if disabled)
    %   .cam                webcam handle (empty if disabled)
    %   .drowsy_p           latest P(drowsy) — initialized to 0.5
    %   .last_inference_t   sim_time of last inference call
    %   .img_mean, img_std  ImageNet normalization tensors (cached)
    %
    % Falls back to .enabled = false on any failure (missing .mat, no
    % webcam, busy webcam) so main.m can drop to scripted incapacitation.

    dms = struct( ...
        'enabled',          false, ...
        'net',              [], ...
        'cam',              [], ...
        'drowsy_p',         0.5, ...
        'last_inference_t', -inf, ...
        'frame_count',      0, ...
        'last_frame',       [], ...
        'img_mean',         reshape([0.485 0.456 0.406], 1, 1, 3), ...
        'img_std',          reshape([0.229 0.224 0.225], 1, 1, 3));

    if nargin < 1 || isempty(net_path)
        net_path = fullfile( ...
            '/Users/hubertm/Documents/College/Events/VASC/Error_Driver_not_found', ...
            'drowsiness_net.mat');
    end

    if ~isfile(net_path)
        warning('DMS:NoNet', ...
            'drowsiness_net.mat not found at %s — DMS disabled.', net_path);
        return;
    end

    try
        data = load(net_path, 'net');
        dms.net = data.net;
    catch ME
        warning('DMS:LoadFail', ...
            'Failed to load DMS network: %s — DMS disabled.', ME.message);
        return;
    end

    if isempty(webcamlist())
        warning('DMS:NoCam', 'No webcam detected — DMS disabled.');
        return;
    end

    try
        dms.cam = webcam(1);
        try, dms.cam.Brightness = 0;   catch, end
        try, dms.cam.Contrast   = 128; catch, end
        try, dms.cam.Saturation = 128; catch, end
    catch ME
        warning('DMS:CamFail', ...
            'Webcam open failed: %s — DMS disabled.', ME.message);
        return;
    end

    dms.enabled = true;
    fprintf('DMS enabled: %s @ %s\n', dms.cam.Name, dms.cam.Resolution);
end
