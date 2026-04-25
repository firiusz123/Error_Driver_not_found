function dms = dms_step(dms, sim_time)
    %DMS_STEP Capture one webcam frame and run drowsiness inference.
    %
    %   dms = dms_step(dms, sim_time)
    %
    % Rate-limits itself: only fires when sim_time has advanced past
    % last_inference_t by DMS_PERIOD seconds. Cheap to call every sim
    % step — most calls early-return and just hand back the cached
    % drowsy_p from the most recent inference.

    DMS_PERIOD = 0.25;   % 4 Hz max inference rate

    if ~dms.enabled
        return;
    end
    if (sim_time - dms.last_inference_t) < DMS_PERIOD
        return;
    end

    try
        frame = snapshot(dms.cam);
        dms.last_frame = frame;

        img = imresize(frame, [224 224]);
        img = single(img) / 255.0;
        img = (img - dms.img_mean) ./ dms.img_std;
        img = permute(img, [3 1 2]);          % HWC -> CHW
        img = reshape(img, [1 3 224 224]);    % add batch dim
        dlImg = dlarray(img, 'BCSS');

        dlOut  = predict(dms.net, dlImg);
        logits = double(extractdata(dlOut));
        e      = exp(logits - max(logits));
        probs  = e / sum(e);

        dms.drowsy_p         = probs(2);
        dms.last_inference_t = sim_time;
        dms.frame_count      = dms.frame_count + 1;
    catch ME
        % Soft-fail: log once per error, hold last drowsy_p value.
        warning('DMS:InferFail', 'Inference error at t=%.2fs: %s', ...
                sim_time, ME.message);
    end
end
