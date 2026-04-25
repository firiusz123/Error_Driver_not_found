# Error_Driver_not_found

## Academic Overview
This project presents a MATLAB-based prototype for **driver incapacity detection** and **minimum-risk maneuver (MRM) execution** in a simulated highway context. A multimodal Driver Monitoring System (webcam-based ViT drowsiness inference fused with face-landmark EAR analysis) triggers a finite-state safety controller, which then coordinates trajectory planning and controlled pull-over behavior.

The solution was developed for the **RCDC Hackathon** and achieved **2nd place** in the **VASC (Vehicle Assist System Challenge)** category:  
https://best.krakow.pl/rcdc/

## System Highlights
- Real-time drowsiness estimation from webcam input (`dms_step`, `face_detect_step`)
- Safety-oriented finite-state logic for takeover and emergency states (`mrm_step` / Stateflow)
- Perception-informed planning with lane safety and shoulder detection (`get_sensor_data`, `planner_logic`)
- Closed-loop simulation orchestration in MATLAB (`main/main.m`)

## Challenge Context
**Category:** VASC Vehicle Assist System Challenge  
**Official brief:**  
*"Design a vehicle support system that improves safety and performance using MATLAB and MathWorks simulation and analytical tools. Emphasis is placed on creativity, real impact on the future of automotive technology, and the ability to apply modeling, data analysis, AI, and control systems."*

**Final topic:** *"Error, driver not found"*

Our response was to engineer a driver-state detection module that addresses not only classical drowsiness patterns, but also broader signs of sudden medical incapacity (e.g., collapse or loss of control associated with acute illness), by combining ocular and **postural** indicators.

## Quick Start
1. Open MATLAB R2025b with required toolboxes (Deep Learning, ONNX conversion, USB Webcam support).
2. Prepare the merged ONNX model via `python_merge.py`.
3. Convert the model to MATLAB format using `convert_model.m`.
4. Run the integrated simulation pipeline from `main/main.m`.

## Repository Focus
The codebase is organized around four research modules: **driver monitoring**, **state-based safety control**, **trajectory planning**, and **scenario-level simulation**.

---
README was created by AI.
