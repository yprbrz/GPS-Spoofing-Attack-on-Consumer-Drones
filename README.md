# GPS Spoofing Attack on Consumer Drones
### Simulation and Detection

> **Course project for:** SCQ3102439 – Cyberphysical and IoT Security 2025–2026  
> **University:** Università degli Studi di Padova  
> **Author:** Yllke Prebreza  
> **Reference paper:** Noh et al. (2019) — *Tractor Beam: Safe-hijacking of Consumer Drones with Adaptive GPS Spoofing*, ACM TOPS. https://doi.org/10.1145/3309735

---

## Project Description

This project implements and evaluates the adaptive GPS spoofing attack described in the Tractor Beam paper. A consumer drone operating in Return-to-Home (RTH) mode is redirected to an attacker-controlled destination by feeding it incrementally spoofed GPS coordinates. The adaptive spoofing strategy keeps GPS velocity consistent with IMU readings to avoid triggering the drone's onboard EKF failsafe.

The project extends the original paper with a detection layer: two statistical anomaly detectors (CUSUM and SPRT) are applied to the GPS innovation signal — the discrepancy between the spoofed GPS position and the drone's true position — to identify the attack in real time.

The entire project is a software-only simulation written in Python. No hardware is required.

**Module 1 & 2 — `drone_gps_spoof.py`**  
Simulates the victim drone under PID-based RTH control and applies the greedy GPS spoofing algorithm to redirect it from home H to attacker destination D.

**Module 3 — `module3_detection.py`**  
Loads the GPS innovation signal produced by Module 1 & 2 and runs CUSUM and SPRT detectors, comparing their detection delay and false alarm behaviour.

---

## Installation

Requires Python 3.8+ and the following packages:

```bash
pip install numpy matplotlib
```

No other dependencies are needed.

---

## How to Run

> **Important:** always run `drone_gps_spoof.py` first. Module 3 reads the `.npy` files it produces.

### Step 1 — Attack simulation

```bash
python drone_gps_spoof.py
```

Saves the trajectory plot and GPS innovation signal to an `outputs/` folder created automatically next to the script. Expected output:

```
✓ Attack SUCCESSFUL — drone redirected toward attacker destination
Plot saved → outputs/gps_spoofing_attack.png
Innovation signal saved → innovation_signal.npy (use in Module 3)
```

### Step 2 — Detection

```bash
python module3_detection.py
```

Reads the innovation signal from Step 1 and saves the detection results plot. Expected output:

```
CUSUM first alarm : t = 10.10s  (+0.10s delay)
SPRT  first alarm : t = 6.70s   (false alarm before attack start)
Plot saved → outputs/detection_results.png
```
