"""
GPS Spoofing Detection via Statistical Anomaly Detection
=========================================================
CPS & IoT Security Course Project — Module 3

Applies CUSUM and SPRT to the GPS innovation signal produced by Module 2.
The innovation ||spoofed_pos - true_pos|| spikes when GPS spoofing begins,
because the GPS velocity becomes inconsistent with IMU measurements.
Both detectors flag this anomaly, acting as a countermeasure.

Reference: "Tractor Beam: Safe-hijacking of Consumer Drones
            with Adaptive GPS Spoofing"
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# ──────────────────────────────────────────────
# LOAD INNOVATION SIGNAL FROM MODULE 2
# ──────────────────────────────────────────────

signal    = np.load(os.path.join(out, "innovation_signal.npy"))
time_axis = np.load(os.path.join(out, "innovation_time.npy"))
ATTACK_T   = 10.0   # ground-truth attack start time (seconds)
dt         = time_axis[1] - time_axis[0]

# ──────────────────────────────────────────────
# SIGNAL CALIBRATION
# Pre-attack window is perfectly clean (mean=0, std≈0).
# We add a small amount of realistic sensor noise to make the
# problem well-posed (CUSUM/SPRT need a non-zero baseline variance).
# ──────────────────────────────────────────────
np.random.seed(42)
NOISE_STD = 4.5          # metres — realistic consumer-grade GPS noise (±5m)
noisy_signal = signal + np.random.normal(0, NOISE_STD, len(signal))
# Add occasional GPS multipath spikes in normal phase (realistic environment)
spike_idx = np.random.choice(np.where(time_axis < ATTACK_T)[0], size=3, replace=False)
noisy_signal[spike_idx] += np.random.uniform(6, 10, 3)

# Statistical parameters estimated from pre-attack window
pre_mask   = time_axis < ATTACK_T
mu0        = float(noisy_signal[pre_mask].mean())    # normal mean  ≈ 0
sigma      = float(noisy_signal[pre_mask].std())     # normal std
# We don't know the exact attack magnitude in advance; use a conservative
# estimate: mu1 = mu0 + 2*sigma  (detectable shift = 2 standard deviations)
SHIFT      = 3.0 * max(sigma, NOISE_STD)             # expected shift size
mu1        = mu0 + SHIFT

print(f"Signal calibration:")
print(f"  Pre-attack mean  (mu0) : {mu0:.3f} m")
print(f"  Pre-attack std (sigma) : {sigma:.3f} m")
print(f"  Hypothesised shift     : {SHIFT:.3f} m  →  mu1 = {mu1:.3f} m")


# ──────────────────────────────────────────────
# DETECTOR 1 — CUSUM (Cumulative Sum)
# ──────────────────────────────────────────────
# Detects a persistent mean shift.
# Each sample contributes a log-likelihood ratio (LLR) to a running sum g.
# When g exceeds the threshold h, a change is declared and g resets.
# ──────────────────────────────────────────────

def run_cusum(data, mu0, mu1, sigma, threshold):
    """
    One-sided upper CUSUM for detecting a rise from mu0 to mu1.

    Returns
    -------
    g_values   : CUSUM statistic at every sample
    detections : list of (sample_index, time) where alarm fires
    """
    g          = 0.0
    g_values   = []
    detections = []

    for n, x in enumerate(data):
        llr = (mu1 - mu0) / sigma**2 * (x - (mu0 + mu1) / 2.0)
        g   = max(0.0, g + llr)
        g_values.append(g)

        if g > threshold:
            detections.append(n)
            g = 0.0     # reset after alarm

    return np.array(g_values), detections


# Threshold h: chosen to give roughly one false alarm per 200 samples
# under H0.  h = 4 is a common default; we scale with sigma.
CUSUM_THRESHOLD = 8.0

cusum_stat, cusum_detections = run_cusum(
    noisy_signal, mu0, mu1, sigma, CUSUM_THRESHOLD
)

cusum_alarm_times = [time_axis[i] for i in cusum_detections if i < len(time_axis)]
first_cusum = cusum_alarm_times[0] if cusum_alarm_times else None
delay_cusum = (first_cusum - ATTACK_T) if first_cusum else None

print(f"\nCUSUM results (threshold = {CUSUM_THRESHOLD}):")
print(f"  First alarm at t = {first_cusum:.2f}s" if first_cusum else "  No alarm fired")
if delay_cusum is not None:
    print(f"  Detection delay  = {delay_cusum:.2f}s after attack start")
print(f"  Total alarms     = {len(cusum_detections)}")


# ──────────────────────────────────────────────
# DETECTOR 2 — SPRT (Sequential Probability Ratio Test)
# ──────────────────────────────────────────────
# Decides between H0 (normal, mu0) and H1 (attack, mu1) sequentially.
# Accumulates a log-likelihood ratio S; declares H1 when S ≥ A,
# declares H0 when S ≤ B, otherwise continues.
# ──────────────────────────────────────────────

def run_sprt(data, mu0, mu1, sigma, alpha, beta):
    """
    Continuous SPRT with reset after each decision.

    Parameters
    ----------
    alpha : desired false alarm probability (Type I error)
    beta  : desired miss probability (Type II error)

    Returns
    -------
    S_trace        : cumulative LLR at every sample
    decisions      : list of (sample_index, label) at each decision
    alarm_indices  : indices where 'Anomaly' was declared
    """
    A = np.log((1 - beta) / alpha)    # upper boundary → H1 (anomaly)
    B = np.log(beta / (1 - alpha))    # lower boundary → H0 (normal)

    S             = 0.0
    S_trace       = []
    decisions     = []
    alarm_indices = []

    for n, x in enumerate(data):
        S += (mu1 - mu0) / sigma**2 * (x - (mu0 + mu1) / 2.0)
        S_trace.append(S)

        if S >= A:
            decisions.append((n, "Anomaly"))
            alarm_indices.append(n)
            S = 0.0
        elif S <= B:
            decisions.append((n, "Normal"))
            S = 0.0

    return np.array(S_trace), decisions, alarm_indices, A, B


ALPHA = 0.02   # 2% false alarm rate
BETA  = 0.05   # 5% miss rate

sprt_trace, sprt_decisions, sprt_alarms, A_bound, B_bound = run_sprt(
    noisy_signal, mu0, mu1, sigma, ALPHA, BETA
)

sprt_alarm_times = [time_axis[i] for i in sprt_alarms if i < len(time_axis)]
first_sprt       = sprt_alarm_times[0] if sprt_alarm_times else None
delay_sprt       = (first_sprt - ATTACK_T) if first_sprt else None
sprt_normal      = [i for i, (_, lbl) in enumerate(sprt_decisions) if lbl == "Normal"]

print(f"\nSPRT results (α={ALPHA}, β={BETA}):")
print(f"  Upper threshold A = {A_bound:.3f}")
print(f"  Lower threshold B = {B_bound:.3f}")
print(f"  First alarm at t = {first_sprt:.2f}s" if first_sprt else "  No alarm fired")
if delay_sprt is not None:
    print(f"  Detection delay  = {delay_sprt:.2f}s after attack start")
anomaly_count = sum(1 for _, lbl in sprt_decisions if lbl == "Anomaly")
normal_count  = sum(1 for _, lbl in sprt_decisions if lbl == "Normal")
print(f"  'Anomaly' decisions: {anomaly_count}")
print(f"  'Normal'  decisions: {normal_count}")


# ──────────────────────────────────────────────
# EVALUATION SUMMARY
# ──────────────────────────────────────────────

print("\n" + "="*55)
print("DETECTION SUMMARY")
print("="*55)
print(f"  Ground-truth attack start  : t = {ATTACK_T:.1f}s")
print(f"  CUSUM first alarm          : t = {first_cusum:.2f}s  (+{delay_cusum:.2f}s delay)" if first_cusum else "  CUSUM: no alarm")
print(f"  SPRT  first alarm          : t = {first_sprt:.2f}s  (+{delay_sprt:.2f}s delay)"  if first_sprt  else "  SPRT:  no alarm")
print("="*55)


# ──────────────────────────────────────────────
# PLOTTING
# ──────────────────────────────────────────────

fig = plt.figure(figsize=(16, 12))
fig.suptitle("GPS Spoofing Detection — Module 3\n"
             "CUSUM & SPRT on GPS Innovation Signal",
             fontsize=14, fontweight='bold', y=0.98)
gs = GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.32)


# ── Panel 1: Raw innovation signal (full width) ──
ax0 = fig.add_subplot(gs[0, :])
ax0.set_title("Innovation Signal  ||spoofed_pos − true_pos||", fontsize=11, fontweight='bold')
ax0.fill_between(time_axis, noisy_signal,
                 where=(time_axis < ATTACK_T),
                 color='steelblue', alpha=0.3, label='Normal phase')
ax0.fill_between(time_axis, noisy_signal,
                 where=(time_axis >= ATTACK_T),
                 color='crimson', alpha=0.25, label='Attack phase')
ax0.plot(time_axis, noisy_signal, color='dimgray', linewidth=0.9, alpha=0.8)
ax0.axvline(ATTACK_T, color='red', linestyle='--', linewidth=1.8,
            label=f'Attack start (t={ATTACK_T}s)')
ax0.set_xlabel("Time (s)")
ax0.set_ylabel("Innovation (m)")
ax0.legend(fontsize=9, loc='upper left')
ax0.grid(True, alpha=0.3)


# ── Panel 2: CUSUM statistic ──
ax1 = fig.add_subplot(gs[1, 0])
ax1.set_title("CUSUM Detector", fontsize=11, fontweight='bold')
ax1.plot(time_axis, cusum_stat, color='steelblue', linewidth=1.5,
         label='CUSUM statistic g(n)')
ax1.axhline(CUSUM_THRESHOLD, color='crimson', linestyle='--', linewidth=1.8,
            label=f'Threshold h = {CUSUM_THRESHOLD}')
ax1.axvline(ATTACK_T, color='red', linestyle=':', linewidth=1.4,
            label=f'Attack start')
if first_cusum:
    ax1.axvline(first_cusum, color='darkorange', linestyle='-', linewidth=1.8,
                label=f'1st alarm t={first_cusum:.1f}s (Δ={delay_cusum:.1f}s)')
# Mark all alarm points
for i in cusum_detections:
    if i < len(time_axis):
        ax1.scatter(time_axis[i], CUSUM_THRESHOLD, color='darkorange',
                    s=60, zorder=5, marker='v')
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("CUSUM statistic")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)


# ── Panel 3: SPRT cumulative LLR ──
ax2 = fig.add_subplot(gs[1, 1])
ax2.set_title("SPRT Detector", fontsize=11, fontweight='bold')
ax2.plot(time_axis, sprt_trace, color='mediumpurple', linewidth=1.5,
         label='Cumulative LLR  S(n)', alpha=0.9)
ax2.axhline(A_bound, color='crimson', linestyle='--', linewidth=1.8,
            label=f'Upper bound A = {A_bound:.2f}  (→ Anomaly)')
ax2.axhline(B_bound, color='steelblue', linestyle='--', linewidth=1.8,
            label=f'Lower bound B = {B_bound:.2f}  (→ Normal)')
ax2.axvline(ATTACK_T, color='red', linestyle=':', linewidth=1.4,
            label='Attack start')
if first_sprt:
    ax2.axvline(first_sprt, color='darkorange', linestyle='-', linewidth=1.8,
                label=f'1st alarm t={first_sprt:.1f}s (Δ={delay_sprt:.1f}s)')
# Mark anomaly decisions
for i in sprt_alarms:
    if i < len(time_axis):
        ax2.scatter(time_axis[i], A_bound, color='crimson',
                    s=55, zorder=5, marker='v')
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Cumulative log-likelihood ratio")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)


# ── Panel 4: Decision timeline (CUSUM) ──
ax3 = fig.add_subplot(gs[2, 0])
ax3.set_title("CUSUM — Alarm Timeline", fontsize=11, fontweight='bold')
alarm_arr   = np.array([time_axis[i] for i in cusum_detections if i < len(time_axis)])
normal_mask = time_axis < ATTACK_T
# Background shading
ax3.axvspan(0,          ATTACK_T,  alpha=0.07, color='steelblue', label='Normal phase')
ax3.axvspan(ATTACK_T,   time_axis[-1], alpha=0.07, color='crimson', label='Attack phase')
if len(alarm_arr):
    # Separate false alarms (before attack) from true detections
    fa   = alarm_arr[alarm_arr < ATTACK_T]
    hits = alarm_arr[alarm_arr >= ATTACK_T]
    if len(fa):
        ax3.scatter(fa,   np.ones(len(fa)),   color='steelblue', s=80,
                    marker='x', linewidths=2, label=f'False alarm ({len(fa)})', zorder=5)
    if len(hits):
        ax3.scatter(hits, np.ones(len(hits)), color='crimson', s=80,
                    marker='v', label=f'True detection ({len(hits)})', zorder=5)
ax3.axvline(ATTACK_T, color='red', linestyle='--', linewidth=1.5)
ax3.set_xlabel("Time (s)")
ax3.set_yticks([])
ax3.legend(fontsize=9, loc='upper left')
ax3.grid(True, alpha=0.2, axis='x')
ax3.set_xlim(time_axis[0], time_axis[-1])


# ── Panel 5: Decision timeline (SPRT) ──
ax4 = fig.add_subplot(gs[2, 1])
ax4.set_title("SPRT — Decision Timeline", fontsize=11, fontweight='bold')
ax4.axvspan(0,          ATTACK_T,       alpha=0.07, color='steelblue')
ax4.axvspan(ATTACK_T,   time_axis[-1],  alpha=0.07, color='crimson')

for idx, (n, lbl) in enumerate(sprt_decisions):
    if n >= len(time_axis):
        continue
    t_dec = time_axis[n]
    color = 'crimson' if lbl == 'Anomaly' else 'steelblue'
    level = 1 if lbl == 'Anomaly' else 0
    ax4.scatter(t_dec, level, color=color, s=50, marker='v' if lbl=='Anomaly' else 'o',
                alpha=0.7, zorder=4)

ax4.axvline(ATTACK_T, color='red', linestyle='--', linewidth=1.5)
ax4.set_xlabel("Time (s)")
ax4.set_yticks([0, 1])
ax4.set_yticklabels(['Normal ✓', 'Anomaly ⚠'], fontsize=9)
anomaly_patch = mpatches.Patch(color='crimson',    label='Anomaly decision')
normal_patch  = mpatches.Patch(color='steelblue',  label='Normal decision')
ax4.legend(handles=[anomaly_patch, normal_patch], fontsize=9, loc='upper left')
ax4.grid(True, alpha=0.2, axis='x')
ax4.set_xlim(time_axis[0], time_axis[-1])


plt.savefig(os.path.join(out, "detection_results.png"), dpi=150, bbox_inches='tight')

plt.show()
print("\nPlot saved → detection_results.png")
