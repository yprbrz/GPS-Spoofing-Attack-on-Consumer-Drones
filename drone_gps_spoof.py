"""
GPS Spoofing Attack on Consumer Drones
=======================================
CPS & IoT Security Course Project
Reference paper: "Tractor Beam: Safe-hijacking of Consumer Drones
                  with Adaptive GPS Spoofing"

Modules:
  1. Drone RTH simulation (PID-controlled 2D position)
  2. Greedy GPS spoofing attack (redirect drone to attacker destination)

Author: [Your Name]
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os

# ── Output directory ──
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(out, exist_ok=True)

# ──────────────────────────────────────────────
# SIMULATION PARAMETERS
# ──────────────────────────────────────────────
dt           = 0.1          # time step (s)
T_MAX        = 200.0        # total simulation time (s)
time         = np.arange(0, T_MAX, dt)

# Drone physical parameters
DRONE_MASS   = 1.5          # kg
DRAG         = 0.6          # drag coefficient (simplified)
MAX_SPEED    = 6.0          # m/s (consumer drone typical cruise)
MAX_FORCE    = 20.0         # N  (max thrust in horizontal plane)

# PID gains for position control (x and y independently)
Kp = 4.0
Ki = 0.05
Kd = 1.2

# GPS spoofing parameters
Vs           = 6.0          # max spoofed-location change speed (m/s)
LEASH_LENGTH = 13.0         # ArduCopter leash length (m) — from paper
SPOOF_START  = 10.0         # seconds: when attacker begins spoofing
GPS_PERIOD   = 0.2          # GPS update period delta (s)

# ──────────────────────────────────────────────
# KEY LOCATIONS  (x, y in metres)
# ──────────────────────────────────────────────
H    = np.array([300.0,   0.0])     # Home location
S    = np.array([100.0, 300.0])     # Drone starting position (where it was when signal jammed)
D    = np.array([0.0,     0.0])     # Attacker's desired destination

# ──────────────────────────────────────────────
# MODULE 1 — PID CONTROLLER FOR 2-D POSITION
# ──────────────────────────────────────────────

class PIDController:
    """Independent PID for one spatial axis."""
    def __init__(self, kp, ki, kd, dt):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.dt = dt
        self.integral = 0.0
        self.prev_error = 0.0

    def step(self, error):
        self.integral    += error * self.dt
        derivative        = (error - self.prev_error) / self.dt
        output            = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error   = error
        return output


class DroneRTH:
    """
    2-D drone in Return-To-Home mode.
    Uses two independent PID controllers (x and y axis) to fly from
    current position toward the target (home or spoofed location).
    Models simplified Newton dynamics: F = ma with drag.
    """
    def __init__(self, start_pos, home_pos, dt):
        self.pos      = start_pos.copy().astype(float)
        self.vel      = np.zeros(2)
        self.home     = home_pos.copy().astype(float)
        self.dt       = dt
        self.pid_x    = PIDController(Kp, Ki, Kd, dt)
        self.pid_y    = PIDController(Kp, Ki, Kd, dt)

        # Logging
        self.pos_log  = [self.pos.copy()]
        self.vel_log  = [self.vel.copy()]

    def step(self, perceived_pos):
        """
        perceived_pos: what the drone *believes* its position is
                       (either true GPS or the spoofed value).
        The drone tries to reach self.home from perceived_pos.
        """
        error = self.home - perceived_pos          # vector toward home

        fx = self.pid_x.step(error[0])
        fy = self.pid_y.step(error[1])
        force = np.array([fx, fy])

        # Clamp to max thrust
        f_mag = np.linalg.norm(force)
        if f_mag > MAX_FORCE:
            force = force / f_mag * MAX_FORCE

        # Newton: a = (F - drag*v) / m
        acc      = (force - DRAG * self.vel) / DRONE_MASS
        self.vel = self.vel + acc * self.dt

        # Clamp speed
        speed = np.linalg.norm(self.vel)
        if speed > MAX_SPEED:
            self.vel = self.vel / speed * MAX_SPEED

        self.pos = self.pos + self.vel * self.dt

        self.pos_log.append(self.pos.copy())
        self.vel_log.append(self.vel.copy())

        return self.pos.copy()


# ──────────────────────────────────────────────
# MODULE 2 — GREEDY GPS SPOOFING ALGORITHM
# ──────────────────────────────────────────────

class GPSSpoofingAttacker:
    """
    Implements the greedy spoofing algorithm from the paper.

    The attacker cannot jump the spoofed GPS position arbitrarily fast
    (constraint Vs). Instead it incrementally steers the spoofed location
    so that the drone's perceived error vector points toward D.

    The drone is in RTH mode: it flies from perceived_pos toward H.
    The attacker exploits this by making perceived_pos drift so that
    the direction (perceived_pos → H) aligns with (actual_pos → D).
    """
    def __init__(self, home, destination, vs, leash, gps_period):
        self.H          = home.copy().astype(float)
        self.D          = destination.copy().astype(float)
        self.Vs         = vs
        self.leash      = leash
        self.gps_period = gps_period

        # Spoofed location starts at the drone's true position
        self.spoofed    = None
        self.spoof_log  = []
        self._time_since_update = 0.0

    def init_spoofed_pos(self, true_pos):
        """
        Compute ainit: the initial fake location.
        ainit = true_pos + k*(D - true_pos)  where k < 0
        so the direction ainit→H equals true_pos→D.
        """
        direction   = self.D - true_pos
        dist        = np.linalg.norm(direction)
        if dist < 1e-6:
            self.spoofed = true_pos.copy()
            return

        unit        = direction / dist
        # k < 0 so ainit is on the opposite side of true_pos from D
        # distance must exceed leash length
        k           = -(self.leash + 5.0) / dist   # slightly beyond leash
        self.spoofed = true_pos + k * (self.D - true_pos)
        self.spoof_log.append(self.spoofed.copy())

    def step(self, true_pos, dt):
        """
        Return the spoofed GPS position the drone will perceive.
        Each GPS update period, advance the spoofed location adaptively:
          a(t) = a(t-Δ) + (p(t) - p(t-Δ))
        This keeps GPS velocity consistent with IMU to avoid EKF failsafe.
        Between updates the spoofed location tracks the drone's true motion.
        """
        if self.spoofed is None:
            self.init_spoofed_pos(true_pos)

        self._time_since_update += dt

        if self._time_since_update >= self.gps_period:
            self._time_since_update = 0.0
            self._update_spoofed(true_pos)

        self.spoof_log.append(self.spoofed.copy())
        return self.spoofed.copy()

    def _update_spoofed(self, true_pos):
        """
        Greedy step: move the spoofed location so that the vector
        (spoofed → H) increasingly aligns with (true → D).
        Displacement is capped at Vs * gps_period.
        """
        # Target for spoofed location: a point such that (a→H) || (true→D)
        hijack_dir = self.D - true_pos
        hijack_dist = np.linalg.norm(hijack_dir)
        if hijack_dist < 0.5:
            # Close enough — done
            return

        unit_hijack = hijack_dir / hijack_dist

        # We want: H - spoofed  ||  D - true_pos
        # i.e. spoofed = H - k*(D - true_pos) for some k > 0
        # Greedy: pick the nearest such point to current spoofed
        # Parametric: spoofed_target(k) = H - k * unit_hijack
        # Find k so that ||spoofed_target - spoofed|| is minimised
        # → project current spoofed onto the line through H in dir -unit_hijack
        v           = self.spoofed - self.H
        k_proj      = -np.dot(v, unit_hijack)   # projection
        k_new       = max(self.leash + 3.0, k_proj + self.Vs * self.gps_period)
        target      = self.H - k_new * unit_hijack

        # Cap movement speed
        delta       = target - self.spoofed
        delta_dist  = np.linalg.norm(delta)
        max_step    = self.Vs * self.gps_period
        if delta_dist > max_step:
            delta = delta / delta_dist * max_step

        self.spoofed = self.spoofed + delta


# ──────────────────────────────────────────────
# RUN SIMULATION
# ──────────────────────────────────────────────

def run_simulation():
    drone_attack   = DroneRTH(S, H, dt)       # victim drone (under attack)
    drone_baseline = DroneRTH(S, H, dt)       # reference: no spoofing
    attacker       = GPSSpoofingAttacker(H, D, Vs, LEASH_LENGTH, GPS_PERIOD)

    spoofed_log    = []
    attack_active  = False
    prev_true_pos  = S.copy()

    for t in time:
        true_pos = drone_attack.pos.copy()

        # ── Baseline drone: always sees true GPS ──
        drone_baseline.step(drone_baseline.pos)

        # ── Attacked drone ──
        if t >= SPOOF_START:
            if not attack_active:
                attack_active = True
                attacker.init_spoofed_pos(true_pos)

            perceived = attacker.step(true_pos, dt)
            spoofed_log.append(perceived.copy())
        else:
            # Before attack: drone sees true position, flies RTH normally
            perceived = true_pos.copy()
            spoofed_log.append(perceived.copy())

        drone_attack.step(perceived)
        prev_true_pos = true_pos

    return drone_attack, drone_baseline, attacker, spoofed_log


# ──────────────────────────────────────────────
# PLOTTING
# ──────────────────────────────────────────────

def plot_results(drone_attack, drone_baseline, attacker, spoofed_log):
    actual_traj   = np.array(drone_attack.pos_log)
    baseline_traj = np.array(drone_baseline.pos_log)
    spoofed_traj  = np.array(spoofed_log)
    spoof_pts     = np.array(attacker.spoof_log) if attacker.spoof_log else None

    # ── Compute distance to D over time ──
    dist_to_D     = np.linalg.norm(actual_traj - D, axis=1)
    dist_baseline = np.linalg.norm(baseline_traj - H, axis=1)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("GPS Spoofing Attack on Consumer Drone (RTH Mode)",
                 fontsize=15, fontweight='bold', y=0.98)
    gs  = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    # ── Panel 1: 2-D Trajectory ──
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.set_title("2-D Trajectory", fontsize=12, fontweight='bold')

    ax1.plot(baseline_traj[:, 0], baseline_traj[:, 1],
             color='steelblue', linewidth=2, label='Baseline RTH (no attack)')
    ax1.plot(actual_traj[:, 0],   actual_traj[:, 1],
             color='crimson',    linewidth=2, linestyle='--', label='Actual trajectory (under attack)')

    if spoof_pts is not None and len(spoof_pts) > 0:
        ax1.plot(spoof_pts[:, 0], spoof_pts[:, 1],
                 color='darkorange', linewidth=1.2, linestyle=':', alpha=0.7,
                 label='Spoofed GPS positions')

    # Key locations
    for loc, name, color, marker in [
        (S, 'S\n(start)', 'black',    'o'),
        (H, 'H\n(home)',  'steelblue','s'),
        (D, 'D\n(dest.)', 'crimson',  '^'),
    ]:
        ax1.scatter(*loc, color=color, s=120, zorder=6, marker=marker)
        ax1.annotate(name, loc, textcoords='offset points',
                     xytext=(8, 6), fontsize=9, color=color, fontweight='bold')

    # Mark attack start
    spoof_start_idx = int(SPOOF_START / dt)
    if spoof_start_idx < len(actual_traj):
        ax1.scatter(*actual_traj[spoof_start_idx],
                    color='gold', s=140, zorder=7, marker='*',
                    label=f'Attack starts (t={SPOOF_START}s)')

    ax1.set_xlabel("X position (m)")
    ax1.set_ylabel("Y position (m)")
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='datalim')

    # ── Panel 2: Distance to destination D ──
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("Distance to Attacker Destination D", fontsize=11, fontweight='bold')
    n_steps = min(len(time), len(dist_to_D))
    t_axis = time[:n_steps]
    dist_to_D = dist_to_D[:n_steps]
    ax2.plot(t_axis, dist_to_D, color='crimson', linewidth=1.8, label='Drone under attack')
    ax2.axvline(SPOOF_START, color='gold', linestyle='--', linewidth=1.5,
                label=f'Attack start (t={SPOOF_START}s)')
    ax2.axhline(LEASH_LENGTH, color='gray', linestyle=':', linewidth=1.2,
                label=f'Leash length ({LEASH_LENGTH}m)')
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Distance to D (m)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: GPS innovation (anomaly signal for detection) ──
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_title("GPS Innovation Signal\n(GPS pos − IMU-estimated pos)", fontsize=11, fontweight='bold')

    actual_arr  = np.array(drone_attack.pos_log[:-1])
    spoof_arr   = np.array(spoofed_log)
    min_len     = min(len(actual_arr), len(spoof_arr))
    innovation  = np.linalg.norm(spoof_arr[:min_len] - actual_arr[:min_len], axis=1)
    t_innov     = time[:min_len]

    ax3.plot(t_innov, innovation, color='darkorange', linewidth=1.5,
             label='||spoofed − true|| (innovation magnitude)')
    ax3.axvline(SPOOF_START, color='gold', linestyle='--', linewidth=1.5,
                label=f'Attack start (t={SPOOF_START}s)')
    ax3.axhline(LEASH_LENGTH, color='gray', linestyle=':', linewidth=1.2,
                label=f'Leash length ({LEASH_LENGTH}m)')
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Innovation magnitude (m)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.savefig(os.path.join(out, "gps_spoofing_attack.png"), dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved → gps_spoofing_attack.png")

    return innovation, t_innov


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("Running GPS spoofing simulation...")
    drone_attack, drone_baseline, attacker, spoofed_log = run_simulation()

    final_pos  = drone_attack.pos_log[-1]
    dist_final = np.linalg.norm(np.array(final_pos) - D)
    print(f"\nDrone final position : ({final_pos[0]:.1f}, {final_pos[1]:.1f}) m")
    print(f"Attacker destination : ({D[0]:.1f}, {D[1]:.1f}) m")
    print(f"Distance to D at end : {dist_final:.1f} m")
    print(f"Home location        : ({H[0]:.1f}, {H[1]:.1f}) m")
    dist_H = np.linalg.norm(np.array(final_pos) - H)
    print(f"Distance to H at end : {dist_H:.1f} m")
    if dist_final < dist_H:
        print("\n✓ Attack SUCCESSFUL — drone redirected toward attacker destination")
    else:
        print("\n✗ Attack did not fully redirect the drone in the given time window")

    innovation, t_innov = plot_results(drone_attack, drone_baseline, attacker, spoofed_log)

    # Export innovation signal for Module 3 (anomaly detection)
    np.save(os.path.join(out, "innovation_signal.npy"), innovation)
    np.save(os.path.join(out, "innovation_time.npy"),   t_innov)
    print("\nInnovation signal saved → innovation_signal.npy (use in Module 3)")
