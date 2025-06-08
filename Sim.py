"""
Synthetic two-camera football-trajectory generator + tracker.
The script
  1. renders a white ball flying on a green background (side & front view);
  2. writes the PNG stacks to disk;
  3. re-detects the ball in every frame and reconstructs (t, x, y, z).

2025-06-08 – corrected version
"""

import math
import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# ── Parabolic Ball ──


class ParabolicBall:
    """
    Generate one random parabola inside realistic launch parameter ranges.
    Coordinates:
        x - depth (towards / away from the front camera)
        y - horizontal (left / right from the front camera)
        z - vertical
    """

    # parameter ranges  (min, max)
    x0_range = (12.0, 18.0)   # m
    y0_range = (-7.0, 7.0)    # m
    z0_range = (0.0, 1.5)     # m
    v0_range = (12.0, 20.0)   # m/s
    elev_range = (15.0, 55.0)     # °
    azim_range = (0.0, 15.0)      # °
    g = 9.81  # m/s²

    def __init__(self, rng: random.Random | None = None) -> None:
        r = rng or random.Random()
        # initial position
        self.x0 = r.uniform(*self.x0_range)
        self.y0 = r.uniform(*self.y0_range)
        self.z0 = r.uniform(*self.z0_range)
        # launch speed & direction
        self.v0 = r.uniform(*self.v0_range)
        elev = math.radians(r.uniform(*self.elev_range))
        azim = math.radians(r.uniform(*self.azim_range))
        # velocity components
        vh = self.v0 * math.cos(elev)
        self.vx = -abs(vh * math.cos(azim))     # always flying towards the goal
        self.vy = vh * math.sin(azim)
        self.vz = self.v0 * math.sin(elev)

    # -----------------------------------------------------------------

    def position(self, t: float) -> Tuple[float, float, float]:
        """(x, y, z) in metres after *t* seconds."""
        x = self.x0 + self.vx * t
        y = self.y0 + self.vy * t
        z = self.z0 + self.vz * t - 0.5 * self.g * t * t
        return x, y, z



    @property
    def flight_time(self) -> float:
        """Time until z = 0."""
        disc = self.vz * self.vz + 2.0 * self.g * self.z0
        return (self.vz + math.sqrt(disc)) / self.g


# ─────────────────────────────────────────────── rendering helpers ──

BALL_R_M = 0.11          # radius in metres
F_PIX_AT_1M = 300        # front cam focal length proxy (px)
F_SIDE_PIX_AT_1M = 550   # side cam focal length proxy (px)
CAMERA_Y_OFFSET = 1.0    # front cam sits 1 m left of goal centre
GOAL_CROSSBAR_HEIGHT = 2.44  # m


def make_stripe_pattern(width_px: int, frame_shape: Tuple[int, int, int]
                        ) -> np.ndarray:
    """Green/black vertical stripes – just so motion is visible."""
    h, w, _ = frame_shape
    img = np.zeros((h, w, 3), np.uint8)
    for x in range(w):
        colour = (0, 128, 0) if (x // width_px) % 2 else (0, 0, 0)
        img[:, x] = colour
    return img


def make_frames(
    ball: ParabolicBall,
    *,
    fps: int = 60,
    canvas_side: Tuple[int, int] = (480, 640),
    canvas_front: Tuple[int, int] = (480, 640),
    scale: float = 30.0,
    colour: Tuple[int, int, int] = (255, 255, 255),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Render the entire flight once → two image stacks.
    Returns
        side_stack  - (T, H, W, 3)
        front_stack - (T, H, W, 3)
    """
    z_s, x_s = canvas_side
    z_f, x_f = canvas_front

    stripes = make_stripe_pattern(10, (z_s, x_s, 3))
    shift_per_frame = 2
    dark_green = (0, 128, 0)

    side_frames: List[np.ndarray] = []
    front_frames: List[np.ndarray] = []

    t_idx = 0
    while True:
        t = t_idx / fps
        x_m, y_m, z_m = ball.position(t)

        if x_m < 0.1 or z_m < 0:          # ball disappeared
            break

        # -------------- side view (x – depth vs z – height) -------------
        side = np.full((z_s, x_s, 3), dark_green, np.uint8)
        dx = (shift_per_frame * t_idx) % x_s
        side[:] = np.roll(stripes, -dx, axis=1)

        xi = x_s - int(x_m * scale)               # invertible mapping
        zi = z_s - int(z_m * scale)
        r_side_px = int(F_SIDE_PIX_AT_1M * BALL_R_M
                        / (CAMERA_Y_OFFSET + abs(y_m)))

        if 0 <= xi < x_s and 0 <= zi < z_s:
            cv2.circle(side, (xi, zi), r_side_px, colour, -1)

        # ---------------- front view (y – lateral vs z – height) --------
        front = np.full((z_f, x_f, 3), dark_green, np.uint8)

        r_pix = int(F_PIX_AT_1M * BALL_R_M / max(x_m, 1e-6))
        cx = int(y_m * scale + x_f / 2)
        cy = int((GOAL_CROSSBAR_HEIGHT - z_m) * scale + z_f / 2)

        if 0 <= cx < x_f and 0 <= cy < z_f:
            cv2.circle(front, (cx, cy), r_pix, colour, -1)

        side_frames.append(side)
        front_frames.append(front)
        t_idx += 1

    return np.stack(side_frames), np.stack(front_frames)


def save_png_stack(stack: np.ndarray, out_dir: str | Path, prefix: str) -> None:
    """Dump each frame to dir/prefix_0000.png …"""
    out_dir = Path(out_dir)
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(stack):
        ok = cv2.imwrite(str(out_dir / f"{prefix}_{i:04d}.png"), frame)
        if not ok:
            raise IOError(f"cv2.imwrite failed for frame {i}")


#  Tracker 


class Tracker:
    """
    Re-detect the ball in each image pair and triangulate (x, y, z).
    """

    def __init__(
        self,
        *,
        side_dir: str | Path,
        front_dir: str | Path,
        fps: float = 60.0,
        scale: float = 30.0,
        ball_radius: float = 0.11,
        f_front: float = 300.0,
    ) -> None:
        self.side_dir = Path(side_dir)
        self.front_dir = Path(front_dir)
        self.fps = fps
        self.scale = scale
        self.ball_radius = ball_radius
        self.f_front = f_front

        self.side_files = sorted(
            f for f in os.listdir(self.side_dir) if f.lower().endswith(".png")
        )
        self.front_files = sorted(
            f for f in os.listdir(self.front_dir) if f.lower().endswith(".png")
        )
        if not self.side_files or not self.front_files:
            raise ValueError("Side or front directory is empty.")

        first_side = cv2.imread(str(self.side_dir / self.side_files[0]))
        first_front = cv2.imread(str(self.front_dir / self.front_files[0]))
        if first_side is None or first_front is None:
            raise ValueError("Failed to read first PNG.")

        self.z_s, self.x_s = first_side.shape[:2]
        self.z_f, self.x_f = first_front.shape[:2]

    # -----------------------------------------------------------------

    def _detect_circle(self, img: np.ndarray) -> Tuple[float, float, float] | None:
        """
        Return (cx, cy, r) of the biggest bright blob, or None if nothing found.
        Uses HSV thresholding – works with the synthetic white ball.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 0, 200), (180, 40, 255))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        (cx, cy), r = cv2.minEnclosingCircle(max(cnts, key=cv2.contourArea))
        return cx, cy, r


    def track(self) -> List[Tuple[float, float, float, float]]:
        """
        Reconstruct (t, x, y, z) for every frame that shows the ball in
        at least one view.  Stop when neither view sees it any more,
        and print status messages when a camera loses / regains sight.
        """
        n_frames = max(len(self.side_files), len(self.front_files))
        positions: List[Tuple[float, float, float, float]] = []

        last_x = last_y = last_z = None

        # remember the visibility state so we only print on **changes**
        side_visible  = True
        front_visible = True

        for i in range(n_frames):
            t = i / self.fps
            side_img  = cv2.imread(str(self.side_dir  / self.side_files[i]))
            front_img = cv2.imread(str(self.front_dir / self.front_files[i]))
            if side_img is None or front_img is None:
                break                    # ran out of frames on disk

            det_side  = self._detect_circle(side_img)
            det_front = self._detect_circle(front_img)

            # ——— status messages when visibility changes —————————
            if det_side is None and side_visible:
                print(f"[{t:6.3f}s] Ball out of frame (side camera)")
                side_visible = False
            elif det_side is not None and not side_visible:
                print(f"[{t:6.3f}s] Ball re-appeared (side camera)")
                side_visible = True

            if det_front is None and front_visible:
                print(f"[{t:6.3f}s] Ball out of frame (front camera)")
                front_visible = False
            elif det_front is not None and not front_visible:
                print(f"[{t:6.3f}s] Ball re-appeared (front camera)")
                front_visible = True
            # ————————————————————————————————————————————————

            # stop if neither camera sees the ball in this frame
            if det_side is None and det_front is None:
                print(f"[{t:6.3f}s] Ball lost in both views – stopping")
                break

            # ── proceed with one- or two-view triangulation ───────────
            if det_side is not None:
                cx_s, cy_s, _ = det_side
                x_m_side = (self.x_s - cx_s) / self.scale
                z_side   = (self.z_s - cy_s) / self.scale
            else:
                x_m_side = last_x
                z_side   = last_z

            if det_front is not None:
                cx_f, cy_f, r_f = det_front
                y_m     = (cx_f - self.x_f / 2) / self.scale
                z_front = GOAL_CROSSBAR_HEIGHT - (cy_f - self.z_f / 2) / self.scale
                x_m_fr  = self.f_front * self.ball_radius / max(r_f, 1e-6)
            else:
                y_m     = last_y
                z_front = last_z
                x_m_fr  = last_x

            z_m = (
                0.5 * (z_side + z_front)
                if det_side is not None and det_front is not None
                else z_side if det_side is not None
                else z_front
            )
            x_m = (
                0.5 * (x_m_side + x_m_fr)
                if det_side is not None and det_front is not None
                else x_m_side if det_side is not None
                else x_m_fr
            )

            if x_m is None or z_m is None or x_m < 0.1 or z_m < 0:
                continue

            positions.append((t, x_m, y_m, z_m))
            last_x, last_y, last_z = x_m, y_m, z_m

        return positions



# ─────────────────────────────────────────────── main / demo ────────


rng = random.Random()          # reproducible demo
ball = ParabolicBall(rng)

side_stack, front_stack = make_frames(ball, fps=60)
save_png_stack(side_stack,  "side_frames",  "side")
save_png_stack(front_stack, "front_frames", "front")

tracker = Tracker(
    side_dir="side_frames",
    front_dir="front_frames",
    fps=60.0,
    scale=30.0,
    ball_radius=BALL_R_M,
    f_front=F_PIX_AT_1M,
)
trajectory = tracker.track()

# ---------- diagnostics ----------
start = ball.position(0.0)
t_land = ball.flight_time
end = ball.position(t_land)

print("=== ParabolicBall flight parameters ===")

print(f"Launch position:  (x, y, z) = {tuple(round(v, 3) for v in start)}")
print(f"Landing time:     t = {t_land:.3f} s")
print(f"Landing position: (x, y, z) = {tuple(round(v, 3) for v in end)}\n")

# Print the ball.positions of the whole flight
for t in range(int(t_land * 60) + 1):
    x, y, z = ball.position(t / 60.0)
    # stop when ball is close to the camera plane (x ≈ 0 m)
    if x < 0.0:
        break
    position = (t / 60.0, x, y, z)

    # Print position at x = 0.0
    if math.isclose(x, 0.0, abs_tol=0.1):
        print(f"Ball at x = 0.0 m: t={t/60.0:6.3f}  y={position[2]:6.3f}  z={position[3]:6.3f}")

print("=== reconstructed (t, x, y, z) ===")
for t, x, y, z in trajectory:
    print(f"t={t:6.3f}  x={x:7.3f}  y={y:6.3f}  z={z:6.3f}")
