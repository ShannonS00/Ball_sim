import pytest
import math
import random
import tempfile
from pathlib import Path
import numpy as np                  # convenience, not strictly required

from example import ParabolicBall, make_frames, save_png_stack, Tracker, BALL_R_M, F_PIX_AT_1M



def test_tracker_matches_ground_truth(tmp_path: Path, seed: int= 42) -> None:
    """
    The tracker's reconstructed (x, y, z) must stay within ±0.30 m of the
    analytical trajectory for every frame that the tracker returns.
    """
    # ------ 1. generate one reproducible synthetic trajectory -----------

    rng = random.Random(seed)
    ball = ParabolicBall(rng)

    side, front = make_frames(ball, fps=60)          # (T, H, W, 3)

    # ------ 2. dump PNG stacks into a throw-away directory --------------
    side_dir  = tmp_path / "side"
    front_dir = tmp_path / "front"
    save_png_stack(side,  side_dir,  "side")
    save_png_stack(front, front_dir, "front")

    # ------ 3. run the tracker -----------------------------------------
    tracker = Tracker(
        side_dir=side_dir,
        front_dir=front_dir,
        fps=60.0,
        scale=30.0,
        ball_radius=BALL_R_M,
        f_front=F_PIX_AT_1M,
    )
    tol = 3
    tracked = tracker.track()   # List[(t, x, y, z)]
    for t, x_trk, y_trk, z_trk in tracked:
        # ➋ compute the ground-truth for *this* t
        x_gt, y_gt, z_gt = ball.position(t)

        # ➌ compare like with like
        assert math.isclose(x_trk, x_gt, abs_tol=tol), f"x @ t={t:.3f}"
        assert math.isclose(y_trk, y_gt, abs_tol=tol), f"y @ t={t:.3f}"
        assert math.isclose(z_trk, z_gt, abs_tol=tol), f"z @ t={t:.3f}"

