A minimal, reproducible 2‑camera football‑trajectory simulator + tracker

Render a synthetic shot, save it to PNG/video, then re‑detect the ball and reconstruct the 3‑D path—all in pure Python & OpenCV.

Physics – A random but realistic launch is sampled (initial position, velocity, elevation, azimuth) and solved analytically under gravity.

Rendering – The ball is drawn frame‑by‑frame from two virtual cameras (side & front) onto green‑striped backgrounds.  The result is a pair of PNG or MP4 stacks.

Detection & triangulation – In a second pass, the script redetects the ball in each image, converts pixel positions/radii back to metres and fuses both views to recover the full 3‑D trajectory.
