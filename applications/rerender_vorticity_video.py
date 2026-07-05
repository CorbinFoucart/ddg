"""Re-render a vorticity video from saved snapshots (.npz) with new
colour limits.

The profile script ``profile_flow_past_square_large.py`` writes both an
MP4 and a ``.npz`` of raw snapshots. This script reloads the .npz and
re-renders, useful when the original render's auto-vmax was dominated
by obstacle-corner singularities.

Usage:
  python -u scripts/rerender_vorticity_video.py \\
      --in path/to/flow_past_square_v2_re100.npz \\
      --out flow_past_square_v2_re100_clipped.mp4 \\
      --vmax 20 --fps 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Build a tiny mesh stand-in that satisfies the parts of the API that
# save_scalar_field_animation_2d uses (just .x, .y, .r, .s, .Np, .K).
class _MeshStub:
    def __init__(self, x, y, r, s):
        self.x = x; self.y = y
        self.r = r; self.s = s
        self.Np, self.K = x.shape


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_npz", required=True,
                   help="Input .npz produced by profile_flow_past_square_large.py")
    p.add_argument("--out", required=True, help="Output .mp4 path")
    p.add_argument("--vmax", type=float, default=None,
                   help="Symmetric color limit (overrides percentile clipping)")
    p.add_argument("--clip-percentile", type=float, default=99.0,
                   help="Use the p-th percentile of |field| as vmax (default 99). "
                        "Ignored if --vmax is given.")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--cmap", default="RdBu_r")
    p.add_argument("--title", default="Vorticity ω")
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from ddg.dg2d._anim_utils_2d import save_scalar_field_animation_2d

    data = np.load(args.in_npz)
    field_traj = data["field_traj"]                # (n_frames, Np, K)
    times = data["times"]
    mesh = _MeshStub(data["mesh_x"], data["mesh_y"], data["mesh_r"], data["mesh_s"])

    print(f"  field_traj shape: {field_traj.shape}  |  times: {len(times)}")
    print(f"  global |ω| max:   {float(np.max(np.abs(field_traj))):.3f}")
    print(f"  99th percentile:  {float(np.percentile(np.abs(field_traj), 99)):.3f}")
    print(f"  95th percentile:  {float(np.percentile(np.abs(field_traj), 95)):.3f}")
    if args.vmax is not None:
        print(f"  → using vmax = {args.vmax} (explicit)")
    else:
        print(f"  → using clip_percentile = {args.clip_percentile}")

    save_scalar_field_animation_2d(
        mesh, field_traj, times,
        output_path=args.out,
        title=args.title,
        cmap=args.cmap,
        fps=args.fps,
        figsize=(9.0, 3.5),
        symmetric=True,
        show_mesh=False,
        vmax=args.vmax,
        clip_percentile=args.clip_percentile,
    )
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
