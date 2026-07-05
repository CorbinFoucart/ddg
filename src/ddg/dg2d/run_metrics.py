"""Run-level metrics aggregation + reporter.

Collects per-step physics + performance metrics during an INS run and
serialises three artefacts at the end:

  * **``<name>_summary.json``** — run-level totals, max-norms, wall-times,
    pairwise scheme distances (if multiple).
  * **``<name>_timeseries.csv``** — one row per recorded step:
    ``step, t, wall_dt, ke, enstrophy, divu_l2_rel, max_speed,
    max_vort, mass_balance, [cg_iters_total]``.
  * **``<name>_report.md``** — auto-generated markdown summary card with
    tables + headline numbers, suitable for pasting into PRs / chat.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ddg.dg2d.ins_metrics import (
    divergence_l2_relative,
    enstrophy,
    kinetic_energy,
    mass_flux_through,
    max_speed,
    max_vorticity,
)
from ddg.dg2d.mesh import Mesh2D


# ---------------------------------------------------------------------------
# Per-step records
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    step: int
    t: float
    wall_dt: float          # seconds for THIS step
    ke: float
    enstrophy: float
    divu_l2_rel: float
    max_speed: float
    max_vort: float
    mass_balance: float
    cg_iters: int = 0       # 0 if direct solver

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class SchemeRun:
    name: str
    n_dofs_per_field: int
    n_steps: int
    wall_time_total: float = 0.0
    setup_time: float = 0.0
    step_records: list[StepRecord] = field(default_factory=list)
    # Forces / Strouhal (optional, populated by E-level metrics):
    Cd_series: list[float] | None = None
    Cl_series: list[float] | None = None
    St: float | None = None

    @property
    def per_step_s(self) -> float:
        return self.wall_time_total / max(self.n_steps, 1)

    @property
    def throughput_dof_step_per_sec(self) -> float:
        return self.n_dofs_per_field * self.n_steps / max(self.wall_time_total, 1e-30)


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------

@dataclass
class RunReporter:
    out_dir: Path
    name: str
    mesh: Mesh2D
    n_dofs_per_field: int
    metadata: dict
    record_every: int = 1
    # When measuring mass-flux balance, we need the inflow + outflow
    # mapB-masks. Caller supplies them as part of the metadata.
    inflow_mapB_mask: jax.Array | None = None
    outflow_mapB_mask: jax.Array | None = None
    _runs: list[SchemeRun] = field(default_factory=list)

    def new_run(self, name: str, n_steps: int) -> SchemeRun:
        run = SchemeRun(
            name=name, n_dofs_per_field=self.n_dofs_per_field, n_steps=n_steps,
        )
        self._runs.append(run)
        return run

    def record_step(
        self,
        run: SchemeRun,
        step: int,
        t: float,
        wall_dt: float,
        ux: jax.Array, uy: jax.Array,
        *,
        cg_iters: int = 0,
    ) -> None:
        if (step % self.record_every) and step != run.n_steps:
            return
        mass_balance = 0.0
        if (
            self.inflow_mapB_mask is not None
            and self.outflow_mapB_mask is not None
        ):
            # Inflow contributes negative (n points into the body); outflow
            # contributes positive. For incompressible, sum ≈ 0.
            mass_balance = (
                mass_flux_through(
                    self.mesh, ux, uy, face_mapB_mask=self.inflow_mapB_mask,
                )
                + mass_flux_through(
                    self.mesh, ux, uy, face_mapB_mask=self.outflow_mapB_mask,
                )
            )
        rec = StepRecord(
            step=step, t=t, wall_dt=wall_dt,
            ke=kinetic_energy(self.mesh, ux, uy),
            enstrophy=enstrophy(self.mesh, ux, uy),
            divu_l2_rel=divergence_l2_relative(self.mesh, ux, uy),
            max_speed=max_speed(ux, uy),
            max_vort=max_vorticity(self.mesh, ux, uy),
            mass_balance=mass_balance,
            cg_iters=cg_iters,
        )
        run.step_records.append(rec)

    # ----- serialisation -----

    def write(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._write_json()
        self._write_csv()
        self._write_markdown()

    def _write_json(self) -> None:
        path = self.out_dir / f"{self.name}_summary.json"
        out = {
            "metadata": _to_jsonable(self.metadata),
            "schemes": [
                {
                    "name": r.name,
                    "n_steps": r.n_steps,
                    "wall_time_total": r.wall_time_total,
                    "setup_time": r.setup_time,
                    "per_step_s": r.per_step_s,
                    "throughput_dof_step_per_sec": r.throughput_dof_step_per_sec,
                    "final_ke": r.step_records[-1].ke if r.step_records else None,
                    "final_enstrophy": r.step_records[-1].enstrophy if r.step_records else None,
                    "final_divu_l2_rel": r.step_records[-1].divu_l2_rel if r.step_records else None,
                    "max_speed_run": max((s.max_speed for s in r.step_records), default=0.0),
                    "max_vort_run": max((s.max_vort for s in r.step_records), default=0.0),
                    "Cd_mean_tail": (
                        float(np.mean(r.Cd_series[-len(r.Cd_series) // 2:]))
                        if r.Cd_series else None
                    ),
                    "Cl_mean_tail": (
                        float(np.mean(r.Cl_series[-len(r.Cl_series) // 2:]))
                        if r.Cl_series else None
                    ),
                    "Strouhal": r.St,
                }
                for r in self._runs
            ],
            "pairwise_l2_velocity": _pairwise_l2(self._runs, self.mesh),
        }
        with open(path, "w") as f:
            json.dump(out, f, indent=2)

    def _write_csv(self) -> None:
        path = self.out_dir / f"{self.name}_timeseries.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "scheme", "step", "t", "wall_dt", "ke", "enstrophy",
                "divu_l2_rel", "max_speed", "max_vort",
                "mass_balance", "cg_iters",
            ])
            for r in self._runs:
                for s in r.step_records:
                    w.writerow([
                        r.name, s.step, s.t, s.wall_dt, s.ke, s.enstrophy,
                        s.divu_l2_rel, s.max_speed, s.max_vort,
                        s.mass_balance, s.cg_iters,
                    ])

    def _write_markdown(self) -> None:
        path = self.out_dir / f"{self.name}_report.md"
        md = self._render_markdown()
        with open(path, "w") as f:
            f.write(md)

    def _render_markdown(self) -> str:
        meta = self.metadata
        lines = [f"# {self.name} — INS scheme comparison\n"]
        lines.append("## Setup\n")
        for k, v in meta.items():
            lines.append(f"- **{k}**: `{v}`")
        lines.append("")

        lines.append("## Per-scheme summary\n")
        header = (
            "| scheme | setup (s) | wall (s) | ms/step | "
            "DOF·step/s | max\\|U\\| | max\\|ω\\| | rel ‖∇·U‖ | mass-bal | "
            "C_d̅ | C_l̅ | St |"
        )
        sep = "|" + "---|" * 12
        lines.append(header)
        lines.append(sep)
        for r in self._runs:
            recs = r.step_records
            final = recs[-1] if recs else None
            max_U = max((s.max_speed for s in recs), default=0.0)
            max_W = max((s.max_vort for s in recs), default=0.0)
            divrel = final.divu_l2_rel if final else float("nan")
            mb = final.mass_balance if final else float("nan")
            Cd = (
                f"{np.mean(r.Cd_series[-len(r.Cd_series)//2:]):.3f}"
                if r.Cd_series else "—"
            )
            Cl = (
                f"{np.mean(r.Cl_series[-len(r.Cl_series)//2:]):.3f}"
                if r.Cl_series else "—"
            )
            St = f"{r.St:.3f}" if r.St is not None else "—"
            lines.append(
                f"| {r.name} | {r.setup_time:.1f} | "
                f"{r.wall_time_total:.1f} | {r.per_step_s * 1000:.0f} | "
                f"{r.throughput_dof_step_per_sec:.2e} | "
                f"{max_U:.3f} | {max_W:.3f} | {divrel:.2e} | "
                f"{mb:.2e} | {Cd} | {Cl} | {St} |"
            )
        lines.append("")

        # Pairwise distances
        if len(self._runs) > 1:
            lines.append("## Pairwise final-velocity rel L2 distance\n")
            table = _pairwise_l2(self._runs, self.mesh)
            names = [r.name for r in self._runs]
            lines.append("| | " + " | ".join(names) + " |")
            lines.append("|" + "---|" * (len(names) + 1))
            for i, na in enumerate(names):
                row = [na]
                for j, nb in enumerate(names):
                    if i == j:
                        row.append("—")
                    else:
                        key = f"{na}_vs_{nb}"
                        v = table.get(key)
                        row.append(f"{v:.2e}" if v is not None else "—")
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_jsonable(d: dict) -> dict:
    """Convert numpy / jax scalars to Python natives so json.dump works."""
    out = {}
    for k, v in d.items():
        if hasattr(v, "tolist"):
            out[k] = v.tolist()
        elif isinstance(v, (np.integer, np.floating)):
            out[k] = v.item()
        elif isinstance(v, (tuple, list)):
            out[k] = [
                vv.tolist() if hasattr(vv, "tolist") else
                vv.item() if isinstance(vv, (np.integer, np.floating)) else vv
                for vv in v
            ]
        else:
            out[k] = v
    return out


def _pairwise_l2(runs: list[SchemeRun], mesh: Mesh2D) -> dict[str, float]:
    """L2 distance between pairs of runs' final velocity fields."""
    # We don't store the final fields in the run; the caller must
    # populate them in metadata or pass through. This helper expects
    # them to be attached as run.final_ux / .final_uy attributes.
    out = {}
    for i, a in enumerate(runs):
        if not hasattr(a, "final_ux"):
            return {}
        for b in runs:
            if a is b:
                continue
            num = jnp.sum(
                mesh.J * (
                    (a.final_ux - b.final_ux) ** 2
                    + (a.final_uy - b.final_uy) ** 2
                )
            )
            den = jnp.sum(
                mesh.J * (b.final_ux ** 2 + b.final_uy ** 2)
            ) + 1e-30
            out[f"{a.name}_vs_{b.name}"] = float(jnp.sqrt(num / den))
    return out
