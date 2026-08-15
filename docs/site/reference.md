# Component reference

Two Tesseracts compose in the pipeline, plus a minimal third used only for the
day-one boundary proof.

## contact-sim

Julia solver, all differentiable endpoints: `apply`, `jacobian`,
`jacobian_vector_product`, `vector_jacobian_product`, `abstract_eval`.

### Inputs

```{list-table}
:header-rows: 1
:widths: 14 12 74

* - name
  - type
  - meaning
* - `v0`
  - diff, (2,)
  - launch velocity. Launch position is `(0, y0)`.
* - `y0`
  - diff, scalar
  - launch height, must start above the terrain
* - `e`
  - diff, scalar
  - normal restitution, in (0, 1]
* - `mu`
  - diff, scalar
  - tangential loss factor, in [0, 1)
* - `amp`, `ctr`, `wid`
  - diff, (nb,)
  - Gaussian bump amplitudes, centres and widths. Any `nb >= 1`; the three
    must share one length. The design vector in E4, E5 and E5b.
* - `drag`
  - scalar
  - linear drag coefficient, not differentiated
* - `t_final`, `dt`
  - scalar
  - horizon and integrator step. Terrain features narrower than
    `|vx|·dt/3` can be stepped over, so choose `dt <= min(wid)/(3|vx|)`.
* - `n_samples`
  - int
  - trajectory rows returned for plotting, 0 to skip
* - `v_stop`
  - scalar
  - normal-velocity floor below which contact is declared settled
```

Every documented bound is enforced by the schema, so an out-of-range
restitution is rejected rather than silently integrated.

### Outputs

```{list-table}
:header-rows: 1
:widths: 16 12 72

* - name
  - type
  - meaning
* - `qf`
  - diff, (4,)
  - state `(x, y, vx, vy)` at `t_end`. At truncation its Jacobian is the
    total derivative including event-time dependence.
* - `impact_x`
  - diff, (8,)
  - x-coordinate of each impact, zero-padded. Only `[:n_events]` is
    meaningful; padded entries are exactly 0 with exactly-zero derivative
    rows, so gradient checkers stay coherent.
* - `n_events`
  - int32
  - number of impacts
* - `traj`
  - (n_samples, 5)
  - sampled `(t, x, y, vx, vy)` rows
* - `status`
  - int32
  - 0 ran to `t_final`, 1 event capacity, 2 settled contact
* - `t_end`
  - scalar
  - time at which `qf` is measured
```

## score-target

JAX objective built with the `tesseract init --recipe jax` endpoints. Takes
`qf`, a `target` position and three `weights`, and returns a `loss` and a
`miss_distance`.

## julia_kernel

The minimal day-one boundary proof: a Julia function with a hand-written
adjoint, used by `scripts/proof_local.py` and `scripts/proof_container.py` to
show `jax.grad` crossing the boundary and matching the analytic derivative to
1e-12. Not part of the E1 to E6 pipeline.
