# Component reference

Two Tesseracts compose in the pipeline, plus a minimal third used only for the
day-one boundary proof.

Every table on this page is generated at build time from the component's own
`tesseract_api.py`. The schema is the only source; nothing here is retyped, so
a field cannot be documented with a shape or a differentiability flag it does
not have. Where a field has units or a valid range, the schema states them in
its description and they are reproduced verbatim.

## contact-sim

Julia solver, all differentiable endpoints: `apply`, `jacobian`,
`jacobian_vector_product`, `vector_jacobian_product`, `abstract_eval`.

Terrain is a sum of Gaussian bumps, `h(x) = Σ ampᵢ exp(-(x - ctrᵢ)² / 2 widᵢ²)`,
with `amp`, `ctr` and `wid` sharing one length, so the parameter count follows
the bump count. The guard is `g(q) = y - h(x)`.

### Inputs

```{schema-table} tesseracts/contact_sim/tesseract_api.py InputSchema
```

### Outputs

```{schema-table} tesseracts/contact_sim/tesseract_api.py OutputSchema
```

Padded entries of `impact_x` are exactly zero and carry exactly-zero
derivative rows, so a gradient checker stays coherent across trajectories with
different impact counts. Only `impact_x[:n_events]` is meaningful.

`status` is `0` when the run reached `t_final`, `1` when it hit the event
capacity, and `2` when contact settled. For a nonzero status, `qf` is the
state at the truncating event and its Jacobian is the total derivative
including the event-time dependence, which is what keeps an optimizer from
consuming a silently nonphysical state.

## score-target

JAX objective built with the `tesseract init --recipe jax` endpoints. It takes
the solver's `qf`, a `target` position and three `weights`, and returns the
scalar being optimized along with the miss distance used for reporting.

### Inputs

```{schema-table} tesseracts/score_target/tesseract_api.py InputSchema
```

### Outputs

```{schema-table} tesseracts/score_target/tesseract_api.py OutputSchema
```

## julia_kernel

The minimal day-one boundary proof: a Julia function with a hand-written
adjoint, used by `scripts/proof_local.py` and `scripts/proof_container.py` to
show `jax.grad` crossing the component boundary and matching the analytic
derivative to 1e-12. It computes `y = sin(x) · x²` elementwise, and is not
part of the E1 to E6 pipeline.

### Inputs

```{schema-table} tesseracts/julia_kernel/tesseract_api.py InputSchema
```

### Outputs

```{schema-table} tesseracts/julia_kernel/tesseract_api.py OutputSchema
```
