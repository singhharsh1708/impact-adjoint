# Upstream fixes from this work

Problems found while building this and reported upstream during the hackathon
period. Two of them are silent wrong gradients in Tesseract's own AD path. Both Tesseract fixes and both Mosaic harness fixes are merged; of the three
issues, two are closed and the juliacall deadlock remains open.

```{list-table}
:header-rows: 1
:widths: 26 74

* - Where
  - What
* - [core#666](https://github.com/pasteurlabs/tesseract-core/issues/666) /
    [PR 667](https://github.com/pasteurlabs/tesseract-core/pull/667)
  - The experimental VJP cache compared keys by hash alone with no stored key,
    so a collision silently served another input's backward pass. Reproduced
    on the shipped `vectoradd_jax` example using `hash(-1) == hash(-2)`, where
    both are valid norm orders. A second bug in the same path broke `apply`
    for non-JAX inputs.
* - [jax#235](https://github.com/pasteurlabs/tesseract-jax/issues/235) /
    [PR 236](https://github.com/pasteurlabs/tesseract-jax/pull/236)
  - `jax.jvp` shipped the tangent under the wrong list index when only some
    elements of a `list[Differentiable[...]]` input were differentiated, so
    forward and reverse mode disagreed with no error raised.
* - [jax#234](https://github.com/pasteurlabs/tesseract-jax/issues/234)
  - Jitted callbacks deadlock a Tesseract that embeds an in-process Julia
    runtime, which is why E2b runs against the container. Filed with a
    [minimal reproducer](https://github.com/singhharsh1708/tesseract-jax-234-repro)
    isolating the trigger to Julia allocation inside an off-main-thread
    callback.
* - [mosaic#126](https://github.com/pasteurlabs/mosaic/pull/126),
    [#141](https://github.com/pasteurlabs/mosaic/pull/141)
  - Harness fixes: Docker setup diagnostics instead of a raw stderr dump, and
    anomaly status reporting from `min_cosine`.
```
