
# impact-adjoint

Exact gradients through impact events, across a Julia and JAX Tesseract
boundary.

:::{container} ia-byline
**Harsh Singh** · independent work

Tesseract Hackathon 2026 entry, Track 1: inverse design and shape
optimization. Version {{ version }}.
:::

:::{container} ia-cta
[Get started](getting-started.md){.ia-cta-primary}
[Write-up](https://github.com/singhharsh1708/impact-adjoint/blob/main/docs/writeup.md)
[Code](https://github.com/singhharsh1708/impact-adjoint)
[Results](results.md)
[Cite](citing.md)
:::

:::{container} ia-abstract
Simulations of contact are easy to write and hard to differentiate.
impact-adjoint computes parameter sensitivities across impact events that
are exact for the hybrid system at fixed event topology, using the classical
saltation matrix, and serves them from a Julia solver
through a Tesseract component boundary, so a JAX program obtains the true
derivative without doing any event handling of its own. The resulting
gradients design passive structures whose function exists only because of
impacts.
:::

```{raw} html
<div class="ia-claim">
Simulate a bouncing ball in JAX the natural way, applying the impact at the
integrator step where it is detected, and <code>jax.grad</code> returns
<strong class="ia-zero">d x(T)/d v0y = 0.0</strong> exactly on flat terrain.
The true value is <strong class="ia-zero">+0.09</strong>.
</div>
```

The gradient is not noisy or approximate. It is confidently, silently wrong,
and refining the step size does not help, because the step at which the event
fires is an integer and autodiff cannot differentiate it.

(demo)=
## Demo

```{raw} html
<figure class="ia-video" id="fig-demo">
  <video controls preload="metadata" width="892"
         poster="_static/demo_poster.jpg"
         aria-describedby="fig-demo-caption">
    <source src="_static/demo.mp4" type="video/mp4">
    <p>Your browser cannot play this video.
       <a href="_static/demo.mp4">Download it instead.</a></p>
  </video>
  <figcaption id="fig-demo-caption">
    A 73-second walkthrough: the silent failure, the Julia and JAX boundary,
    the boundary proof run live, the two designs, and the four oracles. Every
    number on every card is read from <code>results.json</code>, and the
    terminal segment is a real run captured when the video is built.
  </figcaption>
</figure>
```

## See it fail, then see it fixed

Both snippets below run as written. What differs is where the impact is
applied; the second also enables x64, because the component returns Float64
and JAX refuses to hand those back without it.

::::{grid} 2
:gutter: 3
:class-container: ia-faildemo

:::{grid-item-card} The natural JAX program
:class-card: ia-fail
:class-header: ia-fail-head

The impact is applied at the integrator step where it is detected.

```python
def bounce(v0y, n=2000):
    dt = 2.0 / n

    def step(q, _):
        qn = rk4(q, dt)
        hit = (qn[1] < 0) & (q[1] > 0)
        qn = jnp.where(hit, reset(qn), qn)
        return qn, None

    q0 = jnp.array([0., 1., 2., v0y])
    qf, _ = jax.lax.scan(
        step, q0, None, length=n)
    return qf[0]

jax.grad(bounce)(0.5)
# 0.0
```

The step at which the event fires is an integer. Autodiff differentiates a
staircase, and returns zero at every `dt`. Hand-writing an interpolated event
inside the program does recover a converging gradient, to within 0.05% at
this step size and 0.3% across the whole sweep;
what the component removes is the derivation, not the last three digits.
:::

:::{grid-item-card} The same call through impact-adjoint
:class-card: ia-fix
:class-header: ia-fix-head

The impact is applied at the crossing, and its time carries a derivative.

```python
jax.config.update("jax_enable_x64", True)

API = ("tesseracts/contact_sim"
       "/tesseract_api.py")
sim = Tesseract.from_tesseract_api(API)

def bounce(v0y):
    cfg = {**CFG,
           "v0": jnp.array([2.0, v0y])}
    out = apply_tesseract(sim, cfg)
    return out["qf"][0]

jax.grad(bounce)(0.5)
# 0.09037773625811413
```

The saltation matrix supplies the missing term at each impact. Exact at any
`dt`, with no event handling in the JAX program.
:::

::::

## Move the slider

```{raw} html
<div id="ia-sweep"></div>
```

## What the gradients build

```{raw} html
<figure class="ia-video" id="fig-sorter">
  <video controls autoplay muted loop playsinline preload="metadata"
         poster="_static/e5_learning_poster.webp" width="892" height="408"
         aria-describedby="fig-sorter-caption">
    <source src="_static/e5_learning.mp4" type="video/mp4">
    <img src="_static/e5_learning_poster.webp" width="892" height="408"
         alt="The converged surface: two particles that entered identically
              separate over the designed bumps into their own bins.">
  </video>
  <figcaption id="fig-sorter-caption">
    One terrain, two materials, same inlet. Gradient descent through every
    impact of both trajectories reshapes the surface until restitution alone
    routes each particle to its own bin. The poster frame is the converged
    design.
  </figcaption>
</figure>
<script>
  // Readers who ask for reduced motion get the converged frame, not the loop.
  (() => {
    const v = document.querySelector(".ia-video video");
    if (v && window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      v.removeAttribute("autoplay");
      v.removeAttribute("loop");
      v.pause();
    }
  })();
</script>
```

```{stat-cards}
```

## Reproduce it

CPU only, no Docker, no GPU. The four checks take {{ checks_walltime }} on a
warm Julia depot, give or take what else the machine is doing. The first run
adds a few minutes of `pip install` and a Julia bootstrap that runs in two
stages, one on the first call and one the first time `ForwardDiff` is needed,
so budget nearer fifteen minutes from a genuinely cold machine.

{{ repo_note }}

```bash
git clone https://github.com/singhharsh1708/impact-adjoint
cd impact-adjoint
python3 -m venv .venv && source .venv/bin/activate
pip install -r docs/requirements-repro.txt

python scripts/proof_local.py          # boundary proof
python scripts/validate_closed_form.py # symbolic oracle
python scripts/validate_reference.py   # scipy oracle
python experiments/e3_naive_vs_saltation.py
```

The last command is the claim at the top of this page: it prints the
grid-reset gradient as exactly `0.0` at every step size, next to the
saltation gradient. Full instructions are in
[getting started](getting-started.md).

## Where to start

::::{grid} 3
:gutter: 2

:::{grid-item-card} The problem
:link: problem
:link-type: doc

Why autodiff returns a wrong gradient at an impact, what the pure-JAX repair
costs, and what Diffrax can and cannot do about it today.
:::

:::{grid-item-card} The method
:link: method
:link-type: doc

The saltation jump condition, the component boundary that carries it, and the
termination semantics that keep optimizers honest.
:::

:::{grid-item-card} The evidence
:link: studies
:link-type: doc

Seven verification studies and eight experiments, every number generated from a
committed artifact.
:::

::::

## How it is built

```{list-table}
:header-rows: 1
:widths: 26 74

* - Item
  - Detail
* - Track
  - 1, inverse design and shape optimization
* - Components
  - `contact-sim` (Julia) and `score-target` (JAX), composed under one `jax.grad`
* - Gradient method
  - forward variational sensitivities with analytic saltation updates at events
* - Checked against
  - a symbolic closed form and an independent scipy reimplementation, plus
    golden regression tests on every push
```

```{toctree}
:hidden:
:caption: Orientation

problem
method
getting-started
```

```{toctree}
:hidden:
:caption: Evidence

experiments
studies
results
```

```{toctree}
:hidden:
:caption: Reference

reference
artifacts
related
upstream
limitations
changelog
citing
```
