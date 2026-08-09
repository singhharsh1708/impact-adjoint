module ContactSolver

using LinearAlgebra
using ForwardDiff

# 2D ballistic flight over smooth terrain with impact events.
#
# State q = (x, y, vx, vy). Smooth flight q' = f(q) (gravity + linear drag).
# Guard g(q, θ) = y - h(x; θ_terrain) hits zero at impact; reset map applies
# Newtonian restitution e on the normal component and retention (1 - μ) on the
# tangential component of velocity, in the local terrain frame.
#
# Sensitivities: forward variational equation X' = ∂f/∂q · X on smooth
# segments; at events X jumps by the saltation update
#   X⁺ = R_q X⁻ + R_θ - (f⁺ - R_q f⁻) τ_θᵀ,
#   τ_θ = -(X⁻ᵀ g_q + g_θ) / (g_q · f⁻),
# (equivalent to the saltation matrix S = R_q + (f⁺ - R_q f⁻) g_qᵀ / (g_q · f⁻)
# applied to X⁻, plus the explicit R_θ and g_θ event terms),
# which naive autodiff through the integrator misses. Local partials
# (∂f/∂q, g_q, g_θ, R_q, R_θ) come from ForwardDiff dual numbers; the event
# structure is handled analytically.
#
# Termination (status output):
#   0  ran to t_final.
#   1  event capacity: a crossing was located while nev == max_events. The
#      simulation stops AT that crossing; qf is the pre-impact state there and
#      Jqf is its total θ-derivative (X⁻ + f⁻ τ_θᵀ, event time included).
#   2  contact settled: the post-impact normal velocity fell below v_stop
#      (near-Zeno chatter; sticking/sliding contact is outside this model).
#      The simulation stops at the impact; qf is the post-impact state and
#      Jqf its total θ-derivative R_q (X⁻ + f⁻ τ_θᵀ) + R_θ.
# In both truncated modes qf is measured at the event time (t_end), not at
# t_final, and the reported Jacobian accounts for the event-time dependence,
# so gradient-based optimization remains well-posed across the truncation.
#
# Event detection: endpoint sign change of the guard per RK4 step, plus two
# interior probes (dtau/3, 2dtau/3) to catch dips narrower than one step.
# Terrain features narrower than |vx|·dt/3 can still be stepped over —
# choose dt ≲ min(wid) / (3 |vx|).
#
# Parameter vector θ (length NTH):
#   1:2  v0 (launch velocity)
#   3    y0 (launch height, x0 = 0 fixed)
#   4    e  (normal restitution)
#   5    μ  (tangential loss factor)
#   6:8  amp (Gaussian bump amplitudes)
#   9:11 ctr (bump centers)
#  12:14 wid (bump widths)

const GRAV = 9.81
# Number of terrain bumps is derived from the parameter vector length:
# any nb >= 1 works, so the terrain can be an arbitrary-dimensional height
# field (θ has length 5 + 3 nb).
@inline nbumps(θ) = (length(θ) - 5) ÷ 3

@inline function terrain_h(x, amp, ctr, wid)
    h = zero(x) * zero(eltype(amp))
    @inbounds for i in eachindex(amp)
        h += amp[i] * exp(-(x - ctr[i])^2 / (2 * wid[i]^2))
    end
    return h
end

@inline function unpack_terrain(θ)
    nb = nbumps(θ)
    return (view(θ, 6:5 + nb), view(θ, 6 + nb:5 + 2nb), view(θ, 6 + 2nb:5 + 3nb))
end

function guard(q, θ)
    amp, ctr, wid = unpack_terrain(θ)
    return q[2] - terrain_h(q[1], amp, ctr, wid)
end

flow(q, cd) = [q[3], q[4], -cd * q[3], -GRAV - cd * q[4]]

function reset_map(q, θ)
    e = θ[4]
    μ = θ[5]
    amp, ctr, wid = unpack_terrain(θ)
    hp = ForwardDiff.derivative(x -> terrain_h(x, amp, ctr, wid), q[1])
    s = sqrt(1 + hp^2)
    n1, n2 = -hp / s, 1 / s
    t1, t2 = 1 / s, hp / s
    vn = q[3] * n1 + q[4] * n2
    vt = q[3] * t1 + q[4] * t2
    vn2 = -e * vn
    vt2 = (1 - μ) * vt
    return [q[1], q[2], vn2 * n1 + vt2 * t1, vn2 * n2 + vt2 * t2]
end

# One RK4 step of the pair (q, X) where X' = A(q) X, A = ∂f/∂q.
function step_qX(q, X, dtau, cd, want_sens::Bool)
    k1 = flow(q, cd)
    q2 = q .+ (dtau / 2) .* k1
    k2 = flow(q2, cd)
    q3 = q .+ (dtau / 2) .* k2
    k3 = flow(q3, cd)
    q4 = q .+ dtau .* k3
    k4 = flow(q4, cd)
    qn = q .+ (dtau / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
    if !want_sens
        return qn, X
    end
    A1 = ForwardDiff.jacobian(z -> flow(z, cd), q)
    A2 = ForwardDiff.jacobian(z -> flow(z, cd), q2)
    A3 = ForwardDiff.jacobian(z -> flow(z, cd), q3)
    A4 = ForwardDiff.jacobian(z -> flow(z, cd), q4)
    K1 = A1 * X
    K2 = A2 * (X .+ (dtau / 2) .* K1)
    K3 = A3 * (X .+ (dtau / 2) .* K2)
    K4 = A4 * (X .+ dtau .* K3)
    Xn = X .+ (dtau / 6) .* (K1 .+ 2 .* K2 .+ 2 .* K3 .+ K4)
    return qn, Xn
end

# Earliest guard crossing within (0, dtau], or NaN. Checks the endpoint and
# two interior probes so dips narrower than one step are still seen.
function find_crossing(q, X, dtau, cd, θ)
    guard(q, θ) > 0 || return NaN
    qn, _ = step_qX(q, X, dtau, cd, false)
    guard(qn, θ) < 0 && return dtau
    for frac in (1 / 3, 2 / 3)
        sp = frac * dtau
        qp, _ = step_qX(q, X, sp, cd, false)
        guard(qp, θ) < 0 && return sp
    end
    return NaN
end

"""
    run_solver(θ, cd, t_final, dt, max_events, n_samples, v_stop, want_sens)

Returns (qf, Jqf, impact_x, Jimp, n_events, traj, status, t_end):
  qf        state at t_end (4,)
  Jqf       ∂qf/∂θ (4, NTH); at truncation (status != 0) this is the total
            derivative including event-time dependence (zeros when !want_sens)
  impact_x  x of each impact, zero-padded (max_events,); padded rows of Jimp
            are exactly zero (documented convention)
  Jimp      ∂impact_x/∂θ (max_events, NTH)
  n_events  number of impacts
  traj      (n_samples, 5) rows (t, x, y, vx, vy) evenly sampled from the grid
  status    0 ok / 1 event capacity / 2 contact settled (see module docs)
  t_end     time of qf (== t_final iff status == 0)
"""
function run_solver(θvec, cd, t_final, dt, max_events::Int, n_samples::Int, v_stop, want_sens::Bool)
    θ = collect(Float64, θvec)
    nth = length(θ)
    (nth >= 8 && (nth - 5) % 3 == 0) || error("θ must have length 5 + 3*nb, got $nth")
    q = [0.0, θ[3], θ[1], θ[2]]
    X = zeros(4, nth)
    X[2, 3] = 1.0
    X[3, 1] = 1.0
    X[4, 2] = 1.0
    impact_x = zeros(max_events)
    Jimp = zeros(max_events, nth)
    nev = 0
    status = 0
    t = 0.0
    hist_t = [0.0]
    hist_q = [copy(q)]

    guard(q, θ) > 0 || error("initial state must start above the terrain")

    while t < t_final - 1e-12 && status == 0
        dtau = min(dt, t_final - t)
        s_hit = find_crossing(q, X, dtau, cd, θ)
        if isnan(s_hit)
            q, X = step_qX(q, X, dtau, cd, want_sens)
            t += dtau
        else
            # Bisect the step size s ∈ (0, s_hit) to the crossing.
            lo, hi = 0.0, s_hit
            for _ in 1:80
                mid = (lo + hi) / 2
                qm, _ = step_qX(q, X, mid, cd, false)
                if guard(qm, θ) > 0
                    lo = mid
                else
                    hi = mid
                end
            end
            s = (lo + hi) / 2
            qminus, Xminus = step_qX(q, X, s, cd, want_sens)
            fminus = flow(qminus, cd)
            gq = ForwardDiff.gradient(z -> guard(z, θ), qminus)
            denom = dot(gq, fminus)
            abs(denom) > 1e-12 || error("degenerate guard crossing at t=$(t + s)")
            denom < 0 || error("guard crossing with non-approaching velocity at t=$(t + s)")
            τθ = want_sens ?
                -(Xminus' * gq .+ ForwardDiff.gradient(z -> guard(qminus, z), θ)) ./ denom :
                zeros(nth)
            t += s

            if nev >= max_events
                # Capacity reached: stop at the pre-impact state, total derivative.
                q = qminus
                want_sens && (X = Xminus .+ fminus * τθ')
                status = 1
            else
                qplus = reset_map(qminus, θ)
                # Land a hair ABOVE the guard so detection re-arms even for
                # sub-dt hops (guard(q+) > 0 strictly). The lift is far below
                # every validation tolerance and is applied outside reset_map,
                # so the sensitivity propagation is unaffected: the propagated
                # tangent lies in the event manifold, where the lifted and
                # unlifted reset maps agree to first order.
                amp, ctr, wid = unpack_terrain(θ)
                qplus[2] = terrain_h(qplus[1], amp, ctr, wid) + 1e-13 * (1 + abs(qplus[2]))
                fplus = flow(qplus, cd)
                nev += 1
                impact_x[nev] = qminus[1]
                want_sens && (Jimp[nev, :] .= Xminus[1, :] .+ fminus[1] .* τθ)
                vn_plus = dot(gq, fplus) / norm(gq)
                if vn_plus < v_stop
                    # Chatter accumulation: model validity ends, stop at the event.
                    if want_sens
                        Rq = ForwardDiff.jacobian(z -> reset_map(z, θ), qminus)
                        Rθ = ForwardDiff.jacobian(z -> reset_map(qminus, z), θ)
                        X = Rq * (Xminus .+ fminus * τθ') .+ Rθ
                    end
                    q = qplus
                    status = 2
                else
                    if want_sens
                        Rq = ForwardDiff.jacobian(z -> reset_map(z, θ), qminus)
                        Rθ = ForwardDiff.jacobian(z -> reset_map(qminus, z), θ)
                        X = Rq * Xminus .- (fplus .- Rq * fminus) * τθ' .+ Rθ
                    end
                    q = qplus
                end
            end
        end
        push!(hist_t, t)
        push!(hist_q, copy(q))
    end

    traj = zeros(n_samples, 5)
    if n_samples > 0
        m = length(hist_t)
        idxs = n_samples == 1 ? [m] : round.(Int, range(1, m; length=n_samples))
        for (r, idx) in enumerate(idxs)
            traj[r, 1] = hist_t[idx]
            traj[r, 2:5] .= hist_q[idx]
        end
    end

    return q, X, impact_x, Jimp, nev, traj, status, t
end

end # module
