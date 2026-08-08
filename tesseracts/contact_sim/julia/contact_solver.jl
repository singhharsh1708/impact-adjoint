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
# Parameter vector θ (length NTH):
#   1:2  v0 (launch velocity)
#   3    y0 (launch height, x0 = 0 fixed)
#   4    e  (normal restitution)
#   5    μ  (tangential loss factor)
#   6:8  amp (Gaussian bump amplitudes)
#   9:11 ctr (bump centers)
#  12:14 wid (bump widths)

const GRAV = 9.81
const NB = 3
const NTH = 5 + 3 * NB

@inline function terrain_h(x, amp, ctr, wid)
    h = zero(x) * zero(eltype(amp))
    @inbounds for i in 1:NB
        h += amp[i] * exp(-(x - ctr[i])^2 / (2 * wid[i]^2))
    end
    return h
end

@inline unpack_terrain(θ) = (view(θ, 6:5 + NB), view(θ, 6 + NB:5 + 2NB), view(θ, 6 + 2NB:5 + 3NB))

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

"""
    run_solver(θ, cd, t_final, dt, max_events, n_samples, want_sens)

Returns (qf, Jqf, impact_x, Jimp, n_events, traj):
  qf        final state (4,)
  Jqf       ∂qf/∂θ (4, NTH)          (zeros when !want_sens)
  impact_x  x of each impact, NaN-padded (max_events,)
  Jimp      ∂impact_x/∂θ (max_events, NTH)
  n_events  number of impacts
  traj      (n_samples, 5) rows (t, x, y, vx, vy) evenly sampled from the grid
"""
function run_solver(θvec, cd, t_final, dt, max_events::Int, n_samples::Int, want_sens::Bool)
    θ = collect(Float64, θvec)
    length(θ) == NTH || error("θ must have length $NTH, got $(length(θ))")
    q = [0.0, θ[3], θ[1], θ[2]]
    X = zeros(4, NTH)
    X[2, 3] = 1.0
    X[3, 1] = 1.0
    X[4, 2] = 1.0
    impact_x = fill(NaN, max_events)
    Jimp = zeros(max_events, NTH)
    nev = 0
    t = 0.0
    hist_t = [0.0]
    hist_q = [copy(q)]

    guard(q, θ) > 0 || error("initial state must start above the terrain")

    while t < t_final - 1e-12
        dtau = min(dt, t_final - t)
        qn, Xn = step_qX(q, X, dtau, cd, want_sens)
        if guard(q, θ) > 0 && guard(qn, θ) < 0 && nev < max_events
            # Bisect the step size s ∈ (0, dtau) to the crossing.
            lo, hi = 0.0, dtau
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
            if abs(denom) < 1e-8 * (1 + norm(fminus))
                error("grazing impact at t=$(t + s): event sensitivity undefined")
            end
            denom < 0 || error("guard crossing with non-approaching velocity at t=$(t + s)")
            qplus = reset_map(qminus, θ)
            nev += 1
            impact_x[nev] = qminus[1]
            if want_sens
                gθ = ForwardDiff.gradient(z -> guard(qminus, z), θ)
                τθ = -(Xminus' * gq .+ gθ) ./ denom
                Jimp[nev, :] .= Xminus[1, :] .+ fminus[1] .* τθ
                fplus = flow(qplus, cd)
                Rq = ForwardDiff.jacobian(z -> reset_map(z, θ), qminus)
                Rθ = ForwardDiff.jacobian(z -> reset_map(qminus, z), θ)
                X = Rq * Xminus .- (fplus .- Rq * fminus) * τθ' .+ Rθ
            end
            q = qplus
            t += s
        else
            q = qn
            X = Xn
            t += dtau
        end
        push!(hist_t, t)
        push!(hist_q, copy(q))
    end

    traj = zeros(n_samples, 5)
    if n_samples > 0
        m = length(hist_t)
        for (r, idx) in enumerate(round.(Int, range(1, m; length=n_samples)))
            traj[r, 1] = hist_t[idx]
            traj[r, 2:5] .= hist_q[idx]
        end
    end

    return q, X, impact_x, Jimp, nev, traj
end

end # module
