module Kernel

"""
    apply_kernel(x)

Elementwise test kernel y = sin(x) * x^2. Stand-in for a Julia solver:
the point is proving the Tesseract boundary, not the physics.
"""
function apply_kernel(x::AbstractVector{<:Real})
    return sin.(x) .* x .^ 2
end

"""
    vjp_kernel(x, ct)

Hand-written analytic adjoint of `apply_kernel`: returns ctᵀ · (∂y/∂x).
The Jacobian is diagonal with entries cos(x)·x² + 2x·sin(x).
"""
function vjp_kernel(x::AbstractVector{<:Real}, ct::AbstractVector{<:Real})
    return ct .* (cos.(x) .* x .^ 2 .+ 2 .* x .* sin.(x))
end

end # module
