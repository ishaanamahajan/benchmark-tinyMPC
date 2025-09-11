#!/usr/bin/env julia
# Minimal test of PSD projection for lifted "moment" matrices used in the Julia SDP.
# No Convex/SCS. Just: build M, corrupt it, project with eigen clamp, and verify.
#
# Author: (you)
# Usage:  julia sdp_projection_test.jl

using LinearAlgebra
using Random

# ------------------------------- Problem setup ------------------------------ #

const N  = 31                       # horizon
const nx = 4                        # state: [px, py, vx, vy]
const nu = 2                        # control: [ax, ay]

const Ad = [1 0 1 0;
            0 1 0 1;
            0 0 1 0;
            0 0 0 1]

const Bd = [0.5 0;
            0   0.5;
            1.0 0;
            0   1.0]

x0 = [-10.0, 0.1, 0.0, 0.0]

# ---------------------------- Projection function --------------------------- #

"""
    project_psd(M; eps=1e-8)

Symmetrize M, eigen-decompose, clamp eigenvalues to at least `eps`, and reconstruct.
This is the exact "eigendecomposition -> clamping -> reconstructing" step you want to test.
"""
function project_psd(M::AbstractMatrix{<:Real}; eps::Float64=1e-8)
    S = 0.5 .* (M .+ M')                            # enforce symmetry
    F = eigen(Symmetric(S))                         # symmetric eigendecomposition
    d = max.(F.values, eps)                         # clamp eigenvalues at eps
    return F.vectors * Diagonal(d) * F.vectors'     # reconstruct PSD
end

# ------------------------ Moment-matrix constructors ------------------------- #

"""
    build_moment_matrix(x, u; XX, XU, UX, UU)

Build the 7x7 moment matrix:
[ 1  x'  u';
  x  XX  XU;
  u  UX  UU ]

If XX/XU/UX/UU are not provided, uses the rank-1 "consistent" versions: XX=x*x', etc.
"""
function build_moment_matrix(x::AbstractVector, u::AbstractVector;
                             XX=nothing, XU=nothing, UX=nothing, UU=nothing)
    @assert length(x) == nx && length(u) == nu
    XX = isnothing(XX) ? (x * x') : XX
    XU = isnothing(XU) ? (x * u') : XU
    UX = isnothing(UX) ? (u * x') : UX
    UU = isnothing(UU) ? (u * u') : UU

    M = zeros(7,7)
    M[1,1] = 1.0
    M[1,2:5] = x'
    M[1,6:7] = u'
    M[2:5,1] = x
    M[2:5,2:5] = XX
    M[2:5,6:7] = XU
    M[6:7,1] = u
    M[6:7,2:5] = UX
    M[6:7,6:7] = UU
    return M
end

"""
    build_terminal_matrix(x; XX)

Build the 5x5 terminal matrix:
[ 1  x';
  x  XX ]
"""
function build_terminal_matrix(x::AbstractVector; XX=nothing)
    @assert length(x) == nx
    XX = isnothing(XX) ? (x * x') : XX
    M = zeros(5,5)
    M[1,1] = 1.0
    M[1,2:5] = x'
    M[2:5,1] = x
    M[2:5,2:5] = XX
    return M
end

# ----------------------------- Corruption helper ---------------------------- #

"""
    corrupt_second_moments!(M; gamma=1.5, rng)

Intentionally makes M indefinite by subtracting gamma*I from its lower-right moment block.
For 7x7: this is the 6x6 block C = [XX XU; UX UU].
For 5x5: this is the 4x4 block XX.

This ensures the projection has real work to do.
"""
function corrupt_second_moments!(M::AbstractMatrix; gamma::Float64=1.5)
    @assert size(M,1) == size(M,2)
    n = size(M,1)
    if n == 7
        # C block: rows/cols 2..7
        M[2:7, 2:7] .-= gamma .* Matrix(I, 6, 6)
    elseif n == 5
        # XX block: rows/cols 2..5
        M[2:5, 2:5] .-= gamma .* Matrix(I, 4, 4)
    else
        error("Unsupported matrix size $n")
    end
    # Keep symmetry
    M .= 0.5 .* (M .+ M')
    return M
end

# ------------------------------- Check helpers ------------------------------ #

eigmin_safe(A) = minimum(eigvals(Symmetric(A)))

"""
    schur_min_eig_moment(M)

For a 7x7 moment matrix with top-left α and first-column (below) b=[x;u],
and lower-right C, returns (α, λ_min(C - (1/α) b b')).
Must be ≥ 0 (within tolerance) if M is PSD and α>0.
"""
function schur_min_eig_moment(M::AbstractMatrix)
    @assert size(M) == (7,7)
    α = M[1,1]
    b = M[2:7,1]
    C = M[2:7,2:7]
    return α, eigmin_safe(C .- (b*b') ./ α)
end

"""
    schur_min_eig_terminal(M)

For a 5x5 terminal matrix returns (α, λ_min(XX - (1/α) x x')).
"""
function schur_min_eig_terminal(M::AbstractMatrix)
    @assert size(M) == (5,5)
    α = M[1,1]
    x = M[2:5,1]
    XX = M[2:5,2:5]
    return α, eigmin_safe(XX .- (x*x') ./ α)
end

# ------------------------------- Trajectory --------------------------------- #

function rollout_trajectory()
    x = zeros(nx, N)
    u = zeros(nu, N-1)
    x[:,1] = x0
    for k in 1:N-1
        pos = x[1:2,k]; vel = x[3:4,k]
        u[:,k] = clamp.(-0.15 .* pos .- 0.08 .* vel, -1.0, 1.0)  # same seed logic as your C++
        x[:,k+1] = Ad * x[:,k] + Bd * u[:,k]
    end
    return x, u
end

# --------------------------------- Driver ----------------------------------- #

function main(; gamma=1.5, eps=1e-8, tol=1e-8)
    Random.seed!(123)

    x, u = rollout_trajectory()

    # Storage
    before_min = Float64[]
    after_min  = Float64[]
    schur_min  = Float64[]

    println("== 7×7 moment-matrix projection over horizon ==")
    for k in 1:N-1
        xk = x[:,k]
        uk = u[:,k]

        # Build consistent (rank-1) moment matrix, then corrupt the second moments
        M = build_moment_matrix(xk, uk)
        corrupt_second_moments!(M; gamma=gamma)

        push!(before_min, eigmin_safe(M))

        # Project with your PSD projection
        Mproj = project_psd(M; eps=eps)
        push!(after_min, eigmin_safe(Mproj))

        # Schur complement check (must be ≥ 0 if PSD and α>0)
        α, λmin_schur = schur_min_eig_moment(Mproj)
        push!(schur_min, λmin_schur)

        if k ≤ 3  # show a few lines up front
            println(" step $k: eigmin before = $(round(before_min[end], digits=4))  " *
                    "after = $(round(after_min[end], digits=4));  α=$(round(α, digits=6))  " *
                    "Schur λmin = $(round(λmin_schur, digits=4))")
        end
    end

    nfixed = count(>=( -tol ), after_min)  # all should be ≥ -tol
    nschur = count(>=( -tol ), schur_min)

    println(" summary: $(length(before_min)) moment matrices")
    println("   PSD after projection     : $nfixed / $(N-1) (tol=$(tol))")
    println("   Schur complement ≥ 0     : $nschur / $(N-1) (tol=$(tol))")

    # Terminal 5×5
    println("\n== 5×5 terminal matrix projection ==")
    Mterm = build_terminal_matrix(x[:,N])
    corrupt_second_moments!(Mterm; gamma=gamma)
    λmin_before = eigmin_safe(Mterm)
    Mterm_proj  = project_psd(Mterm; eps=eps)
    λmin_after  = eigmin_safe(Mterm_proj)
    αT, λmin_schurT = schur_min_eig_terminal(Mterm_proj)

    println(" terminal: eigmin before = $(round(λmin_before, digits=4))  " *
            "after = $(round(λmin_after, digits=4));  α=$(round(αT, digits=6))  " *
            "Schur λmin = $(round(λmin_schurT, digits=4))")

    ok7  = all(>(-tol), after_min) && all(>(-tol), schur_min)
    ok5  = (λmin_after > -tol) && (λmin_schurT > -tol)
    println("\nRESULT: ", (ok7 && ok5) ? "✅ Projection behaves correctly on both 7×7 and 5×5." :
                                    "⚠️  Projection needs attention (see negatives above).")
end

# Run
main()
