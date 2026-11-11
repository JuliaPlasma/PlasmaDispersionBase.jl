"""
Link per-grid-point complex value sets `vals[I]` into continuous sheets over the
m-D parameter grid.
Each sheet is an index array: `sheet[I]` indexes `vals[I]`, `0` where the sheet is absent.

One sheet is traced at a time, seeded at the highest-growth-rate
unclaimed value and grown outward: each frontier cell takes the
nearest unclaimed value within `gate` of a quadratic extrapolation, within
`_JUMP` local steps, or mutually nearest where the sheet has no step yet.
A seed that links nowhere is deferred until every other seed has grown.
`seed = (I, i)` instead traces the single sheet through `vals[I][i]`.
"""
function linksheets(vals::AbstractArray{<:Any,M}; gate, seed = nothing) where {M}
    V = ComplexF64
    grid = CartesianIndices(size(vals))
    pool = map(c -> V.(vals[c]), grid)               # candidate values per cell
    live = map(p -> trues(length(p)), pool)          # still unclaimed
    steps = [CartesianIndex(ntuple(i -> ifelse(i == j, s, 0), M)) for j in 1:M for s in (-1, 1)]
    δ = _grid_step(pool, grid, steps)
    grow!(c, i) = _grow!(pool, live, grid, steps, c, i, gate, δ)
    isnothing(seed) || return [grow!(CartesianIndex(seed[1]), seed[2])]
    sheets = Array{Int,M}[]
    # Beside a crossing a seed's nearest neighbours belong to the other line, so it links
    # nowhere; kept as a one-cell sheet it would split its branch. Deferred so its branch can reach it by extrapolation.
    deferred = Tuple{CartesianIndex{M},Int}[]
    seeds = sort!([(c, i) for c in grid for i in eachindex(pool[c])]; by = ((c, i),) -> -imag(pool[c][i]))
    for (c, i) in seeds
        live[c][i] || continue
        sheet = grow!(c, i)
        if count(!iszero, sheet) == 1
            live[c][i] = true
            push!(deferred, (c, i))
        else
            push!(sheets, sheet)
        end
    end
    for (c, i) in deferred
        live[c][i] && push!(sheets, grow!(c, i))
    end
    return sheets
end

# Claim `pool[sc][si]` and grow its sheet outward over unclaimed values.
function _grow!(pool, live, grid, steps, sc, si, gate, δ)
    live[sc][si] = false
    idx = zeros(Int, size(pool))
    idx[sc] = si
    frontier = [sc + s for s in steps if (sc + s) in grid]
    retry = empty(frontier)
    while true
        grown = false
        while !isempty(frontier)
            c = popfirst!(frontier)
            iszero(idx[c]) || continue
            w, step = _quadratic_extrapolate(pool, idx, c, steps)
            i = _nearest_live(pool[c], live[c], w, isnothing(step) ? gate : _jump_tol(gate, step, δ))
            # No step: take only a mutual nearest match, so a sheet absent here can't steal
            # another's root; retry once the sheet has grown elsewhere
            if isnothing(step) && i != 0 && !_matched_back(pool, idx, c, steps, pool[c][i])
                push!(retry, c)
                continue
            end
            i == 0 && continue
            live[c][i] = false
            idx[c] = i
            grown = true
            for s in steps
                nb = c + s
                (nb in grid && iszero(idx[nb])) && push!(frontier, nb)
            end
        end
        (grown && !isempty(retry)) || break
        frontier, retry = retry, frontier
    end
    return idx
end

# Extrapolate by backward differences ∇, ∇², ∇³. Quadratic's error is about ∇³;
# while ∇³ is small beside ∇ or ∇², the branch is smooth on the grid scale. 
# Where ∇³ rivals both, the branch has a kink, typically a root pair meeting
# or splitting at an instability's onset or end, and no trend holds: stay at the last value.
const _KINK = 0.7
function _quadratic_extrapolate(pool, idx, c, steps)
    acc = zero(eltype(eltype(pool)))
    step = nothing
    n = 0
    for s in steps
        a = _claimed(pool, idx, c + s)
        isnothing(a) && continue
        n += 1
        b = _claimed(pool, idx, c + 2s)
        isnothing(b) && (acc += a; continue)
        step = max(something(step, 0.0), abs(a - b))
        d = _claimed(pool, idx, c + 3s)
        isnothing(d) && (acc += 2a - b; continue)
        e = _claimed(pool, idx, c + 4s)
        ∇, ∇² = a - b, a - 2b + d
        kink = !isnothing(e) && abs(∇² - (b - 2d + e)) > _KINK * max(abs(∇²), abs(∇))
        acc += kink ? a : a + ∇ + ∇²
    end
    return acc / n, step
end

_matched_back(pool, idx, c, steps, v) = all(steps) do s
    a = _claimed(pool, idx, c + s)
    isnothing(a) || minimum(w -> abs(w - v), pool[c + s]) == abs(a - v)
end

_claimed(pool, idx, c) = checkbounds(Bool, idx, c) && !iszero(idx[c]) ? pool[c][idx[c]] : nothing

_jump_tol(gate, step, δ) = min(gate, _JUMP * max(step, δ))

const _JUMP = 3

# Grid-wide per-step motion: the `_STEP_Q` quantile of how far a value at one cell sits from
# the nearest value at the adjacent cell.
const _STEP_Q = 0.9
function _grid_step(pool, grid, steps)
    ds = Float64[]
    for c in grid, s in steps
        nb = c + s
        (nb in grid && !isempty(pool[nb])) || continue
        for v in pool[c]
            push!(ds, minimum(w -> abs(w - v), pool[nb]))
        end
    end
    isempty(ds) && return 0.0
    sort!(ds)
    return ds[clamp(ceil(Int, _STEP_Q * length(ds)), 1, length(ds))]
end

# Index of the nearest unclaimed value to `w` within `gate`; 0 if none.
function _nearest_live(vs, live, w, gate)
    best = 0
    bd = gate
    for i in eachindex(vs)
        live[i] || continue
        d = abs(vs[i] - w)
        d ≤ bd && (bd=d; best=i)
    end
    return best
end
