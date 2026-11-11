"""
    muller(f, x0; atol = 1e-10, maxiter = 100) -> (root, nevals, retcode)
    muller(f, x0, x1, x2; atol, maxiter)

Find a complex root of f using Muller's quadratic interpolation method.
Accept either one initial guess (automatically seeded) or three starting points.

Converged when a step falls below `atol·max(|x|, 1)` where `|f|` has fallen by `√atol`. 
Returns (root, nevals, retcode), with root = NaN on failure (ReturnCode.Failure) or iteration limit (ReturnCode.MaxIters).
"""
function muller(f, x0; kw...)
    h = _seed_offset(x0)
    return muller(f, x0 - h, x0 + h, x0 + h * im; kw...)
end

# Absolute-floored relative step capped at 0.1|ω0|; ω0 = 0 has no scale ⇒ a small absolute step.
_seed_offset(ω0) = iszero(ω0) ? 1.0e-3 : min(1.0e-3 * max(abs(ω0), 1.0), 0.1 * abs(ω0))

function muller(f, x0, x1, x2; atol = 1.0e-10, maxiter = 100)
    x0, x1, x2 = promote(complex(x0), complex(x1), complex(x2))
    nan = typeof(x2)(NaN, NaN)
    f0, f1, f2 = f(x0), f(x1), f(x2)
    a0, a1, a2 = abs(f0), abs(f1), abs(f2)
    nevals = 3
    fell = false
    rtol = sqrt(atol)
    for _ in 1:maxiter
        h1 = x1 - x0
        h2 = x2 - x1
        (h1 == 0 || h2 == 0 || h1 + h2 == 0) && return nan, nevals, ReturnCode.Failure
        delta1 = (f1 - f0) / h1
        delta2 = (f2 - f1) / h2
        a = (delta2 - delta1) / (h2 + h1)
        b = a * h2 + delta2
        c = f2

        disc = sqrt(b^2 - 4a * c)
        # The larger of |b ± disc|: |b + disc|² − |b − disc|² = 4 Re(b̄·disc)
        denom = real(conj(b) * disc) > 0 ? b + disc : b - disc
        denom == 0 && return nan, nevals, ReturnCode.Failure

        dx = -2c / denom
        x3 = x2 + dx
        f3 = f(x3)
        a3 = abs(f3)
        nevals += 1
        # Backtrack steps that produce nonfinite or excessively large determinant values.
        flimit = 1.0e6 * max(a0, a1, a2)
        for _ in 1:12
            (isfinite(f3) && a3 <= flimit) && break
            dx /= 2
            x3 = x2 + dx
            f3 = f(x3)
            a3 = abs(f3)
            nevals += 1
        end
        isfinite(f3) || return nan, nevals, ReturnCode.Failure
        # Accept convergence only after a substantial residual decrease and a small step.
        # - Exponential decay can mimic approach to a root and absolute residuals depend on normalization.
        # - Widely differing residuals can also collapse the model's step without improving f.
        #  Retain evidence of decrease locally: evaluation error near a root can prevent further residual improvement.
        mid = max(min(a0, a1), min(max(a0, a1), a2))
        step = abs(dx) / max(abs(x3), 1)
        fell = (a3 <= a2 / 10 && a3 <= rtol * mid) || (fell && step <= rtol)
        if iszero(f3) || (fell && step <= atol)
            return x3, nevals, ReturnCode.Success
        end

        x0, x1, x2 = x1, x2, x3
        f0, f1, f2 = f1, f2, f3
        a0, a1, a2 = a1, a2, a3
    end
    return nan, nevals, ReturnCode.MaxIters
end
