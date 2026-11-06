# Julia port of the transmission functions of coverage_models.py (achromatic source).
module CoverageModels

export COVERAGE_MODE_NAMES, coverage_mode_name, region_transmission

const COVERAGE_MODE_NAMES = ("full", "ppc", "powerlaw", "powerlaw_ppc", "two_zone", "cfov",
                             "powerlaw_radial", "powerlaw_radial_ppc", "powerlaw_ppc_sysa",
                             "powerlaw_sysa_global")

function coverage_mode_name(value::AbstractString)
    v = lowercase(strip(value))
    v in COVERAGE_MODE_NAMES && return String(v)
    i = tryparse(Float64, v)
    return i === nothing ? "ppc" : coverage_mode_name(i)
end
function coverage_mode_name(value::Real)
    isfinite(value) || return "ppc"
    i = round(Int, value)
    return 0 <= i < length(COVERAGE_MODE_NAMES) ? COVERAGE_MODE_NAMES[i + 1] : "ppc"
end

function _tau(tau)
    all(isfinite, tau) || throw(ArgumentError("Optical depth contains non-finite values"))
    any(<(0), tau) && throw(ArgumentError("Optical depth must be non-negative"))
    return float.(tau)
end

function _fraction(value, name)
    value = Float64(value)
    (isfinite(value) && 0 <= value <= 1) || throw(ArgumentError("$name must lie in [0, 1], got $value"))
    return value
end

function _exponent(log10_a)
    isfinite(log10_a) || throw(ArgumentError("log10_a must be finite"))
    return 10.0^Float64(log10_a)
end

full_coverage(tau) = exp.(-_tau(tau))

# Barlow & Sargent 1997, AJ 113, 136.
partial_covering(tau, cf) = (c = _fraction(cf, "cf"); (1 - c) .+ c .* exp.(-_tau(tau)))

function overlap_bounds(cf1, cf2)
    cf1, cf2 = _fraction(cf1, "cf1"), _fraction(cf2, "cf2")
    return max(0.0, cf1 + cf2 - 1), min(cf1, cf2)
end

function overlap_from_normalized(cf1, cf2, cfov_norm)
    q = _fraction(cfov_norm, "cfov_norm")
    lower, upper = overlap_bounds(cf1, cf2)
    return lower + q * (upper - lower)
end

function overlap_partial_covering(tau1, tau2, cf1, cf2, cfov_norm)
    tau1, tau2 = _tau(tau1), _tau(tau2)
    cf1, cf2 = _fraction(cf1, "cf1"), _fraction(cf2, "cf2")
    cfov = overlap_from_normalized(cf1, cf2, cfov_norm)
    flux = (1 - cf1 - cf2 + cfov) .+ (cf1 - cfov) .* exp.(-tau1) .+ (cf2 - cfov) .* exp.(-tau2) .+
           cfov .* exp.(-(tau1 .+ tau2))
    return clamp.(flux, 0, 1)
end

# log Gamma(x), x > 0: upward recurrence to x >= 15, then the Stirling series
# (Abramowitz & Stegun 1964, 6.1.41).
function _lgamma(x::Float64)
    shift = 0.0
    while x < 15
        shift += log(x)
        x += 1
    end
    z = 1 / (x * x)
    series = (1/12 + z * (-1/360 + z * (1/1260 + z * (-1/1680 + z * (1/1188 + z * (-691/360360 + z / 156)))))) / x
    return (x - 0.5) * log(x) - x + 0.5 * log(2pi) + series - shift
end

# J_m = int_0^1 u^m exp(-tau_max u^a) du = gamma(s, tau_max) / (a tau_max^s), s = (m+1)/a.
# Lower incomplete gamma: series for tau < s+1, continued fraction otherwise
# (Press et al. 2007, Numerical Recipes 3rd ed., sec. 6.2).
function _powerlaw_moment_scalar(t::Float64, a::Float64, m::Float64)
    zero_value = 1 / (m + 1)
    t > 1e-12 || return zero_value
    s = (m + 1) / a
    if t < s + 1
        term = sum = 1 / s
        k = 0
        while abs(term) > eps() * abs(sum) && k < 100_000
            k += 1
            term *= t / (s + k)
            sum += term
        end
        value = exp(-t) * sum / a
    else
        tiny = 1e-300
        b = t + 1 - s
        c = 1 / tiny
        d = 1 / b
        h = d
        for i in 1:100_000
            an = -i * (i - s)
            b += 2
            d = an * d + b
            abs(d) < tiny && (d = tiny)
            c = b + an / c
            abs(c) < tiny && (c = tiny)
            d = 1 / d
            delta = d * c
            h *= delta
            abs(delta - 1) < eps() && break
        end
        value = (exp(_lgamma(s) - s * log(t)) - exp(-t) * h) / a
    end
    return clamp(value, 0.0, zero_value)
end

function _powerlaw_moment(tau_max, a, moment=0)
    a = Float64(a)
    (isfinite(a) && a > 0) || throw(ArgumentError("power-law exponent a must be positive"))
    moment >= 0 || throw(ArgumentError("moment must be non-negative"))
    return [_powerlaw_moment_scalar(Float64(t), a, Float64(moment)) for t in _tau(tau_max)]
end

function _tau_max(tau, a, tau_kind, mean_factor)
    tau_kind == "mean" && return mean_factor .* tau
    tau_kind == "max" && return tau
    throw(ArgumentError("tau_kind must be 'mean' or 'max'"))
end

# N(x) = N_max x^a over the source: de Kool, Korista & Arav 2002, ApJ 580, 54; Arav et al. 2005, ApJ 620, 665.
function powerlaw_x_inhomogeneous(tau, log10_a; tau_kind="mean")
    tau, a = _tau(tau), _exponent(log10_a)
    return clamp.(_powerlaw_moment(_tau_max(tau, a, tau_kind, a + 1), a, 0), 0, 1)
end

function powerlaw_radial_inhomogeneous(tau, log10_a; tau_kind="mean")
    tau, a = _tau(tau), _exponent(log10_a)
    tau_max = _tau_max(tau, a, tau_kind, 0.5 * (a + 1) * (a + 2))
    return clamp.(2 .* (_powerlaw_moment(tau_max, a, 0) .- _powerlaw_moment(tau_max, a, 1)), 0, 1)
end


function powerlaw_with_uncovered_fraction(tau, cf, log10_a; tau_kind="mean", geometry="radial")
    cf = _fraction(cf, "cf")
    covered = geometry == "radial" ? powerlaw_radial_inhomogeneous(tau, log10_a; tau_kind=tau_kind) :
              geometry == "x" ? powerlaw_x_inhomogeneous(tau, log10_a; tau_kind=tau_kind) :
              throw(ArgumentError("geometry must be 'radial' or 'x'"))
    return (1 - cf) .+ cf .* covered
end

function powerlaw_system_product_with_uncovered(tau_by_system::AbstractDict, cf, log10_a_by_system::AbstractDict)
    cf = _fraction(cf, "cf")
    isempty(tau_by_system) && throw(ArgumentError("tau_by_system is empty"))
    ids = sort(collect(keys(tau_by_system)))
    covered = ones(length(tau_by_system[ids[1]]))
    for k in ids
        haskey(log10_a_by_system, k) ||
            throw(ArgumentError("Missing system power-law exponent for sys$k; expected covasys_$k"))
        covered .*= powerlaw_x_inhomogeneous(tau_by_system[k], log10_a_by_system[k])
    end
    return (1 - cf) .+ cf .* covered
end

function two_zone(tau_high, cf_high, tau_ratio)
    tau_high = _tau(tau_high)
    cf_high, tau_ratio = _fraction(cf_high, "cf_high"), _fraction(tau_ratio, "tau_ratio")
    return (1 - cf_high) .* exp.(-tau_ratio .* tau_high) .+ cf_high .* exp.(-tau_high)
end

# params: "cf", "log10_a", "tau_ratio", "cf1", "cf2", "cfov_norm", "log10_a_by_system" (as in Python).
function region_transmission(mode, tau_by_system::AbstractDict, params::AbstractDict)
    mode = coverage_mode_name(mode)
    isempty(tau_by_system) && throw(ArgumentError("tau_by_system is empty"))
    ids = sort(collect(keys(tau_by_system)))
    tau = sum(_tau(tau_by_system[k]) for k in ids)
    mode == "full" && return full_coverage(tau)
    mode == "ppc" && return partial_covering(tau, params["cf"])
    mode == "powerlaw" && return powerlaw_x_inhomogeneous(tau, params["log10_a"])
    mode == "powerlaw_ppc" && return powerlaw_with_uncovered_fraction(tau, params["cf"], params["log10_a"]; geometry="x")
    mode == "powerlaw_ppc_sysa" && return powerlaw_system_product_with_uncovered(tau_by_system, params["cf"], params["log10_a_by_system"])
    mode == "powerlaw_sysa_global" && return powerlaw_system_product_with_uncovered(tau_by_system, 1.0, params["log10_a_by_system"])
    mode == "powerlaw_radial" && return powerlaw_radial_inhomogeneous(tau, params["log10_a"])
    mode == "powerlaw_radial_ppc" && return powerlaw_with_uncovered_fraction(tau, params["cf"], params["log10_a"]; geometry="radial")
    mode == "two_zone" && return two_zone(tau, params["cf"], params["tau_ratio"])
    if mode == "cfov"
        length(ids) == 1 && return partial_covering(tau, ids[1] == 0 ? params["cf1"] : params["cf2"])
        length(ids) == 2 || throw(ArgumentError("The overlap-Cf model currently supports exactly two active systems per covering region; found $(length(ids))"))
        return overlap_partial_covering(tau_by_system[ids[1]], tau_by_system[ids[2]], params["cf1"], params["cf2"], params["cfov_norm"])
    end
    throw(ArgumentError("Unknown coverage model: $mode"))
end

end # module CoverageModels
