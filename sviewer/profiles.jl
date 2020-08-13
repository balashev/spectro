using AffineInvariantMCMC
using DataStructures
using Interpolations
using LsqFit
using Roots
using SpecialFunctions

function argclose2(x, value)
    return argmin(abs.(x .- value))
end

function binsearch(x, item; type="close")

    if item <= x[1]
        return 1
    elseif item >= x[end]
        return size(x)[1]
    else
        first = 1
        last = length(x)

        while first < last - 1
            middle = fld(first+last, 2)
            if item > x[middle]
                first = middle
            else
                last = middle
            end
        end

        if type == "min"
            return first
        elseif type == "max"
            return last
        elseif type == "close"
            if item - x[first] < x[last] - item
                return first
            else
                return last
            end
        end
    end
end

function Dawson(x)
    y = x .* x
    return x .* (1 .+ y .* (0.1107817784 .+ y .* (0.0437734184 .+ y .* (0.0049750952 .+ y .* 0.0015481656)))) / (1 .+ y .* (0.7783701713 .+ y .* (0.2924513912 .+ y .* (0.0756152146 .+ y .* (0.0084730365 .+ 2 .*  0.0015481656 .* y)))))
end

function Voigt(a, x)
    return exp.(-1 .* x .* x) .* (1 .+ a^2 .* (1 .- 2 .* x .* x)) - 2 / sqrt(π) .* (1 .- 2 .* x .* Dawson.(x))
end

function voigt_range_old(a, level)
    f(x) = real(SpecialFunctions.erfcx(a - im * x)) - level
    return find_zero(f, (min(sqrt(a / level / sqrt(π)), sqrt(max(0, -log(level)))) / 2, 3*max(sqrt(a / level / sqrt(π)), sqrt(max(0, -log(level))))), tol=0.001)
end

function voigt_range(a, level)
    if level < 1
        α = 1.3
        if exp(sqrt(a / sqrt(π) / level)) ^ α == Inf
            return sqrt(a / sqrt(π) / level)
        else
            return log((exp(sqrt(-log(level))) ^ α + exp(sqrt(a / sqrt(π) / level)) ^ α - 1) ^ (1 / α))
        end
    elseif level == 1
        return 0
    else
        return 0
    end
end

function voigt_deriv(x, a, tau_0)
    w = SpecialFunctions.erfcx.(a .- im .* x)
    return exp.( - tau_0 .* real(w)) .* tau_0 .* 2 .* (imag(w) .* a .- real(w) .* x)
end

function voigt_step(a, tau_0; level=0.002, step=0.03)
    x_0 = voigt_max_deriv(a, tau_0)
    x = [-abs(x_0)]
    w = real(SpecialFunctions.erfcx.(a .- im .* x[1])) * tau_0
    der = Vector{Float64}()
    while x[end] < 0
        append!(der, voigt_deriv(x[end], a, tau_0))
        append!(x, x[end] + step / der[end])
    end
    deleteat!(x, size(x))
    t = real(SpecialFunctions.erfcx.(a .- im .* x[1])) * tau_0
    while t > level
        pushfirst!(x, x[1] - step / der[1])
        w = SpecialFunctions.erfcx.(a .- im .* x[1])
        t = tau_0 .* real(w)
        pushfirst!(der, exp.( - t) .* tau_0 .* 2 .* (imag(w) .* a .- real(w) .* x[1]))
    end
    append!(x, -x[end:-1:1])
    append!(der, der[end:-1:1])
    return x, der
end

function df(x, a, tau_0)
    v = SpecialFunctions.erfcx(a - im * x)
    return tau_0 * (imag(v) * a - real(v) * x)^2 + real(v) * (a^2 - x^2 + 0.5) + 2 * imag(v) * a * x - a / sqrt(π)
end

function voigt_max_deriv(a, tau_0)
    f = (x -> df(x, a, tau_0))
    level = tau_0 > 0.3 ? 0.1 / tau_0 : 1 / 3
    try
        r = find_zero(f, (0, abs(voigt_range(a, level))))
        return r
    catch
        r = find_zero(f, abs(voigt_range(a, level) / 2))
        return r
    end
end

function voigt_grid(l, a, tau_0; step=0.03)
    x, r = voigt_step(a, tau_0)
    k_min, k_max = binsearch(l, x[1], type="min"), binsearch(l, x[end], type="max")-1
    g = Vector{Float64}()
    for k in k_min:k_max
        i_min, i_max = binsearch(x, l[k]), binsearch(x, l[k+1])
        #append!(g, maximum(r[i_min:i_max]))
        append!(g, Int(floor((l[k+1] - l[k]) / (step / maximum(r[i_min:i_max]))))+1)
    end
    return k_min:k_max, g
end

##############################################################################
##############################################################################
##############################################################################

mutable struct par
    name::String
    val::Float64
    min::Float64
    max::Float64
    step::Float64
    vary::Bool
    addinfo::String
end

function make_pars(p_pars)
    pars = OrderedDict{String, par}()
    for p in p_pars
        pars[p.__str__()] = par(p.__str__(), p.val, p.min, p.max, p.step, p.fit * p.vary, p.addinfo)
        if occursin("cf", p.__str__())
            pars[p.__str__()].min, pars[p.__str__()].max = 0, 1
        end
    end
    return pars
end

function update_pars(pars, spec)
    for (k, v) in pars
        if occursin("res", pars[k].name)
            #println(pars[k].name, " ", pars[k].val, " ", parse(Int, pars[k].addinfo[5:end]))
            spec[parse(Int, pars[k].addinfo[5:end]) + 1].resolution = pars[k].val
        end
        if occursin("disps", pars[k].name)
            spec[parse(Int, split(pars[k].name, "_")[2]) + 1].disps = pars[k].val
        end
        if occursin("dispz", pars[k].name)
            spec[parse(Int, split(pars[k].name, "_")[2]) + 1].dispz = pars[k].val
        end
    end
end

mutable struct prior
    name::String
    val::Float64
    plus::Float64
    minus::Float64
end

function make_priors(p_priors)
    priors = OrderedDict{String, prior}()
    for (k, p) in p_priors
        priors[k] = prior(k, p.val, p.plus, p.minus)
    end
    return priors
end

function use_prior(prior, val)
    a = val - prior.val
    if a > 0
        return 0.5 * (a / prior.plus) ^ 2
    else
        return 0.5 * (a / prior.minus) ^ 2
    end
end

##############################################################################

mutable struct line
    name::String
    sys::Int64
    lam::Float64
    f::Float64
    g::Float64
    b::Float64
    logN::Float64
    z::Float64
    l::Float64
    tau0::Float64
    a::Float64
    ld::Float64
    dx::Float64
    cf::Int64
end

function update_lines(lines, pars; ind=0)
    mask = Vector{Bool}(undef, 0)
    for line in lines
        if pars["b_" * string(line.sys) * "_" * line.name].addinfo != ""
            line.b = pars["b_" * string(line.sys) * "_" * pars["b_" * string(line.sys) * "_" * line.name].addinfo].val
        else
            line.b = pars["b_" * string(line.sys) * "_" * line.name].val
        end
        line.logN = pars["N_" * string(line.sys) * "_" * line.name].val
        line.z = pars["z_" * string(line.sys)].val
        line.l = line.lam * (1 + line.z)
        line.tau0 = sqrt(π) * 0.008447972556327578 * (line.lam * 1e-8) * line.f * 10 ^ line.logN / (line.b * 1e5)
        line.a = line.g / 4 / π / line.b / 1e5 * line.lam * 1e-8
        line.ld = line.lam * line.b / 299794.26 * (1 + line.z)
        append!(mask, (ind == 0) || (line.sys == ind-1))
    end
    return mask
end

function prepare_lines(lines, pars)
    fit_lines = Vector{line}(undef, size(lines)[1])
    for (i, l) in enumerate(lines)
        fit_lines[i] = line(l.name, l.sys, l.l(), l.f(), l.g(), l.b, l.logN, l.z, l.l()*(1+l.z), 0, 0, 0, 0, l.cf)
    end
    return fit_lines
end

##############################################################################

mutable struct spectrum
    x::Vector{Float64}
    y::Vector{Float64}
    unc::Vector{Float64}
    mask::BitArray
    resolution::Float64
    lines::Vector{line}
    disps::Float64
    dispz::Float64
end


function prepare(s, pars)
    spec = Vector(undef, size(s)[1])
    for (i, si) in enumerate(s)
        spec[i] = spectrum(si.spec.norm.x, si.spec.norm.y, si.spec.norm.err, si.mask.norm.x .== 1, si.resolution, prepare_lines(si.fit_lines, pars), 0, 0)
    end
    update_pars(pars, spec)
    return spec
end

function calc_spectrum(spec, pars; ind=0, regular=-1, regions="fit", out="all")

    timeit = 0
    if timeit == 1
        start = time()
        println("start ", spec.resolution)
    end

    line_mask = update_lines(spec.lines, pars, ind=ind)

    x_instr = 1.0 / spec.resolution / 2.355
    x_grid = -1 .* ones(Int8, size(spec.x)[1])
    x_grid[spec.mask] = zeros(sum(spec.mask))
    for line in spec.lines[line_mask]
        i_min, i_max = binsearch(spec.x, line.l * (1 - 4 * x_instr), type="min"), binsearch(spec.x, line.l * (1 + 4 * x_instr), type="max")
        if i_max - i_min > 1 && i_min > 1
            for i in i_min:i_max
                x_grid[i] = max(x_grid[i], round(Int, (spec.x[i] - spec.x[i-1]) / line.l / x_instr * 4))
            end
        end
        line.dx = voigt_range(line.a, 0.001 / line.tau0)
        #println(line.logN, "  ", line.b)
        x, r = voigt_step(line.a, line.tau0)
        x = line.l .+ x * line.ld
        i_min, i_max = binsearch(spec.x, x[1], type="min"), binsearch(spec.x, x[end], type="max")-1
        if i_max - i_min > 1 && i_min > 1
            for i in i_min:i_max
                k_min, k_max = binsearch(x, spec.x[i]), binsearch(x, spec.x[i+1])
                x_grid[i] = max(x_grid[i], Int(floor((spec.x[i+1] - spec.x[i]) / (0.2  / maximum(r[k_min:k_max]) * line.ld)))+1)
            end
        end
    end

    if timeit == 1
        println("update ", start - time())
    end

    if regular == 0
        x = spec.x[x_grid .> -1]
        x_mask = ~isinf(x)
    else
        x = [0.0]
        x_mask = Vector{Int64}(undef, 0)
        k = 1
        if regular == -1
            for i in 1:size(x_grid)[1]-1
                if spec.mask[i] > 0
                    append!(x_mask, k)
                end
                if x_grid[i] == 0
                    splice!(x, k, [spec.x[i], spec.x[i]])
                    k += 1
                elseif x_grid[i] > 0
                    step = (spec.x[i+1] - spec.x[i]) / (x_grid[i] + 1)
                    splice!(x, k, range(spec.x[i], length=x_grid[i]+2, step=step))
                    k += x_grid[i]+1
                end

            end
        elseif regular > 0
            for i in 1:size(x_grid)[1]-1
                if spec.mask[i] > 0
                    append!(x_mask, k)
                end
                if x_grid[i] > -1 && x_grid[i+1] > -1
                    step = (spec.x[i+1] - spec.x[i]) / (regular + 1)
                    splice!(x, k, range(spec.x[i], stop=spec.x[i+1], length=regular+2))
                    k += kind + 1
                end
            end
        end
    end

    if timeit == 1
        println("make grid ", start - time())
    end

    if ~any([occursin("cf", p.first) for p in pars])
        y = ones(size(x))
        for line in spec.lines[line_mask]
            i_min, i_max = binsearch(x, line.l - line.dx * line.ld, type="min"), binsearch(x, line.l + line.dx * line.ld, type="max")
            @. @views y[i_min:i_max] = y[i_min:i_max] .* exp.(-1 .* line.tau0 .* real.(SpecialFunctions.erfcx.(line.a .- im .* (x[i_min:i_max] .- line.l) ./ line.ld)))
        end
    else
        y = zeros(size(x))
        cfs, inds = [], []
        for (i, line) in enumerate(spec.lines[line_mask])
            append!(cfs, line.cf)
            append!(inds, i)
        end
        #println(cfs, inds)
        for l in unique(cfs)
            if l > -1
                cf = pars["cf_" * string(l)].val
            else
                cf = 1
            end
            #println(l, " ", cf)
            profile = zeros(size(x))
            for (i, c) in zip(inds, cfs)
                if c == l
                    line = spec.lines[line_mask][i]
                    i_min, i_max = binsearch(x, line.l - line.dx * line.ld, type="min"), binsearch(x, line.l + line.dx * line.ld, type="max")
                    @. @views profile[i_min:i_max] -= line.tau0 .* real.(SpecialFunctions.erfcx.(line.a .- im .* (x[i_min:i_max] .- line.l) ./ line.ld))
                end
            end
            y += log.(exp.(profile) .* cf .+ (1 .- cf))
        end
        y = exp.(y)
    end

    if timeit == 1
        println("calc lines ", start - time())
        #println(size(x))
    end

    #if any([occursin("disp", p.first) for p in pars])
    #    n = Int(sum([occursin("disp", p.first) for p in pars]) / 2)
    #    for i in 0:n-1
    #        println(i)
    #        for p in pars
    #            #println(p.first, " ", occursin("disp", p.first), " ", parse(Int, split(p.first, "_")[2]) == i, " ", occursin("disp", p.first) & (parse(Int, split(p.first, "_")[2]) == i))
    #            if occursin("disp", p.first) & (parse(Int, split(p.first, "_")[2]) == i)
    #                println(p.first, " ", p.second.addinfo)
    #            end
    #        end
    #    end
    #end
    #println(spec.dispz, " ", spec.disps)
    if (spec.dispz != 0) & (spec.disps != 0)
        println(spec.dispz, " ", spec.disps)
        inter = LinearInterpolation(x, y, extrapolation_bc=Flat())
        y = inter(x .+ (x .- spec.dispz) .* spec.disps)
    end


    if spec.resolution != 0
        y = 1 .- y
        y_c = Vector{Float64}(undef, size(y)[1])
        for (i, xi) in enumerate(x)
            sigma_r = xi / spec.resolution / 1.66511
            k_min, k_max = binsearch(x, xi - 3 * sigma_r), binsearch(x, xi + 3 * sigma_r)
            #println(k_min, "  ", k_max)
            instr = exp.( -1 .* ((view(x, k_min:k_max) .- xi) ./ sigma_r ) .^ 2)
            s = 0
            @inbounds for k = k_min+1:k_max
                s = s + (y[k] * instr[k-k_min+1] + y[k-1] * instr[k-k_min]) * (x[k] - x[k-1])
            end
            y_c[i] = s / 2 / sqrt(π) / sigma_r  + y[k_min] * (1 - SpecialFunctions.erf((xi - x[k_min]) / sigma_r)) / 2 + y[k_max] * (1 - SpecialFunctions.erf((x[k_max] - xi) / sigma_r)) / 2
            #sleep(5)
        end

        if timeit == 1
            println("convolve ", start - time())
        end

        if out == "all"
            return x, 1 .- y_c
        elseif out == "init"
            #println("all done ", sum(y_c[x_mask]))
            return 1 .- y_c[x_mask]
        end
    else
        if out == "all"
            return x, y
        elseif out == "init"
            return y[x_mask]
        end
    end

end


function fitLM(spec, p_pars)

    function cost(p)
        i = 1
        #println(p)
        for (k, v) in pars
            if v.vary == 1
                pars[k].val = p[i]
                i += 1
            end
        end

        update_pars(pars, spec)

        res = Vector{Float64}()
        for s in spec
            if sum(s.mask) > 0
                append!(res, (calc_spectrum(s, pars, out="init") .- s.y[s.mask]) ./ s.unc[s.mask])
            end
        end
        #println("chi ", sum(res .^ 2))
        return res
    end

    pars = make_pars(p_pars)

    println("fitLM ", pars)
    params = [p.val for (k, p) in pars if p.vary == true]
    lower = [p.min for (k, p) in pars if p.vary == true]
    upper = [p.max for (k, p) in pars if p.vary == true]

    println(params, " ", lower, " ", upper)
    fit = LsqFit.lmfit(cost, params, Float64[]; maxIter=300, lower=lower, upper=upper, show_trace=true)
    sigma = stderror(fit)
    covar = estimate_covar(fit)

    println(dof(fit))
    println(fit.param)
    println(sigma)
    println(covar)

    return dof(fit), fit.param, sigma

end

