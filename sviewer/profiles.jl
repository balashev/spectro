using AffineInvariantMCMC
using DataFitting
using LsqFit
using OrderedCollections
using SpecialFunctions


function argclose2(x, value)
    return argmin(abs.(x .- value))
end

function binsearch(x, item; type="close")
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

function Dawson(x)
    y = x .* x
    return x .* (1 .+ y .* (0.1107817784 .+ y .* (0.0437734184 .+ y .* (0.0049750952 .+ y .* 0.0015481656)))) / (1 .+ y .* (0.7783701713 .+ y .* (0.2924513912 .+ y .* (0.0756152146 .+ y .* (0.0084730365 .+ 2 .*  0.0015481656 .* y)))))
end

function Voigt(a, x)
    return exp.(-1 .* x .* x) .* (1 .+ a^2 .* (1 .- 2 .* x .* x)) - 2 / sqrt(π) .* (1 .- 2 .* x .* Dawson.(x))
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
    pars = Vector{par}(undef, length(collect(keys(p_pars))))
    i = 1
    for p in p_pars
        pars[i] = par(p.__str__(), p.val, p.min, p.max, p.step, p.fit * p.vary, p.addinfo)
        i += 1
    end
    return pars
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
    b_ind::Int64
    N_ind::Int64
    z_ind::Int64
end

function update_lines(lines, pars; ind=0)
    mask = Vector{Bool}(undef, 0)
    for line in lines
        line.b = pars[line.b_ind].val
        line.logN = pars[line.N_ind].val
        line.z = pars[line.z_ind].val
        line.l = line.lam * (1 + line.z)
        line.tau0 = sqrt(π) * 0.008447972556327578 * (line.lam * 1e-8) * line.f * 10 ^ line.logN / (line.b * 1e5)
        line.a = line.g / 4 / π / line.b / 1e5 * line.lam * 1e-8
        line.ld = line.lam * line.b / 299794.26 * (1 + line.z)
        append!(mask, (ind == 0) || (line.sys == ind-1))
    end
    return mask
end

function get_b_index(name, pars)::Int64
    for (i, p) in enumerate(pars)
        if name == p.name
            if p.addinfo == ""
                return i
            else
                return get_b_index(rsplit(name, "_", limit=2)[1] * "_" * p.addinfo, pars)
            end
        end
    end
end

function get_N_index(name, pars)::Int64
    for (i, p) in enumerate(pars)
        if name == p.name
            if p.addinfo == ""
                return i
            else
                return i
            end
        end
    end
end

function get_z_index(name, pars)::Int64
    for (i, p) in enumerate(pars)
        if name == p.name
            return i
        end
    end
end

function prepare_lines(lines, pars)
    fit_lines = Vector{line}(undef, size(lines)[1])
    for (i, l) in enumerate(lines)
        fit_lines[i] = line(l.name, l.sys, l.l(), l.f(), l.g(), l.b, l.logN, l.z, l.l()*(1+l.z), 0, 0, 0, 0, 0, 0, 0)
        fit_lines[i].b_ind = get_b_index("b_" * string(fit_lines[i].sys) * "_" * fit_lines[i].name, pars)
        fit_lines[i].N_ind = get_N_index("N_" * string(fit_lines[i].sys) * "_" * fit_lines[i].name, pars)
        fit_lines[i].z_ind = get_z_index("z_" * string(fit_lines[i].sys), pars)
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
end


function prepare(s, pars)
    spec = Vector(undef, size(s)[1])
    for (i, si) in enumerate(s)
        spec[i] = spectrum(si.spec.norm.x, si.spec.norm.y, si.spec.norm.err, si.mask.norm.x .== 1, si.resolution, prepare_lines(si.fit_lines, pars))
    end
    return spec
end

if 1 == 0
    const COUNTERS = Dict{String, Int}()
    COUNTERS["num"] = 100

    macro counted(f)
        name = f.args[1].args[1]
        name_str = String(name)
        body = f.args[2]
        counter_code = quote
            if !haskey(COUNTERS, $name_str)
                COUNTERS[$name_str] = 0
            end
            COUNTERS[$name_str] += 1
            if COUNTERS[$name_str] % COUNTERS["num"] == 0
                println("iter: ", COUNTERS[$name_str], " ", COUNTERS["num"])
            end
        end
        insert!(body.args, 1, counter_code)
        return f
    end
end

function calc_spectrum(spec, pars; ind=0, regular=-1, regions="fit", out="all")

    start = time()
    #println("start")

    line_mask = update_lines(spec.lines, pars, ind=ind)

    x_instr = 1.0 / spec.resolution / 2.355
    x_grid = -1 .* ones(Int8, size(spec.x)[1])
    x_grid[spec.mask] = zeros(sum(spec.mask))
    for line in spec.lines[line_mask]
        i_min, i_max = binsearch(spec.x, line.l * (1 - 4 * x_instr), type="min"), binsearch(spec.x, line.l * (1 + 4 * x_instr), type="max")
        if i_max - i_min > 1 && i_min > 0
            for i in i_min:i_max
                ###### PROBLEMS HERE ############
                x_grid[i] = max(x_grid[i], round(Int, (spec.x[i] - spec.x[i-1]) / line.l / x_instr * 5))
                ###### PROBLEMS HERE ############
            end
        end
        line.dx = sqrt(max(-log10(0.001 / line.tau0), line.tau0 / 0.001 * line.a / sqrt(π))) * 1.5
        i_min, i_max = binsearch(spec.x, line.l - line.dx * line.ld, type="min"), binsearch(spec.x, line.l + line.dx * line.ld, type="max")
        if i_max - i_min > 1 && i_min > 0
            for i in i_min:i_max
                x_grid[i] = max(x_grid[i], round(Int, (spec.x[i] - spec.x[i-1]) / line.ld * 3) + 1)
            end
        end
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
        #splice!(x, k, spec.x[end])
    end

    y = ones(size(x))

    for line in spec.lines[line_mask]
        i_min, i_max = binsearch(x, line.l - line.dx * line.ld, type="min"), binsearch(x, line.l + line.dx * line.ld, type="max")
        #println(i_min, " ", i_max)
        @. @views y[i_min:i_max] = y[i_min:i_max] .* exp.(-1 .* line.tau0 .* real.(SpecialFunctions.erfcx.(line.a .- im .* (x[i_min:i_max] .- line.l) ./ line.ld)))
    end

    if spec.resolution != 0
        y = 1 .- y
        y_c = Vector{Float64}(undef, size(y)[1])
        for (i, xi) in enumerate(x)
            sigma_r = xi / spec.resolution / 1.66511
            k_min, k_max = binsearch(x, xi - 3 * sigma_r), binsearch(x, xi + 3 * sigma_r)
            instr = exp.( -1 .* ((view(x, k_min:k_max) .- xi) ./ sigma_r ) .^ 2)
            s = 0
            @inbounds for k = k_min+1:k_max
                s = s + (y[k] * instr[k-k_min+1] + y[k-1] * instr[k-k_min]) * (x[k] - x[k-1])
            end
            y_c[i] = s / 2 / sqrt(π) / sigma_r  + y[k_min] * (1 - SpecialFunctions.erf((xi - x[k_min]) / sigma_r)) / 2 + y[k_max] * (1 - SpecialFunctions.erf((x[k_max] - xi) / sigma_r)) / 2
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

    counter.next()
end


function fitLM(spec, pars)

    function lnlike(x, p)
        #println("lnlike: ", p)
        k = 1
        for i in 1:size(pars)[1]
            if pars[i].vary == 1
                pars[i].val = p[k]
                k += 1
            end
        end
        calc_spectrum(spec[1], pars, out="init")
    end

    x = spec[1].x[spec[1].mask]
    y = spec[1].y[spec[1].mask]
    w = 1 ./ spec[1].unc[spec[1].mask] .^ 2

    params = [p.val for p in pars if p.vary == true]
    lower = [p.min for p in pars if p.vary == true]
    upper = [p.max for p in pars if p.vary == true]

    println(params, " ", lower, " ", upper)

    fit = curve_fit(lnlike, x, y, w, params; lower=lower, upper=upper)
    sigma = stderror(fit)
    covar = estimate_covar(fit)

    println(dof(fit))
    println(fit.param)
    println(sigma)
    println(covar)

    return dof(fit), fit.param, sigma

end

function fitLM_2(spec, pars)

    function lnlike(x, p...)
        println("lnlike: ", p)
        for i in 1:size(pars)[1]
            l = p[i]
            if p[i] < pars[i].min
                println(typeof(p[i]), " ",  p[i], " ", pars[i].min)
                model.comp[:comp1].p[i].val = pars[i].min
                l =  pars[i].min
            elseif p[i] > pars[i].max
                model.comp[:comp1].p[i].val = pars[i].max
                l = pars[i].max
            end
            pars[i].val = l #model.comp[:comp1].p[i].val
        end

        calc_spectrum(spec[1], pars, out="init")
    end

    dom = Domain(spec[1].x[spec[1].mask])
    data = Measures(spec[1].y[spec[1].mask], spec[1].unc[spec[1].mask])

    params = [p.val for p in pars]
    println(params)
    model = Model(:comp1 => FuncWrap(lnlike, params...))

    for (i, p) in enumerate(pars)
        model.comp[:comp1].p[i].low, model.comp[:comp1].p[i].high, model.comp[:comp1].p[i].fixed = p.min, p.max, ~p.vary
        println(p, " ", model.comp[:comp1].p[i].fixed)
    end

    prepare!(model, dom, :comp1)

    result1 = fit!(model, data)
end
