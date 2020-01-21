using BenchmarkTools
using DataFitting
#using Plots
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
    vary::Int64
    addinfo::String
end

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

mutable struct spectrum
    x::Vector{Float64}
    y::Vector{Float64}
    s::Vector{Float64}
    mask::BitArray
    resolution::Float64
    lines::Vector{line}
end

function make_pars(p_pars)
    pars = Vector{par}(undef, length(collect(keys(p_pars))))
    i = 1
    for (k, p) in p_pars
        pars[i] = par(k, p.val, p.min, p.max, p.step, p.fit, p.addinfo)
        i += 1
    end
    return pars
end

function update_lines(lines, pars)
    for line in lines
        line.b = pars[line.b_ind].val
        line.logN = pars[line.N_ind].val
        line.z = pars[line.z_ind].val
        line.l = line.lam * (1 + line.z)
        line.tau0 = sqrt(π) * 0.008447972556327578 * (line.lam * 1e-8) * line.f * 10 ^ line.logN / (line.b * 1e5)
        line.a = line.g / 4 / π / line.b / 1e5 * line.lam * 1e-8
        line.ld = line.lam * line.b / 299794.26 * (1 + line.z)
    end
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

function prepare(s, pars)
    spec = Vector(undef, size(s)[1])
    for (i, si) in enumerate(s)
        spec[i] = spectrum(si.spec.norm.x, si.spec.norm.y, si.spec.norm.err, si.mask.norm.x .== 1, si.resolution, prepare_lines(si.fit_lines, pars))
    end
    return spec
end


function calc_spectrum_old(x_spec, mask, resolution, lines, pars; kind=-1)

    start = time()

    println(typeof(x_spec))
    #@time pars = make_pars(p_pars)
    #@time lines = prepare_lines(lines, pars)

    @time update_lines(lines, pars)

    x_instr = 1.0 / resolution / 2.355
    x_grid = -1 .* ones(Int8, size(x_spec)[1])
    x_grid[mask] = zeros(sum(mask))
    if 1 == 0
        num = size(x_grid)[1]
        println(num, sum(x_grid))
        for i in range(2, stop=num)
            println(i)
            if mask[i] != mask[i-1]
                k = i
                s = 1 - 2 * mask[i]
                while k > 0 && k <= num && abs(x_spec[k] / x_spec[i] - 1) <  3 * x_instr
                    x_grid[k] = 0
                    k += s
                end
            end
        end
        println(sum(x_grid))
    end
    @time for line in lines
        i_min, i_max = binsearch(x_spec, line.l * (1 - 3 * x_instr), type="min"), binsearch(x_spec, line.l * (1 + 3 * x_instr), type="max")
        if i_max - i_min > 1 && i_min > 0
            for i in range(i_min+1, i_max, step=1)
                x_grid[i] = max(x_grid[i], round(Int, (x_spec[i] - x_spec[i-1]) / line.l / x_instr * 5))
            end
        end
        line.dx = sqrt(max(-log10(0.001 / line.tau0), line.tau0 / 0.001 * line.a / sqrt(π))) * 1.5
        i_min, i_max = binsearch(x_spec, line.l - line.dx * line.ld, type="min"), binsearch(x_spec, line.l + line.dx * line.ld, type="max")
        #println((x_spec[i_max] + x_spec[i_min]) / 2, ' ', i_max-i_min, ' ', line.li, ' ', line.ld)
        if i_max - i_min > 1 && i_min > 0
            #println(x_spec[i_min], ' ', x_spec[i_max], ' ', line.l, ' ', line.ld)
            for i in range(i_min, i_max, step=1)
                #println(x_spec[i], ' ', round(Int, (x_spec[i] - x_spec[i-1]) / line.ld * 3) + 1)
                x_grid[i] = max(x_grid[i], round(Int, (x_spec[i] - x_spec[i-1]) / line.ld * 3) + 1)
            end
        end
    end

    if kind == 0
        x = x_spec[x_grid .> -1]
    else
        x = [0.0]
        k = 1
        if kind == -1
            @time for i in 1:size(x_grid)[1]
                #println(i, ' ', x_grid[i], ' ', x[k])
                if x_grid[i] == 0
                    splice!(x, k, [x_spec[i], x_spec[i]])
                    k += 1
                elseif x_grid[i] > 0
                    step = (x_spec[i+1] - x_spec[i]) / (x_grid[i] + 1)
                    splice!(x, k, range(x_spec[i], length=x_grid[i]+2, step=step))
                    k += x_grid[i]+1
                end
            end
        elseif kind > 0
            @time for i in range(1, size(x_grid)[1], step=1)
                if x_grid[i] > -1
                    step = (x_spec[i+1] - x_spec[i]) / (kind + 1)
                    splice!(x, k, range(x_spec[i], stop=x_spec[i+1], length=kind+2))
                    k += kind + 1
                end
            end
        end
        splice!(x, k, x_spec[end])
    end

    y = ones(size(x))

    @time for line in lines
        i_min, i_max = binsearch(x, line.l - line.dx * line.ld, type="min"), binsearch(x, line.l + line.dx * line.ld, type="max")
        @. @views y[i_min:i_max] = y[i_min:i_max] .* exp.(-1 .* line.tau0 .* real.(SpecialFunctions.erfcx.(line.a .- im .* (x[i_min:i_max] .- line.l) ./ line.ld)))
    end

    if resolution != 0
        y = 1 .- y
        y_c = Vector{Float64}(undef, size(y)[1])
        @time for (i, xi) in enumerate(x)
            sigma_r = xi / resolution / 1.66511
            k_min, k_max = binsearch(x, xi - 3 * sigma_r), binsearch(x, xi + 3 * sigma_r)
            instr = exp.( -1 .* ((view(x, k_min:k_max) .- xi) ./ sigma_r ) .^ 2)
            s = 0
            @inbounds for k = k_min+1:k_max
                s += (y[k] * instr[k-k_min+1] + y[k-1] * instr[k-k_min]) * (x[k] - x[k-1])
            end
            y_c[i] = s / 2 / sqrt(π) / sigma_r  + y[k_min] * (1 - SpecialFunctions.erf((xi - x[k_min]) / sigma_r)) / 2 + y[k_max] * (1 - SpecialFunctions.erf((x[k_max] - xi) / sigma_r)) / 2
        end

        println(time() - start)
        return x, 1 .- y_c
    else
        return x, y
    end
end


function calc_spectrum(spec, pars; kind=-1)

    start = time()

    @time update_lines(spec.lines, pars)

    x_instr = 1.0 / spec.resolution / 2.355
    x_grid = -1 .* ones(Int8, size(spec.x)[1])
    println(size(spec.mask), " ", size(x_grid))
    x_grid[spec.mask] = zeros(sum(spec.mask))
    if 1 == 0
        num = size(x_grid)[1]
        println(num, sum(x_grid))
        for i in range(2, stop=num)
            println(i)
            if spec.mask[i] != spec.mask[i-1]
                k = i
                s = 1 - 2 * spec.mask[i]
                while k > 0 && k <= num && abs(spec.x[k] / spec.x[i] - 1) <  3 * x_instr
                    x_grid[k] = 0
                    k += s
                end
            end
        end
        println(sum(x_grid))
    end
    @time for line in spec.lines
        i_min, i_max = binsearch(spec.x, line.l * (1 - 3 * x_instr), type="min"), binsearch(spec.x, line.l * (1 + 3 * x_instr), type="max")
        if i_max - i_min > 1 && i_min > 0
            for i in range(i_min+1, i_max, step=1)
                x_grid[i] = max(x_grid[i], round(Int, (spec.x[i] - spec.x[i-1]) / line.l / x_instr * 5))
            end
        end
        line.dx = sqrt(max(-log10(0.001 / line.tau0), line.tau0 / 0.001 * line.a / sqrt(π))) * 1.5
        i_min, i_max = binsearch(spec.x, line.l - line.dx * line.ld, type="min"), binsearch(spec.x, line.l + line.dx * line.ld, type="max")
        if i_max - i_min > 1 && i_min > 0
            for i in range(i_min, i_max, step=1)
                x_grid[i] = max(x_grid[i], round(Int, (spec.x[i] - spec.x[i-1]) / line.ld * 3) + 1)
            end
        end
    end

    if kind == 0
        x = spec.x[x_grid .> -1]
    else
        x = [0.0]
        k = 1
        if kind == -1
            @time for i in 1:size(x_grid)[1]
                #println(i, ' ', x_grid[i], ' ', x[k])
                if x_grid[i] == 0
                    splice!(x, k, [spec.x[i], spec.x[i]])
                    k += 1
                elseif x_grid[i] > 0
                    step = (spec.x[i+1] - spec.x[i]) / (x_grid[i] + 1)
                    splice!(x, k, range(spec.x[i], length=x_grid[i]+2, step=step))
                    k += x_grid[i]+1
                end
            end
        elseif kind > 0
            @time for i in range(1, size(x_grid)[1], step=1)
                if x_grid[i] > -1
                    step = (spec.x[i+1] - spec.x[i]) / (kind + 1)
                    splice!(x, k, range(spec.x[i], stop=spec.x[i+1], length=kind+2))
                    k += kind + 1
                end
            end
        end
        splice!(x, k, spec.x[end])
    end

    y = ones(size(x))

    @time for line in spec.lines
        i_min, i_max = binsearch(x, line.l - line.dx * line.ld, type="min"), binsearch(x, line.l + line.dx * line.ld, type="max")
        @. @views y[i_min:i_max] = y[i_min:i_max] .* exp.(-1 .* line.tau0 .* real.(SpecialFunctions.erfcx.(line.a .- im .* (x[i_min:i_max] .- line.l) ./ line.ld)))
    end

    if spec.resolution != 0
        y = 1 .- y
        y_c = Vector{Float64}(undef, size(y)[1])
        @time for (i, xi) in enumerate(x)
            sigma_r = xi / spec.resolution / 1.66511
            k_min, k_max = binsearch(x, xi - 3 * sigma_r), binsearch(x, xi + 3 * sigma_r)
            instr = exp.( -1 .* ((view(x, k_min:k_max) .- xi) ./ sigma_r ) .^ 2)
            s = 0
            @inbounds for k = k_min+1:k_max
                s = s + (y[k] * instr[k-k_min+1] + y[k-1] * instr[k-k_min]) * (x[k] - x[k-1])
            end
            y_c[i] = s / 2 / sqrt(π) / sigma_r  + y[k_min] * (1 - SpecialFunctions.erf((xi - x[k_min]) / sigma_r)) / 2 + y[k_max] * (1 - SpecialFunctions.erf((x[k_max] - xi) / sigma_r)) / 2
        end

        println(time() - start)
        return x, 1 .- y_c
    else
        return x, y
    end
end


function fitLM(spec, pars)

    p0 = [p.val for p in pars if p.vary == 1]

    #fit = curve_fit(model, x_spec[mask], y_spec[mask], p0)
end

