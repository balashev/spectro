using DataStructures
using Interpolations
using ImageFiltering
using LeastSquaresOptim
using LinearAlgebra
using LsqFit
using PeriodicTable
using Polynomials
using PythonCall
using Roots
using SpecialFunctions

function argclose2(x, value)
    return argmin(abs.(x .- value))
end

function convert(x, t)
    return typeof(x) != t ? pyconvert(t, x) : x
end

function convert!(x, t)
    x = typeof(x) != t ? pyconvert(t, x) : x
    return nothing
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

function mergesorted(a, b)
    if size(b)[1] > 0
        i_min = searchsortedlast(a, b[1])
        #println(i_min)
        i, k, k_ind = i_min, 1, 1
        if (i > 0)
            k_ind += Int(b[k] == a[i])
        end
        while k <= size(b)[1]
            #println(i, " ", k, " ", k_ind)
            if i >= size(a)[1]
                append!(a, b[k_ind:end])
                #println("break")
                break
            end
            if b[k] <= a[i+1]
                k += 1
            else
                o = Int(b[k-1]==a[i+1])
                #println(o)
                #println("append: ", i+1, " ", a[i+1], " ", k, " ", k_ind, " ", b[k_ind:k-o-1])
                splice!(a, i+1, vcat(b[k_ind:k-o-1], a[i+1]))
                i += maximum([1, k + 1 - o - k_ind])
                k_ind = k
                #println(a[i], " ", a[i+1])
            end
        end
        if k > k_ind
            o = Int(b[k-1]==a[i+1])
            #println("append: ", i+1, " ", a[i+1], " ", k, " ", k_ind, " ", b[k_ind:k-o-1])
            splice!(a, i+1, vcat(b[k_ind:k-o-1], a[i+1]))
        end
        #println(size(a), " ", a)
        #println(a[2:end] .- a[1:end-1])
    end
    return a
end

function Dawson(x)
    y = x .* x
    return x .* (1 .+ y .* (0.1107817784 .+ y .* (0.0437734184 .+ y .* (0.0049750952 .+ y .* 0.0015481656)))) / (1 .+ y .* (0.7783701713 .+ y .* (0.2924513912 .+ y .* (0.0756152146 .+ y .* (0.0084730365 .+ 2 .*  0.0015481656 .* y)))))
end

function make_Voigt_grid()
    a = linspace(-10, 1, 100)
    tau = linspace(-3, 10, 100)
end

function Voigt(a, x)
    return exp.(-1 .* x .* x) .* (1 .+ a^2 .* (1 .- 2 .* x .* x)) - 2 / sqrt(π) .* (1 .- 2 .* x .* Dawson.(x))
end

function voigt_range_accurate(a, level)
    f(x) = real(SpecialFunctions.erfcx(a - im * x)) - level
    if level < 1
        try
            return find_zero(f, (min(sqrt(a / level / sqrt(π)), sqrt(max(0, -log(level)))) / 2, 3 * max(sqrt(a / level / sqrt(π)), sqrt(max(0, -log(level))))), tol=0.0001)
        catch
            return find_zero(f, 3 * max(sqrt(a / level / sqrt(π)), sqrt(max(0, -log(level)))), tol=0.0001)
        end
    else
        return sqrt(max(0, -log(level)))
    end
end

function voigt_range(a, level)
    if level < 1
        α = 1.3
        if exp(sqrt(a / sqrt(π) / level)) ^ α == Inf
            return sqrt(a / sqrt(π) / level)
        else
            return log((exp(sqrt(-log(level))) ^ α + exp(sqrt(a / sqrt(π) / level)) ^ α - 1) ^ (1 / α))
        end
        #println(a, " ", level, " ", -log(level), " ", a / sqrt(π) / level)
        #return sqrt(max(-log(level), a / sqrt(π) / level))
    elseif level == 1
        return 0
    else
        return 0
    end
end

function voigt_deriv(x, a, tau_0)
    """
    The first partial derivative of exp(-H * tau_0) over the velocity (x) argument, where H is Voigt function
    """
    w = SpecialFunctions.erfcx.(a .- im .* x)
    return -1 .* exp.( - tau_0 .* real(w)) .* tau_0 .* 2 .* (imag(w) .* a .- real(w) .* x)
end

function df(x, a, tau_0)
    """
    The function to find zero of the second derivative of exp(-H * tau_0) over the velocity (x) argument:
        tau_0 * (H^1_x)^2 - H^2_x
    """
    w = SpecialFunctions.erfcx(a - im * x)
    return tau_0 * (imag(w) * a - real(w) * x)^2 + real(w) * (a^2 - x^2 + 0.5) + 2 * imag(w) * a * x - a / sqrt(π)
end

function voigt_max_deriv(a, tau_0)
    """
    Find the position of the maximum of the Voigt function derivative over velocity (x) argument
    """
    f = (x -> df(x, a, tau_0))
    level = tau_0 > 0.3 ? 0.001 / tau_0 : 1 / 3
    try
        r = find_zero(f, (0, abs(voigt_range(a, level))))
        return r
    catch
        r = find_zero(f, abs(voigt_range(a, level) / 2))
        return r
    end
end

function voigt_fwhm(a, tau_0, x0)
    """
    Find the position of the maximum of the Voigt function derivative over velocity (x) argument
    """
    #println(0.5 * (1 - exp(- tau_0)))
    f(x) = 0.5 - exp( -tau_0 * real(SpecialFunctions.erfcx(a - im * x))) + 0.5 * exp(- tau_0)
    #println(x0, " ", f(0.0), " ", f(x0), " ", exp( -tau_0 * real(SpecialFunctions.erfcx(a - im * x0))))
    try
        r = find_zero(f, (0, x0), tol=0.0001)
        return r
    catch
        r = find_zero(f, x0, tol=0.0001)
        return r
    end
end

function voigt_step(a, tau_0; tau_limit=0.001, accuracy=0.1)
    timeit = false
    #println("voigt range: ", voigt_range(a, tau_limit / tau_0))
    if timeit == 1
        start = time()
    end
    x0 = voigt_range(a, tau_limit / tau_0)
    #println(x0, " ", tau_limit, " ", real(SpecialFunctions.erfcx.(a .- im .* x0)) * tau_0, " ", tau_0)
    if timeit == 1
        println("Voigt range ", time() - start)
    end
    #x_0 = voigt_max_deriv(a, tau_0)
    x_0 = voigt_fwhm(a, tau_0, x0)
    #println("voigt_max_deriv: ",  x_0)
    x = [-abs(x_0)]
    #println("voigt_max_deriv: ",  x)
    #w = real(SpecialFunctions.erfcx.(a .- im .* x[1])) * tau_0
    der = Vector{Float64}()
    if timeit == 1
        println("max_deriv ", time() - start)
    end
    while x[end] < 0
        append!(der, voigt_deriv(x[end], a, tau_0))
        #println(x[end], " ", der[end])
        append!(x, x[end] - accuracy / der[end])
    end
    deleteat!(x, size(x))
    if timeit == 1
        println("forward step ", time() - start)
    end
    t = real(SpecialFunctions.erfcx.(a .- im .* x[1])) * tau_0
    #println(t)
    #app = false
    while t > tau_limit
        pushfirst!(x, x[1] + accuracy / der[1])
        w = SpecialFunctions.erfcx.(a .- im .* x[1])
        t = tau_0 .* real(w)
        pushfirst!(der, -exp.( - t) .* tau_0 .* 2 .* (imag(w) .* a .- real(w) .* x[1]))
        #println(x[1], " ", der[1])
        #app = true
    end
    if timeit == 1
        println("back step ", time() - start)
    end
    #println(x, " ", der)
    if size(x)[1] > 1
        deleteat!(x, 1)
        deleteat!(der, 1)
    end
    if -x0 < x[1]
        pushfirst!(x, -x0)
        pushfirst!(der, der[1])
    end


    append!(x, [0], -x[end:-1:1])
    append!(der, [0], der[end:-1:1])
    #println(x, " ", der)
    return x, der
end

function voigt_grid(l, a, tau_0; step=0.03)
    x, r = voigt_step(a, tau_0)
    k_min, k_max = binsearch(l, x[1], type="min"), binsearch(l, x[end], type="max") - 1
    g = Vector{Float64}()
    for k in k_min:k_max
        i_min, i_max = binsearch(x, l[k]), binsearch(x, l[k+1])
        #append!(g, maximum(r[i_min:i_max]))
        append!(g, Int(floor((l[k+1] - l[k]) / (step / maximum(r[i_min:i_max])))) + 1)
    end
    return k_min:k_max, g
end

function z_to_v(;z=nothing, v=nothing, z_ref=0)
    c = 299792.458
    if v == nothing
        return c * (z - z_ref) / (1 + z_ref)
    elseif z == nothing
        return z_ref + v / c * (1 + z_ref)
    end
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
    tied::String
    fit::Bool
    unc::Float64
    ref
end

function make_pars(p_pars; tieds=Dict(), z_ref=nothing, parnames=nothing)
    #println(p_pars)
    pars = OrderedDict{String, par}()
    if parnames != nothing
        for p in parnames
            pars[p] = p_pars[p]
        end
    else
        for p in p_pars
            if occursin("z_", pyconvert(String, p.__str__())) * (z_ref == true)
                pars[pyconvert(String, p.__str__())] = par(pyconvert(String, p.__str__()), 0.0, z_to_v(z=pyconvert(Float64, p.min), z_ref=pyconvert(Float64, p.val)), z_to_v(z=pyconvert(Float64, p.max), z_ref=pyconvert(Float64, p.val)), pyconvert(Float64, p.step), pyconvert(Bool, p.fit * p.vary), pyconvert(String, p.addinfo), "", false, 0.0, pyconvert(Float64, p.val))
            else
                pars[pyconvert(String, p.__str__())] = par(pyconvert(String, p.__str__()), pyconvert(Float64, p.val), pyconvert(Float64, p.min), pyconvert(Float64, p.max), pyconvert(Float64, p.step), pyconvert(Bool, p.fit * p.vary), pyconvert(String, p.addinfo), "", false, 0.0, nothing)
            end
            if occursin("cf", pyconvert(String, p.__str__()))
                pars[pyconvert(String, p.__str__())].min, pars[pyconvert(String, p.__str__())].max = 0, 1
            end
            #println(p, " ", pars[p.__str__()])
        end
        for (k, v) in tieds
            pars[pyconvert(String, k)].vary = false
            pars[pyconvert(String, k)].tied = pyconvert(String, v)
        end
    end
    for (k, p) in pars
        pars[k].fit = copy(pars[k].vary)
    end
    return pars
end

function get_element_name(name)
    st = name
    if occursin("j", st)
        st = st[1:findfirst("j", st)[1]-1]
    end
    for s in ["I", "V", "X", "*"]
        st = replace(st, s => "")
    end
    return st
end

function doppler(name, turb, kin)
    name = get_element_name(name)
    if name == "D"
        mass = 2
    else
        for e in elements
            if e.symbol == name
                mass = e.atomic_mass.val
            end
        end
        #mass = element(get_element_name(name)).atomic_mass.val
    end
    #println(mass)
    #println(element(get_element_name(name)).atomic_mass.val)
    return (turb ^ 2 + 0.0164 * kin / mass) ^ .5
end

function abundance(name, logN_ref, me)
    d = Dict([
    ("H", [12, 0, 0]),
    ("D", [7.4, 0, 0]), # this is not in Asplund
    ("He", [10.93, 0.01, 0.01]), #be carefull see Asplund 2009
    ("Li", [1.05, 0.10, 0.10]),
    ("Be", [1.38, 0.09, 0.09]),
    ("B", [2.70, 0.20, 0.20]),
    ("C", [8.43, 0.05, 0.05]),
    ("N", [7.83, 0.05, 0.05]),
    ("O", [8.69, 0.05, 0.05]),
    ("F", [4.56, 0.30, 0.30]),
    ("Ne", [7.93, 0.10, 0.10]),  #be carefull see Asplund 2009
    ("Na", [6.24, 0.04, 0.04]),
    ("Mg", [7.60, 0.04, 0.04]),
    ("Al", [6.45, 0.03, 0.03]),
    ("Si", [7.51, 0.03, 0.03]),
    ("P", [5.41, 0.03, 0.03]),
    ("S", [7.12, 0.03, 0.03]),
    ("Cl", [5.50, 0.30, 0.30]),
    ("Ar", [6.40, 0.13, 0.13]),  #be carefull see Asplund 2009
    ("K", [5.03, 0.09, 0.09]),
    ("Ca", [6.34, 0.04, 0.04]),
    ("Sc", [3.15, 0.04, 0.04]),
    ("Ti", [4.95, 0.05, 0.05]),
    ("V", [3.93, 0.08, 0.08]),
    ("Cr", [5.64, 0.04, 0.04]),
    ("Mn", [5.43, 0.04, 0.04]),
    ("Fe", [7.50, 0.04, 0.04]),
    ("Co", [4.99, 0.07, 0.07]),
    ("Ni", [6.22, 0.04, 0.04]),
    ("Cu", [4.19, 0.04, 0.04]),
    ("Zn", [4.56, 0.05, 0.05]),
    ("Ga", [3.04, 0.09, 0.09]),
    ("Ge", [3.65, 0.10, 0.10]),
    ])
    return logN_ref - (12 - d[get_element_name(name)][1]) + me
end

function update_pars(pars, spec, add)
    for (k, v) in pars
        if v.tied != ""
            pars[k].val = pars[v.tied].val
        end
        if occursin("res", pars[k].name)
            #println(pars[k].name, " ", pars[k].val, " ", parse(Int, pars[k].addinfo[5:end]))
            spec[parse(Int, pars[k].addinfo[5:end]) + 1].resolution = pars[k].val
        end
        if occursin("displ", pars[k].name)
            spec[parse(Int, split(pars[k].addinfo, "_")[2]) + 1].displ = pars[k].val
        end
        if occursin("disps", pars[k].name)
            spec[parse(Int, split(pars[k].addinfo, "_")[2]) + 1].disps = pars[k].val
        end
        if occursin("dispz", pars[k].name)
            spec[parse(Int, split(pars[k].addinfo, "_")[2]) + 1].dispz = pars[k].val
        end
        if occursin("Ntot", pars[k].name)
            ind = split(pars[k].name, "_")[2]
            pr = add["pyratio"][parse(Int, ind) + 1]
            x = pyratio_predict(pr, pars)
            col = [v.val - log10(sum(x)) + log10(x[i]) for i in 1:pr.num]
            #println("julia: ", col)
            for (k1, v1) in pars
                if startswith(k1, "N_" * ind * "_" * pr.species) * occursin("Ntot", v1.addinfo)
                    i = occursin(pr.species * "j", k1) ? parse(Int, replace(k1, "N_" * ind * "_" * pr.species * "j" => "")) + 1 : 1
                    pars[k1].val = col[i]
                end
            end
        end
        #println(occursin("iso", pars[k].name))
        if occursin("iso", pars[k].name)
            for (k1, v1) in pars
                if occursin("D/H", pars[k].addinfo) * occursin("N_", k1) * occursin("DI", k1) * occursin("iso", v1.addinfo)
                    pars[k1].val = pars[replace(k1, "DI" => "HI")].val + pars[k].val
                end
                if occursin("13C/12C", pars[k].addinfo) * occursin("N_", k1) * occursin("13CI", k1) * occursin("iso", v1.addinfo) * haskey(pars, replace(k1, "13" => ""))
                    pars[k1].val = pars[replace(k1, "13" => "")].val + pars[k].val
                end
                #println(occursin("13C/12C", pars[k].addinfo), " ", occursin("N_", k1) * occursin("13CO", k1), " ", haskey(pars, replace(k1, "13" => "")))
                if occursin("13C/12C", pars[k].addinfo) * occursin("N_", k1) * occursin("13CO", k1) * occursin("iso", v1.addinfo) * haskey(pars, replace(k1, "13" => ""))
                    pars[k1].val = pars[replace(k1, "13" => "")].val + pars[k].val
                end
            end
        end
        if occursin("me", pars[k].name)
            for (k1, v1) in pars
                if occursin(pars[k].name, v1.addinfo)
                    pars[k1].val = abundance(split(k1, "_")[3], pars[replace(k1, split(k1, "_")[3] => "HI")].val, pars[k].val)
                end
            end
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
        priors[k] = prior(pyconvert(String, k), pyconvert(Float64, p.val), pyconvert(Float64, p.plus), pyconvert(Float64, p.minus))
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
    max_deriv::Float64
    cf::Int64
    stack::Int64
    stv::Dict{}
end

function update_lines(lines, pars; comp=0, tau_limit=0.001)
    mask = Vector{Bool}(undef, 0)
    for line in lines
        if pars["b_" * string(line.sys) * "_" * line.name].addinfo == "consist"
            line.b = doppler(line.name, pars["turb_" * string(line.sys)].val, pars["kin_" * string(line.sys)].val)
        elseif pars["b_" * string(line.sys) * "_" * line.name].addinfo != ""
            line.b = pars["b_" * string(line.sys) * "_" * pars["b_" * string(line.sys) * "_" * line.name].addinfo].val
        else
            line.b = pars["b_" * string(line.sys) * "_" * line.name].val
        end
        line.logN = pars["N_" * string(line.sys) * "_" * line.name].val
        if pars["z_" * string(line.sys)].ref == nothing
            line.z = pars["z_" * string(line.sys)].val
        else
            line.z = z_to_v(v=pars["z_" * string(line.sys)].val, z_ref=pars["z_" * string(line.sys)].ref)
        end
        line.l = line.lam * (1 + line.z)
        line.tau0 = sqrt(π) * 0.008447972556327578 * (line.lam * 1e-8) * line.f * 10 ^ line.logN / (line.b * 1e5)
        line.a = line.g / 4 / π / line.b / 1e5 * line.lam * 1e-8
        line.ld = line.lam * line.b / 299794.26 * (1 + line.z)
        if line.stack > -1
            line.stv = Dict([s => pars[s * "_" * string(line.stack)].val for s in ["sts", "stNl", "stNu"]])
        end
        append!(mask, (comp == 0) || (line.sys == comp - 1))
    end
    return mask
end

function prepare_lines(lines)
    fit_lines = Vector{line}(undef, pylen(lines))
    for (i, l) in enumerate(lines)
        fit_lines[i] = line(pyconvert(String, l.name),
                            pyconvert(Int64, l.sys),
                            pyconvert(Float64, l.l()),
                            pyconvert(Float64, l.f()),
                            pyconvert(Float64, l.g()),
                            pyconvert(Float64, l.b),
                            pyconvert(Float64, l.logN),
                            pyconvert(Float64, l.z),
                            pyconvert(Float64, l.l()*(1+l.z)),
                            0, 0, 0, 0, 0,
                            pyconvert(Int64, l.cf),
                            pyconvert(Int64, l.stack),
                            Dict())
        #fit_lines[i] = line(l.name, l.sys, l.l(), l.f(), l.g(), l.b, l.logN, l.z, l.l()*(1+l.z), 0, 0, 0, 0, 0, l.cf, l.stack, Dict())
    end
    return fit_lines
end

function prepare_cheb(pars, ind)
    cont = []
    d = [[parse(Int, split(k, "_")[2]), parse(Int, split(k, "_")[3]), parse(Int, split(v.addinfo, "_")[3]), v] for (k, v) in pars if occursin("cont_", k)]
    if length(d) > 0
        d = permutedims(reshape(hcat(d...), (length(d[1]), length(d))))
        for k in unique(d[:, 1][d[:, 3] .== ind - 1])
            append!(cont, [cheb([], 0, 0, 0)])
            for i in sort(d[:, 2][(d[:, 1] .== k) .& (d[:, 3] .== ind - 1)])
                p = d[:, 4][(d[:, 1] .== k) .& (d[:, 2] .== i) .& (d[:, 3] .== ind - 1)]
                if i == 0
                    cont[end].left, cont[end].right, cont[end].disp = parse(Float64, split(split(p[1].addinfo, "_")[1], "..")[1]), parse(Float64, split(split(p[1].addinfo, "_")[1], "..")[2]), parse(Float64, split(p[1].addinfo, "_")[4])
                end
                append!(cont[end].c, [p[1].name])
            end
        end
    end
    return cont
end

function prepare_coll(pr, s)
    #println(keys(pr.species[s].coll))
    c = Dict()
    for sp in pr.species[s].coll.keys()
        num = pyconvert(Int64, pr.species[s].num)
        c[pyconvert(String, sp)] = Array{coll}(undef, num, num)
        #println(pr.species[s].num, " ", pr.species[s].coll[sp].c[1].rates)
        for i in 1:num
            for j in 1:num
                if i != j
                    if pyconvert(Bool, pytype(pr.species[s].coll[sp].c[0].rates) == pytype(pybuiltins.None))
                        coll(i, j, LinearInterpolation([0, 6], [0, 0], extrapolation_bc=Flat()))
                    else
                        c[pyconvert(String, sp)][i,j] = coll(i, j, LinearInterpolation(pyconvert(Vector{Float64}, pr.species[s].coll[sp].c[1].rates[0]), pyconvert(Vector{Float64}, pr.species[s].coll[sp].rate(pyint(i-1), pyint(j-1), pr.species[s].coll[sp].c[1].rates[0])), extrapolation_bc=Flat()))
                    end
                end
            end
        end
        #for col in pr.species[s].coll[sp].c
        #    println(col.rate(1, 0, 2), " ", col.rate(0, 1, 2))
        #end
    end
    return c
end

function prepare_add(fit, pars)
    add = Dict()
    if any(occursin("Ntot", pyconvert(String, k)) for k in keys(pars))
        add["pyratio"] = Dict()
    end
    for (k, v) in pars
        if occursin("Ntot", pyconvert(String, pars[k].name))
            ind = split(pyconvert(String, pars[k].name), "_")[2]
            #add["pyratio"][parse(Int, split(pars[k].name, "_")[2]) + 1] = fit.sys[parse(Int, split(pars[k].name, "_")[2])+1].pr
            pr = fit.sys[parse(Int, ind)].pr

            #s = collect(keys(pr.species))[1]
            s = pylist(pr.species.keys())[0]
            #println(prepare_coll(pr, s))
            #println(ind)
            #println(collect(keys(pr.pars)))
            #println(keys(pr.species))
            #println(pr.balance(debug="A"))
            #println(pr.balance(debug="C"))
            #println(pr.balance(debug="IR"))
            #println(pr.balance(debug="UV"))
            add["pyratio"][parse(Int, ind) + 1] = pyratio(ind,
                                                          collect(pyconvert(Array{String}, pr.pars.keys())),
                                                          pyconvert(String, s),
                                                          pyconvert(Int64, pr.species[s].num),
                                                          pyconvert(Array{Float64, 2}, pr.species[s].Eij),
                                                          pyconvert(Array{Float64, 2}, pr.species[s].Aij),
                                                          pyconvert(Array{Float64, 2}, pr.species[s].Bij),
                                                          pyconvert(Array{Float64, 2}, pr.balance(debug="C")),
                                                          pyconvert(Array{Float64, 2}, pr.species[s].rad_rate),
                                                          pyconvert(Array{Float64, 2}, pr.species[s].pump_rate),
                                                          prepare_coll(pr, s))
        end
    end
    #println(add)
    return add
end

function escape_prob(N)
    esc = 1
    if N > 15
        esc = esc * exp(- (N - 15) / 2)
    end
    return esc
end

##############################################################################
function pyratio_predict(pr, pars)
    #pars["logT_" * ind].val
    #println("Ntot_" * pr.ind, " ", pars["Ntot_" * pr.ind].val, " ", escape_prob(pars["Ntot_" * pr.ind].val))
    #println(pars["Ntot_" * pr.ind].val, " ", escape_prob(pars["Ntot_" * pr.ind].val))
    W = pr.Aij .* escape_prob(pars["Ntot_" * pr.ind].val)

    update_coll_rate(pr, pars)
    if any(s in pr.pars for s in ["n", "e"])
        W = W .+ pr.coll_rate
    end

    if "CMB" in pr.pars
        TCMB = pars["CMB_" * pr.ind].val
    else
        TCMB = 2.726 * (pars["z_" * pr.ind].val + 1)
    end
    W = W .+ cmb_rate(pr.Bij, pr.Eij, TCMB) .* escape_prob(pars["Ntot_" * pr.ind].val)

    if "rad" in pr.pars
        W = W .+ pr.rad_rate .* 10 ^ pars["rad_" * pr.ind].val .* escape_prob(pars["Ntot_" * pr.ind].val)
    end
    if "rad" in pr.pars
        W = W .+ pr.pump_rate .* 10 ^ pars["rad_" * pr.ind].val
    end
    #println(W)
    K = copy(transpose(W))
    for i = 1:size(W)[1]
        K[i, i] -= sum(W, dims=2)[i]
    end
    #println(K)
    #println(K[2:end, 2:end])
    #println(K[2:end, 1])
    #println(K[2:end, 2:end] \ (-1 .* K[2:end, 1]))
    return insert!(abs.(K[2:end, 2:end] \ (-1 .* K[2:end, 1])), 1, 1)
end

function cmb_rate(Bij, Eij, TCMB)
    return Bij .* 8 .* π .* 6.62607015e-27 .* Eij .^ 3 ./ (exp.(Eij .* 6.62607015e-27 ./ 1.380649e-16 .* 2.99792458e10 ./ TCMB) .- 1 .+ 1.66533e-16)
end

function update_coll_rate(pr, pars)
    f_He = 0 #0.08
    for i in 1:pr.num
        for j in 1:pr.num
            pr.coll_rate[i, j] = 0
            if i != j
                for p in pr.pars
                    if p in ["e", "H"]
                        pr.coll_rate[i, j] += 10 ^ (pars["logn_" * pr.ind].val) * pr.coll[p][i, j].rate(pars["logT_" * pr.ind].val)
                    elseif p in ["n"]
                        m_fr = "f" in pr.pars ? 10 ^ pars["logf_" * pr.ind].val : mol_fr
                        f_HI, f_H2 = (1 - m_fr) / (f_He + 1 - m_fr / 2), m_fr / 2 / (f_He + 1 - m_fr / 2)
                        pr.coll_rate[i, j] += 10 ^ (pars["logn_" * pr.ind].val) * (pr.coll["H"][i, j].rate(pars["logT_" * pr.ind].val)) * f_HI
                        otop = 9 * exp(-170.6 / 10 ^ pars["logT_" * pr.ind].val)
                        #println(m_fr, " ", f_HI, " ", f_H2, " ", otop)
                        pr.coll_rate[i, j] += 10 ^ (pars["logn_" * pr.ind].val) * pr.coll["pH2"][i, j].rate(pars["logT_" * pr.ind].val) * f_H2 / (1 + otop)
                        pr.coll_rate[i, j] += 10 ^ (pars["logn_" * pr.ind].val) * pr.coll["oH2"][i, j].rate(pars["logT_" * pr.ind].val) * f_H2 * otop / (1 + otop)
                        if f_He != 0
                            pr.coll_rate[i, j] += 10 ^ (pars["logn_" * pr.ind].val) * pr.coll["He4"][i, j].rate(pars["logT_" * pr.ind].val) * f_He / (f_He + 1 - m_fr / 2)
                        end
                    end
                end
            end
        end
    end
    #println(pr.coll_rate)
end

mutable struct coll
    i::Int64
    j::Int64
    rate::Interpolations.Extrapolation
end

mutable struct pyratio
    ind::String
    pars::Array{String}
    species::String
    num::Int64
    Eij::Array{Float64, 2}
    Aij::Array{Float64, 2}
    Bij::Array{Float64, 2}
    coll_rate::Array{Float64, 2}
    rad_rate::Array{Float64, 2}
    pump_rate::Array{Float64, 2}
    coll::Dict{}
end

##############################################################################
mutable struct cheb
    c::Vector{String}
    left::Float64
    right::Float64
    disp::Float64
end

module spectrum
    using Main: line
    using Interpolations
    #using Dierckx

    mutable struct spec
        x::Vector{Float64}
        y::Vector{Float64}
        unc::Vector{Float64}
        mask::BitArray
        sky::Interpolations.Extrapolation
        lsf_type::String
        resolution::Float64
        lines::Vector{line}
        displ::Float64
        disps::Float64
        dispz::Float64
        cont::Vector{Any}
        bins::Vector{Any}
        bin_mask::BitArray
        cos::Interpolations.Extrapolation
        #cos::Dierckx.Spline2D
        cos_disp::Float64
    end

    function COS(obj::spec, x, xcenter)
        return obj.cos.((x .- xcenter) ./ obj.cos_disp, xcenter) ./ obj.cos_disp
    end

    function Base.getproperty(obj::spec, prop::Symbol)
        if prop in fieldnames(spec)
            return getfield(obj, prop)
        elseif prop == :COS
            return (x, xcenter,)->COS(obj, x, xcenter)
        else
            throw(UndefVarError(prop))
        end
    end
end

function prepare_COS(s)
    cos = Dict()
    for (i, si) in enumerate(s)
        if pyconvert(String, si.lsf_type) == "cos"
            t = si.prepare_COS()
            cos[i] = [pyconvert(Vector{Float64}, t[0]), pyconvert(Vector{Float64}, t[1]), pyconvert(Array{Float64}, t[2]), pyconvert(Float64, t[3])]
        else
            x = 0:1:1
            cos[i] = (x, x, [xs + ys for xs in x, ys in x], 0)
        end
    end
    #println("cos ", cos)
    return cos #LinearInterpolation((lsf_cent, pix), lsf)
end

function prepare(s, pars, add, COS)
    #print("COS ", COS)
    #c = Vector{Any}
    #println(append!(c,  [cheb([], 0, 0, 0)]))
    spec = Vector(undef, size(s)[1])
    for (i, si) in enumerate(s)
        #println("sky ", si.sky_cont.norm.x, " ", si.sky_cont.norm.y)
        #println(isempty(si.sky.norm.x), " ", isempty(si.sky_cont.norm.x))
        #println(size(si.sky.norm.x))
        spec[i] = spectrum.spec(pyconvert(Vector{Float64}, si.spec.norm.x),
                                pyconvert(Vector{Float64}, si.spec.norm.y),
                                pyconvert(Vector{Float64}, si.spec.norm.err),
                                pyconvert(Vector{Int64}, si.mask.norm.x) .== 1,
                                linear_interpolation(pyconvert(Vector{Float64}, si.sky_cont.norm.x), pyconvert(Vector{Float64}, si.sky_cont.norm.y), extrapolation_bc=Flat()),
                                pyconvert(String, si.lsf_type),
                                pyconvert(Float64, si.resolution),
                                prepare_lines(si.fit_lines),
                                NaN,
                                NaN,
                                NaN,
                                prepare_cheb(pars, i),
                                Vector(undef, 0), BitArray(0), linear_interpolation((COS[i][1], COS[i][2]), COS[i][3], extrapolation_bc=Flat()), COS[i][4])
        #spec[i] = spectrum.spec(si.spec.norm.x, si.spec.norm.y, si.spec.norm.err, si.mask.norm.x .== 1, si.lsf_type, si.resolution, prepare_lines(si.fit_lines), NaN, NaN, NaN, prepare_cheb(pars, i), Vector(undef, 0), BitArray(0), Spline2D(COS[i][1], COS[i][2], COS[i][3]; kx=3, ky=3, s=0.0), COS[i][4])
        spec[i].bins = (spec[i].x[2:end] + spec[i].x[1:end-1]) / 2
        spec[i].bin_mask = spec[i].mask[1:end-1] .|| spec[i].mask[2:end]
    end
    update_pars(pars, spec, add)
    return spec
end

function correct_continuum(conts, pars, x)
    c = ones(size(x))
    if size(conts)[1] > 0
        for cont in conts
             m = (x .> cont.left) .& (x .< cont.right)
             cheb = ChebyshevT([pars[name].val for name in cont.c])
             c[m] = cheb.((x[m] .- cont.left) .* 2 ./ (cont.right - cont.left) .- 1)
             #if any([occursin("hcont", p.first) for p in pars])
             #   c[m] *= (1 + parse(Float64, split(pars[cont.c[1]].addinfo, "_")[4]) * randn(1)[1] * pars["hcont"].val)
             #   #println(parse(Float64, split(pars[cont.c[1]].addinfo, "_")[4]), " ", randn(1)[1], " ", pars["hcont"].val)
             #end
        end
    end
    return c
end

function line_profile(line, x; toll=1e-6)
    if line.stack == -1
        return exp.( - line.tau0 .* real.(SpecialFunctions.erfcx.(line.a .- im .* (x .- line.l) ./ line.ld)))
    else
        tau = line.tau0 .* real.(SpecialFunctions.erfcx.(line.a .- im .* (x .- line.l) ./ line.ld))
        tau_low =  tau .* 10 ^ (line.stv["stNl"] - line.logN)
        tau_up = tau .* 10 ^ (line.stv["stNu"] - line.logN)
        p = (exp.(- tau_low) .- exp.(- tau_up) .* 10 ^ ((line.stv["stNu"] - line.stv["stNl"]) * (line.stv["sts"] + 1)) .- (gamma.(line.stv["sts"] + 2, tau_low) .- gamma.(line.stv["sts"] + 2, tau_up)) ./ tau_low .^ (line.stv["sts"] + 1)) ./ (1 - 10 ^ ((line.stv["stNu"]-line.stv["stNl"]) * (line.stv["sts"] + 1)))
        p[p .< toll] .= toll
        return p
    end
end

function make_grid(spec, lines; grid_type="uniform", grid_num=1, binned=true, tau_limit=0.005, accuracy=0.2)

    timeit = 0
    if timeit == 1
        start = time()
        println("regular ", grid_type)
    end
    x_instr = 1.0 / spec.resolution / 2.355
    #println(spec.lsf_type, " ", spec.resolution, " ", x_instr)

    #println("spec_bins: ", size(spec.bins), " ", size(spec.bin_mask))
    if grid_type in ["uniform", "adaptive"]
        if binned
            xs, mask = copy(spec.bins), copy(spec.bin_mask)
        else
            xs, mask = copy(spec.x), copy(spec.mask)
        end
        x_grid = -1 .* ones(Int8, size(xs)[1])
    end
    #println("x_grid-1: ", size(x_grid[x_grid .>= 0]))

    # >>> regular grid of the pixels:
    if grid_type == "uniform"
        x_grid[mask] = ones(Int8, sum(mask)) .* grid_num
        for line in lines
            line.dx = voigt_range(line.a, tau_limit / line.tau0)
            d = isfinite(x_instr) ? 4 * x_instr : 0
            i_min, i_max = binsearch(xs, (line.l - line.dx * line.ld) * (1 - d), type="min"), binsearch(xs, (line.l + line.dx * line.ld) * (1 + d), type="max")
            if i_max - i_min > 0 && i_min > 0 && i_max <= length(xs)
                x_grid[i_min:i_max] .= grid_num
            end
        end

        #println(sum(x_grid[x_grid .> -1]), " ", x_grid[x_grid .> -1])
        if grid_num == 0
            x = xs[x_grid .> -1]
        else
            x = [0.0]
            k = 1
            for i in 1:size(x_grid)[1]-1
                if x_grid[i] > -1 && x_grid[i+1] > -1
                    #println(i, " ", spec.bins[i], " ", spec.bins[i+1])
                    step = (xs[i+1] - xs[i]) / (grid_num + 1)
                    splice!(x, k, range(xs[i], stop=xs[i+1], length=grid_num+2))
                    k += x_grid[i] + 1
                end
            end
        end

    # >>> adaptive grid pixel based:
    elseif grid_type == "adaptive"
        x_grid[mask] = ones(Int8, sum(mask)) .* grid_num
        if isfinite(x_instr)
            for i in findall(!=(0), mask[2:end])
                #println(round(Int, (spec.bins[i] - spec.bins[i-1]) / spec.bins[i] / x_instr * 2))
                x_grid[i] = max(x_grid[i], round(Int, (1 - xs[i-1] / xs[i]) / x_instr * 2))
            end
            for i in findall(!=(0), mask[1:end-1] - mask[2:end])
                for k in binsearch(xs, xs[i] * (1 - 6 * x_instr), type="min"):binsearch(xs, xs[i] * (1 + 6 * x_instr), type="max")
                    x_grid[k] = max(x_grid[k], round(Int, (1 - xs[i-1] / xs[i]) / x_instr * 2))
                end
            end
        end

        for line in lines
            line.dx = voigt_range(line.a, tau_limit / line.tau0)
            line.max_deriv = voigt_max_deriv(line.a, line.tau0)
            d = isfinite(x_instr) ? 2 * x_instr : 0
            i_min, i_max = binsearch(xs, (line.l - line.dx * line.ld) * (1 - d), type="min"), binsearch(xs, (line.l + line.dx * line.ld) * (1 + d), type="max")
            if i_max - i_min > 0 && i_min > 0 && i_max <= length(spec.bins)
                t = (xs[i_min:i_max] .- line.l) ./ line.ld
                for i in i_min:i_max-1
                    x_grid[i] = max(x_grid[i], Int(floor((t[i-i_min+2] - t[i-i_min+1]) / (accuracy / maximum([abs(voigt_deriv(t[i-i_min+1], line.a, line.tau0)), abs(voigt_deriv(t[i-i_min+2], line.a, line.tau0))])))+binned))
                end
            end
            for m in [line.l - line.max_deriv * line.ld, line.l + line.max_deriv * line.ld]
                i_min = binsearch(xs, m, type="min")
                if i_min > 0 && i_min < size(xs)[1]
                    x_grid[i_min] = max(x_grid[i_min], Int(floor((xs[i_min+1] - xs[i_min]) / (accuracy / voigt_deriv(line.max_deriv, line.a, line.tau0) * line.ld))) + binned)
                end
            end
        end
        x_grid .+= iseven.(x_grid)
        if timeit == 1
            println("update ", time() - start)
        end

        x_grid[x_grid .>= 0] = round.(imfilter(x_grid[x_grid .>= 0], ImageFiltering.Kernel.gaussian((2,))))

        if timeit == 1
            println("grid conv ", time() - start)
        end

        x = [0.0]
        k = 1
        for i in 1:size(x_grid)[1]-1
            if x_grid[i] == 0
                splice!(x, k, [xs[i], xs[i]])
                k += 1
            elseif x_grid[i] > 0
                splice!(x, k, range(xs[i], length=x_grid[i]+2, step=(xs[i+1] - xs[i]) / (x_grid[i] + 1)))
                k += x_grid[i] + 1
            end
        end

    # >>> minimized grid pixel based:
    elseif grid_type == "minimized"

        if timeit == 1
            println("merge ", time() - start)
        end

        mask = copy(spec.mask)
        for i in 1:5
            mask[1+i:end] .|= spec.mask[1:end-i]
            mask[1:end-i] .|= spec.mask[1+i:end]
        end

        if binned
            x = mergesorted(spec.bins[spec.bin_mask], spec.x[mask])
        else
            x = copy(spec.x[mask])
        end

        if grid_num > 0
            for i in size(x)[1]-1:-1:1
                d = x[i+1] - x[i]
                for k in grid_num:-1:1
                    insert!(x, i+1, (x[i] + d * k / (grid_num + 1)))
                end
            end
        end

        if timeit == 1
            println("mask ", time() - start)
        end
        for line in lines
            if line.tau0 > tau_limit
                start = time()
                #println(line.l, " ", line.tau0, " ", line)
                line.dx = voigt_range(line.a, tau_limit / line.tau0)
                d = isfinite(x_instr) ? 2 * x_instr : 0
                i_min, i_max = binsearch(spec.x, (line.l - line.dx * line.ld) * (1 - d), type="min"), binsearch(spec.x, (line.l + line.dx * line.ld) * (1 + d), type="max")
                #println(i_min, " ", i_max, " ", size(spec.x), " ", spec.x[end], " ", (line.l + line.dx * line.ld) * (1 + d))
                if i_max > i_min
                    if binned
                        x = mergesorted(x, mergesorted(spec.x[i_min:i_max], (spec.x[i_min+1:i_max] + spec.x[i_min:i_max-1]) / 2))
                    else
                        x = mergesorted(x, spec.x[i_min:i_max])
                    end
                    #println(size(x)[1], " ", size(x)[1] - size(mergesorted(spec.x[i_min:i_max], (spec.x[i_min+1:i_max] + spec.x[i_min:i_max-1]) / 2))[1])
                end
                if timeit == 1
                    println("add bins ", time() - start)
                end
                step = line.l .+ voigt_step(line.a, line.tau0, tau_limit=tau_limit, accuracy=accuracy)[1] .* line.ld
                #println(line.name, " ", size(step))
                #println(voigt_step(line.a, line.tau0, tau_limit=tau_limit)[1])
                #i_min = binsearch(x, step[1], type="min"), i_min = binsearch(x, step[end], type="max")
                if timeit == 1
                    println("add step ", time() - start)
                end
                x = mergesorted(x, step)
                if timeit == 1
                    println("merge step ", time() - start)
                end
                #println(line.name, " ", size(x))
            end
        end
        if timeit == 1
            println("add lines ", time() - start)
        end
    end

    i_min, i_max = binsearch(x, spec.x[1], type="max"), binsearch(x, spec.x[end], type="min")
    return x[i_min:i_max]
end

function calc_spectrum(spec, pars; comp=0, x=nothing, grid_type="minimized", grid_num=1, binned=true, out="all", telluric=false, tau_limit=0.005, accuracy=0.2)

    timeit = 0
    if timeit == 1
        start = time()
    end
    #println(grid_type, " ", grid_num, " ", binned)

    line_mask = update_lines(spec.lines, pars, comp=comp)

    if x == nothing
        x = make_grid(spec, spec.lines[line_mask], grid_type=grid_type, grid_num=grid_num, binned=binned, tau_limit=tau_limit, accuracy=accuracy)
    end

    if timeit == 1
        println("make grid ", time() - start)
    end

    if ~any([occursin("cf", p.first) for p in pars])
        y = ones(size(x))
        for line in spec.lines[line_mask]
            if (line.tau0 > tau_limit)
                i_min, i_max = binsearch(x, line.l - line.dx * line.ld, type="min"), binsearch(x, line.l + line.dx * line.ld, type="max")
                t = line_profile(line, x[i_min:i_max])
                #println("t: ", t)
                @. @views y[i_min:i_max] = y[i_min:i_max] .* t
            end
        end
    else
        y = zero(x)
        cfs, inds = [], []
        for (i, line) in enumerate(spec.lines[line_mask])
            append!(cfs, line.cf)
            append!(inds, i)
        end
        for l in unique(cfs)
            if l > -1
                cf = pars["cf_" * string(l)].val
            else
                cf = 1
            end
            profile = zero(x)
            for (i, c) in zip(inds, cfs)
                if c == l
                    line = spec.lines[line_mask][i]
                    i_min, i_max = binsearch(x, line.l - line.dx * line.ld, type="min"), binsearch(x, line.l + line.dx * line.ld, type="max")
                    t = line_profile(line, x[i_min:i_max])
                    @. @views profile[i_min:i_max] += log.(t)
                end
            end
            #y += log.(exp.(profile) .* cf .+ (1 .- cf))
            y += real.(log.(exp.(profile) .* cf .+ (1 .- cf) .+ 0im))
        end
        y = exp.(y)
    end

    if timeit == 1
        println("calc lines ", time() - start)
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
    if (~isnan(spec.displ)) & (~isnan(spec.disps)) & (~isnan(spec.dispz))
        inter = LinearInterpolation(x, y, extrapolation_bc=Flat())
        y = inter(x .+ (x .- spec.displ) .* spec.disps .+ spec.dispz)
    end

    if size(spec.cont)[1] > 0
        y .*= correct_continuum(spec.cont, pars, x)
    end

    if any([occursin("zero", p.first) for p in pars])
        y .+= pars["zero"].val
    end

    if (spec.resolution > 0) && (spec.lsf_type != "none")
        y = 1 .- y
        y_c = zero(y)
        if spec.lsf_type == "gauss" # Gaussian function
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

        elseif spec.lsf_type == "cos" # COS line spread function
            for (i, xi) in enumerate(x)
                sigma_r = xi / spec.resolution / 1.66511
                k_min, k_max = binsearch(x, xi - 7 * sigma_r), binsearch(x, xi + 7 * sigma_r)
                #println(k_min, "  ", k_max)
                instr = spec.COS(view(x, k_min:k_max), xi)
                #instr = exp.( -1 .* ((view(x, k_min:k_max) .- xi) ./ sigma_r ) .^ 2)
                s = 0
                @inbounds for k = k_min+1:k_max
                    s = s + (y[k] * instr[k-k_min+1] + y[k-1] * instr[k-k_min]) * (x[k] - x[k-1])
                end
                y_c[i] = s / 2
                #sleep(5)
            end
        end
        if timeit == 1
            println("convolve ", time() - start)
        end

        y_c = 1 .- y_c
    else
        y_c = y
    end

    #println(~isempty(knots(spec.sky)))
    if (comp == 0) & (telluric==true) & ~isempty(knots(spec.sky))
        #println("telluric: ", spec.sky(x))
        y_c .*= spec.sky(x)
    end

    if out == "old all"
        return x, y_c

    elseif out in ["all", "binned", "init"]
        if binned
            inds = searchsortedlast.(Ref(x), spec.bins[spec.bin_mask])
            binned = zero(spec.x[spec.mask])
            sind = searchsortedlast.(Ref(spec.bins[spec.bin_mask]), spec.x[spec.mask])
            for (l, k) in enumerate(sind)

                for i in inds[k]:inds[k+1]-1
                    binned[l] += (x[i+1] - x[i]) * (2 - y_c[i+1] - y_c[i])
                end

                binned[l] = 1 - binned[l] / (x[inds[k+1]] - x[inds[k]]) / 2
            end
        else
            binned = linear_interpolation(x, y_c, extrapolation_bc=Flat())[spec.x[spec.mask]]
        end

        if out == "all"
            return pylist(x), pylist(y_c), pylist(spec.x[spec.mask]), pylist(binned)
        elseif out == "binned"
            return binned
        elseif out == "init"
            return y_c[spec.mask]
        end

    end
end


function fitLM(spec, p_pars, add; tieds=Dict(), opts=Dict(), blindMode=false, method="LsqFit.lmfit", maxiter=50,
               grid_type="minimized", grid_num=1, binned=true, telluric=false, tau_limit=0.001, accuracy=0.1, toll=1e-4)

    println(toll)
    function cost(p)
        i = 1
        for (k, v) in pars
            if v.fit == 1
                pars[k].val = p[i]
                i += 1
            end
        end

        update_pars(pars, spec, add)

        res = Vector{Float64}()
        for s in spec
            if sum(s.mask) > 0
                #println(calc_spectrum(s, pars, out="binned", regular=regular, telluric=telluric, tau_limit=tau_limit, accuracy=accuracy))
                append!(res, (calc_spectrum(s, pars, out="binned", grid_type=grid_type, grid_num=grid_num, binned=binned, telluric=telluric, tau_limit=tau_limit, accuracy=accuracy) .- s.y[s.mask]) ./ s.unc[s.mask])
            end
        end

        # add constraints to the fit set by opts parameter

        # constraints for H2 on increasing b parameter with J level increase
		if (haskey(opts, "b_increase"))
		    if (opts["b_increase"] == true)
		    retval = 0
                for (k, v) in pars
                    if occursin("H2j", k) & occursin("b_", k) & (strip(v.addinfo) == "")
                        if ~occursin("v", k)
                            j = parse(Int64, k[8:end])
                        else
                            j = parse(Int64, k[8:findfirst('v', k)-1])
                        end
                        for (k1, v1) in pars
                            if occursin(k[1:7], k1) & ~occursin(k, k1) & (strip(v1.addinfo) == "")
                                if ~occursin("v", k1)
                                    j1 = parse(Int64, k1[8:end])
                                else
                                    j1 = parse(Int64, k1[8:findfirst('v', k1)-1])
                                end
                                #j, j1 = parse(Int64, k[8:end]), parse(Int64, k1[8:end])
                                if (~occursin("v", k) & ~occursin("v", k1)) || (occursin("v", k) & occursin("v", k1))
                                    #println(j, " ", j1, " ", (~occursin("v", k) & ~occursin("v", k1)), " ", (occursin("v", k) & occursin("v", k1)))
                                    x = sign(j - j1) * (v.val / v1.val - 1) * 10
                                    retval -= (x < 0 ? x : 0)
                                end
                            end
                        end
                    end
                end
            end
			#println("b_incr ", retval)
			append!(res, retval)
		end

		# constraints for H2 on on excitation diagram to be gradually increasing with J
		if (haskey(opts, "H2_excitation"))
		    if (opts["H2_excitation"] == true)
                retval = 0
                T = Dict()
                E = [[0, 118.5, 354.35, 705.54, 1168.78, 1740.21, 2414.76, 3187.57, 4051.73, 5001.97, 6030.81, 7132.03, 8298.61, 9523.82, 10800.6, 12123.66, 13485.56] * 1.42879,
                     [4161.14, 4273.75, 4497.82, 4831.41, 5271.36, 5813.95, 6454.28, 7187.44, 8007.77, 8908.28, 9883.79, 10927.12, 12031.44, 13191.06] * 1.42879
                    ]  #Energy difference in K
                g = [(2 * level + 1) * ((level % 2) * 2 + 1) for level in 0:15]  #statweights
                nu = append!([0], unique([parse(Int, k[findfirst('v', k)+1]) for (k,v) in pars if occursin("v", k)]))
                #println(nu)
                for n in nu
                    for (k, v) in pars
                        if occursin("H2j", k) & occursin("N_", k)
                            nextlev = ""
                            #println(k, " ", occursin("v", k), " ", findfirst('v', k))
                            if ~occursin("v", k) & (n == 0)
                                j = parse(Int64, k[8:end])
                                nextlev = k[1:7] * string(j+2)
                            elseif occursin("v", k)
                                if n == parse(Int, k[findfirst('v', k)+1])
                                    j = parse(Int64, k[8:findfirst('v', k)-1])
                                    nextlev = k[1:findfirst('j', k)] * string(parse(Int, k[findfirst('j', k)+1:findfirst('v', k)-1]) + 2) * k[findfirst('v', k):end]
                                end
                            end
                            if haskey(pars, nextlev)
                                #println(j, " ", nextlev, " ", k[3])
                                #println(g[j+1], " ", E[n+1][j+1], " ", g[j+3], " ", E[n+1][j+3])
                                if ~haskey(T, k[3])
                                    T[k[3]] = Dict()
                                end
                                T[k[3]][j] = (E[n+1][j+3] - E[n+1][j+1]) / log(10^v.val / 10^pars[nextlev].val * g[j+3] / g[j+1])
                                #println(j, " ", v.val, " ", E[n+1][j+1], " ", g[j+1], " ", T[k[3]][j])
                            end
                        end
                    end
                    #println(n, " ", T)
                    op = 1
                    for (k, d) in T
                        #println(n, " ", k, " ", d)
                        for (k, v) in d
                            if haskey(d, k + op)
                                #println(k, " ", v, " ", d[k+op], " ", (d[k+op] - v < 0 ? (d[k+op] - v) / 50 : 0) ^ 2 + (v < 0 ? v / 100 : 0) ^ 2, " ", (d[k+op] - v < 0 ? (d[k+op] / v - 1) * 10 : 0) ^ 2 + (v < 0 ? v / 100 : 0) ^ 2)
                                retval -= (d[k+op] - v < 0 ? (d[k+op] / v - 1) * 100 : 0) + (v < 0 ? v / 10 : 0)
                            end
                        end
                    end
                end
            end
			println("H2_exc ", retval)
			append!(res, retval)
		end

        return res
    end

    opts = pyconvert(Dict, opts)

    pars = make_pars(p_pars, tieds=tieds, z_ref=true)

    params = [p.val for (k, p) in pars if p.fit == true]
    lower = [p.min for (k, p) in pars if p.fit == true]
    upper = [p.max for (k, p) in pars if p.fit == true]

    if method == "LsqFit.lmfit"
        fit = LsqFit.lmfit(cost, params, Float64[], maxIter=maxiter, lower=lower, upper=upper, show_trace=true, x_tol=toll, g_tol=toll)
        converged = fit.converged
    elseif method == "lmfit"
        iter = true
        while iter
            iter = false
            fit = optimize(cost, params, LevenbergMarquardt(), iterations=maxiter, lower=lower, upper=upper, show_trace=true, x_tol=toll, f_tol=toll)
            for (i, p) in enumerate([p.name for (k, p) in pars if p.fit == true])
                if (fit.minimizer[i] == pars[p].min) || (fit.minimizer[i] == pars[p].max)
                    pars[p].fit = false
                    pars[p].val = fit.minimizer[i]
                    iter = true
                end
            end
            params = [p.val for (k, p) in pars if p.fit == true]
            lower = [p.min for (k, p) in pars if p.fit == true]
            upper = [p.max for (k, p) in pars if p.fit == true]
        end
        converged = fit.converged
        fit = LsqFit.lmfit(cost, copy(fit.minimizer), Float64[]; maxIter=1, lower=lower, upper=upper, show_trace=true, x_tol=toll, g_tol=toll)
    end

    A = fit.jacobian' * fit.jacobian
    println(isfinite(cond(A)))
    sigma = zeros(size([p.val for (k, p) in pars if p.vary == true]))
    try
        if isfinite(cond(A))
            sigma = stderror(fit)
        else
            println((sum(A, dims=1) .!= 0)[:])
            B = stack(filter(!iszero,eachcol(permutedims(stack(filter(!iszero,eachrow(A)))))))
            sigma = zeros(size(A)[1])
            sigma[(sum(A, dims=1) .!= 0)[:]] = sqrt.(abs.(diag(inv(B' * B))))
        end
    catch
        println("The was a problem during calculation of the covariance matrix")
    end
    #param, sigma = copy(fit.param), stderror(fit)
    i = 1
    for (k, p) in pars
        if p.fit
            pars[k].val = fit.param[i]
            pars[k].unc = sigma[i]
            if startswith(p.name, "z_")
                pars[k].val = z_to_v(v=pars[k].val, z_ref=p.ref)
                pars[k].unc = z_to_v(v=pars[k].unc, z_ref=p.ref) - p.ref
            end
            i += 1
        end
        if p.vary
            if !blindMode
                println(k, ": ", p.val, " ± ", p.unc)
            end
        end
    end

    return [p.val for (k, p) in pars if p.vary == true], [p.unc for (k, p) in pars if p.vary == true], pybool(converged)
end