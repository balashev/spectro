using Distributed

if nprocs() > 1
    rmprocs(2:nprocs())
end

nthreads = 1
open("config/options.ini") do f
    for l in eachline(f)
        if occursin("MCMC_threads", l)
            global nthreads = parse(Int64, split(l)[3])
        end
    end
end

if nthreads > 1
    addprocs(nthreads - 1)
end

println("procs: ", nprocs())

@everywhere using SpecialFunctions
@everywhere include("profiles.jl")

function fitMCMC(spec, pars; nwalkers=100, nsteps=1000, nthreads=1, init=nothing)

    #COUNTERS["num"] = nwalkers

    params = [p.val for p in pars if p.vary == 1]

    numdims = size(params)[1]
    thinning = 10

    if init == nothing
        init = Matrix{Float64}(undef, numdims, nwalkers)
        i = 1
        for p in pars
            if p.vary == 1
                init[i,:] = p.val .+ randn(nwalkers) .* p.step
                i += 1
            end
        end
    end

    lnlike = p->begin
        k = 1
        for i in 1:size(pars)[1]
            if pars[i].vary == 1
                if p[k] < pars[i].min
                    p[k] = pars[i].min
                elseif p[k] > pars[i].max
                    p[k] = pars[i].max
                end
                pars[i].val = p[k]
                #println(pars[i].name, " ", p[k])
                k += 1
            end
        end

        retval = 0
        for s in spec
            retval -= .5 * sum(((calc_spectrum(s, pars, out="init") - s.y[s.mask]) ./ s.unc[s.mask]) .^ 2)
        end
        return retval
    end

    chain, llhoodvals = AffineInvariantMCMC.sample(lnlike, nwalkers, init, nsteps, 1)

end