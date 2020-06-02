using Distributed
import RobustPmap

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

function fitMCMC(spec, par; nwalkers=100, nsteps=1000, nthreads=1, init=nothing)

    #COUNTERS["num"] = nwalkers

    pars = make_pars(par)
    params = [p.val for (k, p) in pars if p.vary == 1]

    numdims = size(params)[1]
    thinning = 10

    #init2 = Matrix{Float64}(undef, numdims, nwalkers)
    #if 1==1 #init == nothing
    #    i = 1
    #    for p in pars
    #        if p.vary == 1
    #            init2[i,:] = p.val .+ randn(nwalkers) .* p.step
    #            i += 1
    #        end
    #    end
    #end
    #println(size(init2), init2[3])

    lnlike = p->begin
        #println(p)
        i = 1
        for (k, v) in pars
            if v.vary == 1
                if p[i] < v.min
                    p[i] = v.min
                elseif p[i] > v.max
                    p[i] = v.max
                end
                pars[k].val = p[i]
                i += 1
            end
        end

        retval = 0

        #if H2_cond == true
        for (k, v) in pars
            if occursin("H2j", k) & occursin("b_", k)
                for (k1, v1) in pars
                    if occursin(k[1:7], k1) & ~occursin(k, k1)
                        j, j1 = parse(Int64, k[8:end]), parse(Int64, k1[8:end])
                        x = sign(j - j1) * (v.val / v1.val - 1) * 10
                        retval -= (x < 0 ? x : 0) ^ 2
                        #println(j, " ", j1, " ", v.val, " ", v1.val, " ", x, " ", retval)
                    end
                end
            end
        end
        #end

        for s in spec
            if sum(s.mask) > 0
                retval -= .5 * sum(((calc_spectrum(s, pars, out="init") - s.y[s.mask]) ./ s.unc[s.mask]) .^ 2)
            end
        end
        #println(retval)
        return retval
    end

    bounds = hcat([p.min for (k, p) in pars if p.vary], [p.max for (k, p) in pars if p.vary])
    chain, llhoodvals = sample(lnlike, nwalkers, init, nsteps, 1, bounds)

end

function check_pars(proposal, pars)
    return all([proposal[i] > p.min && proposal[i] < p.max for (i, p) in enumerate([p for p in pars if p.vary])])
end

function sample(llhood::Function, nwalkers::Int, x0::Array, nsteps::Integer, thinning::Integer, bounds::Array; a::Number=2.)
    """
    This function is modified version of AffineInvariantMCMC by MADS (see copyright information in initial module)
    """
	@assert length(size(x0)) == 2

	x = copy(x0)
	chain = Array{Float64}(undef, size(x0, 1), nwalkers, div(nsteps, thinning))
	llhoodvals = Array{Float64}(undef, nwalkers, div(nsteps, thinning))
	lastllhoodvals = RobustPmap.rpmap(llhood, map(i->x[:, i], 1:size(x, 2)))
	chain[:, :, 1] = x0
	llhoodvals[:, 1] = lastllhoodvals
	for i = 2:nsteps
		println(i)
		for ensembles in [(1:div(nwalkers, 2), div(nwalkers, 2) + 1:nwalkers), (div(nwalkers, 2) + 1:nwalkers, 1:div(nwalkers, 2))]
			active, inactive = ensembles
			zs = map(u->((a - 1) * u + 1)^2 / a, rand(length(active)))
			proposals = map(i-> min.(max.(zs[i] * x[:, active[i]] + (1 - zs[i]) * x[:, rand(inactive)], bounds[:,1]), bounds[:,2]), 1:length(active))
            newllhoods = RobustPmap.rpmap(llhood, proposals)
			for (j, walkernum) in enumerate(active)
				z = zs[j]
				newllhood = newllhoods[j]
				proposal = proposals[j]
				logratio = (size(x, 1) - 1) * log(z) + newllhood - lastllhoodvals[walkernum]
				if log(rand()) < logratio
					lastllhoodvals[walkernum] = newllhood
					x[:, walkernum] = proposal
			    else
    			    x[:, walkernum] = chain[:, walkernum, i-1]
                end
				if i % thinning == 0
					chain[:, walkernum, div(i, thinning)] = x[:, walkernum]
					llhoodvals[walkernum, div(i, thinning)] = lastllhoodvals[walkernum]
				end
			end
		end
	end
	return chain, llhoodvals

end
