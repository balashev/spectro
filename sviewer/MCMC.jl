using DelimitedFiles
using Distributed
@everywhere using PyCall
using Random
#import RobustPmap

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

function fitMCMC(spec, ppar, add; prior=nothing, nwalkers=100, nsteps=1000, nthreads=1, init=nothing, opts=0)

    #COUNTERS["num"] = nwalkers

	println(init)

    pars = make_pars(ppar)
    priors = make_priors(prior)
    params = [p.val for (k, p) in pars if p.vary == 1]

    numdims = size(params)[1]
    thinning = 10

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

        update_pars(pars, spec, add)

		retval = 0

        if priors != nothing
            for (k, p) in priors
                #println(p.name, " ", pars[p.name].val, " ", use_prior(p, pars[p.name].val))
                retval -= use_prior(p, pars[p.name].val)
            end
        end

        for s in spec
            if sum(s.mask) > 0
    			model = calc_spectrum(s, pars, out="init")
                retval -= .5 * sum(((model .- s.y[s.mask]) ./ s.unc[s.mask]) .^ 2)

				if opts["hier_continuum"] == true
    				for cont in s.cont
						mask = (s.x[s.mask] .> cont.left) .& (s.x[s.mask] .< cont.right)
						c = parse(Float64, split(pars[cont.c[1]].addinfo, "_")[4])
						A = sum((model[mask] .* c ./ s.unc[s.mask][mask]) .^ 2) + 1 / pars["hcont"].val ^ 2
						B = sum(model[mask] .* c .* (model[mask] .- s.y[s.mask][mask]) ./ s.unc[s.mask][mask] .^ 2)
						retval -= .5 * (log(A * pars["hcont"].val ^ 2) - B ^ 2 / A)
					end
				end
            end
        end

        # add constraints to the fit set by opts parameter

		# constraints for H2 on increasing b parameter with J level increase
        if opts["b_increase"] == true
            for (k, v) in pars
                if occursin("H2j", k) & occursin("b_", k)
                    for (k1, v1) in pars
                        if occursin(k[1:7], k1) & ~occursin(k, k1)
                            j, j1 = parse(Int64, k[8:end]), parse(Int64, k1[8:end])
                            x = sign(j - j1) * (v.val / v1.val - 1) * 10
                            retval -= (x < 0 ? x : 0) ^ 2
                        end
                    end
                end
            end
        end

        # constraints for H2 on on excitation diagram to be gradually increasing with J
        if opts["H2_excitation"] == true
            T = Dict()
            E = [170.5, 339.35, 505.31, 666.53, 822.20, 970.58, 1111.96, 1243.40, 1367.25, 1480.35] #Energy difference in K
            g = [(2 * level + 1) * ((level % 2) * 2 + 1) for level in 0:11]  #statweights
            for (k, v) in pars
                if occursin("H2j", k) & occursin("N_", k)
                    j = parse(Int64, k[8:end])
                    if haskey(pars, k[1:7] * string(j+1))
                        if ~haskey(T, k[3])
                            T[k[3]] = Dict()
                        end
                        #println(j, " ", v.val, " ", E[j+1], " ", g[j+1])
                        T[k[3]][j] = E[j+1] / log(10^v.val / 10^pars[k[1:7] * string(j+1)].val * g[j+2] / g[j+1])
                    end
                end
            end
            for (k, d) in T
                #println(k)
                for (k, v) in d
                    if haskey(d, k+1)
                        #println(k, " ", v, " ", d[k+1])
                        retval -= (d[k+1] - v < 0 ? (d[k+1] - v) / 50 : 0) ^ 2 + (v < 0 ? v / 100 : 0) ^ 2
                    end
                end
            end
            #println(T)
        end

        return retval
    end

    bounds = hcat([p.min for (k, p) in pars if p.vary], [p.max for (k, p) in pars if p.vary])
	#println(bounds)
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
	#lastllhoodvals = RobustPmap.rpmap(llhood, map(i->x[:, i], 1:size(x, 2)))
	lastllhoodvals = pmap(llhood, map(i->x[:, i], 1:size(x, 2)))
	chain[:, :, 1] = x0
	llhoodvals[:, 1] = lastllhoodvals
	for i = 2:nsteps
		println(i)
		for ensembles in [(1:div(nwalkers, 2), div(nwalkers, 2) + 1:nwalkers), (div(nwalkers, 2) + 1:nwalkers, 1:div(nwalkers, 2))]
			active, inactive = ensembles
			zs = map(u->((a - 1) * u + 1)^2 / a, rand(length(active)))
			proposals = map(i-> min.(max.(zs[i] * x[:, active[i]] + (1 - zs[i]) * x[:, rand(inactive)], bounds[:,1]), bounds[:,2]), 1:length(active))
            #newllhoods = RobustPmap.rpmap(llhood, proposals)
			newllhoods = pmap(llhood, proposals)
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
		open("output/mcmc_last.dat", "w") do io
			writedlm(io, chain[:, :, i], " ")
		end
	end
	return chain, llhoodvals
end
