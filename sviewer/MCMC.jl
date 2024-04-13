using ClusterManagers
using DelimitedFiles
using Distributed
using Measures
#using Plots
using Random
using Serialization
using Statistics
@everywhere using SpecialFunctions
@everywhere using Statistics
@everywhere include("profiles.jl")
#@everywhere using PyCall
@everywhere using Combinatorics
@everywhere using AdvancedHMC, ForwardDiff

function initJulia(filename, spec, pars, add, parnames; sampler="Affine", prior=nothing, nwalkers=100, nsteps=1000, nthreads=1, thinning=1, init=nothing, opts=0)
	serialize(filename, [spec, pars, add, parnames, sampler, prior, nwalkers, nsteps, thinning, init, opts])
end

function initJulia2(filename, self; fit=nothing, fit_list=nothing, parnames=nothing, tieds=nothing, sampler="Affine", prior=nothing, nwalkers=100, nsteps=1000, nthreads=1, thinning=1, init=nothing, opts=0)
    pars = make_pars(fit_list, tieds=tieds)
    add = prepare_add(fit, pars)
    spec = prepare(self, pars, add)
    priors = make_priors(prior)
    serialize(filename, [spec, pars, add, parnames, sampler, priors, nwalkers, nsteps, thinning, init, opts])
end

function runMCMC(filename, nthreads; nstep=nothing, cont=false, last=false, sampler_type=nothing)
    spec, pars, add, parnames, sampler, prior, nwalkers, nsteps, thinning, init, opts = deserialize(filename)
    if nstep != nothing
        nsteps = parse(Int, nstep)
    end
    if sampler_type != nothing
        sampler = sampler_type
    end
    println("sampler: ", sampler)
    println("steps to go: ", nsteps)
    println(cont, " ", last)
    if cont
        println("continue from the last step...")
        if last
            init = deserialize(replace(filename, ".spj" => ".spl"))
        else
            chain, llhoodvals = readMCMC(replace(filename, ".spj" => ".spr"))
            println(size(chain))
            init = chain[:, :, end]
        end
        println(size(init))
        #println(init)
    end

    println("size of init: ", size(init))
	#println(sampler, " ", parnames, " ", prior, " ", nwalkers, " ", nsteps, " ", thinning)
	chain, llhoodvals = fitMCMC(spec, pars, add, parnames, sampler=sampler, prior=prior, nwalkers=nwalkers, nsteps=nsteps, nthreads=parse(Int, nthreads), thinning=thinning, init=init, opts=opts, filename=filename)
	serialize(replace(filename, ".spj" => ".spr"), [parnames, chain, llhoodvals])
	#plotChain(filename)
end

function plotChain(filename) #(pars, chain, llhoodvals)
    spec, pars, add, parnames, sampler, prior, nwalkers, nsteps, thinning, init, opts = deserialize(filename)
    pars = make_pars(pars, parnames=parnames)
    chain, llhoodvals = readMCMC(replace(filename, ".spj" => ".spr"))
    println(size(chain))
    n = size(chain)[2]
    nrows, ncols = round(Int, sqrt(n)) + (n % (round(Int, sqrt(n))) > 0), n ÷ (round(Int, sqrt(n)) + 1) + (n % (round(Int, sqrt(n)) + 1) > 0)
    #println(nrows, " ", ncols)
    x = range(1, size(chain)[3])
    params = [p.name for (k, p) in pars if p.vary == 1]
    p = Vector{}()
    ind = rand(1:size(chain)[1])
    for (i, par) in enumerate(params)
        if i < n + 1
            row, col = i ÷ ncols + 1, i - (i ÷ ncols) * ncols
            mean = quantile(mapslices(x -> quantile(filter(!isnan, x), 0.5), chain[:, i, :], dims = 1)[1, :], 0.5)
            if startswith(par, "z")
                y = (chain[:, i, :] .- mean) ./ (1 + mean) .* 3e5
            else
                y = chain[:, i, :] ./ mean .- 1.0
            end
            ymin = mapslices(x -> quantile(filter(!isnan, x), 0.15), y, dims = 1)[1, :]
            ymax = mapslices(x -> quantile(filter(!isnan, x), 0.95), y, dims = 1)[1, :]
            #push!(p, plot(x, [ymin, ymax], lw=5, xtickfontsize=50, ytickfontsize=50, title=par, legend=false))
            push!(p, plot(x, ymin, fillrange=ymax, fillalpha=0.5, xtickfontsize=100, ytickfontsize=100, xlabel="iteration", label=par, legendfontsize=100))
            plot!(x, y[ind, :], lw=20, label=false)
        end
    end
    #println(p)
    plot(p..., layout=grid(ncols, nrows), size=(4000 * ncols, 2000 * nrows), dpi=20)
    savefig(replace(filename, ".spj" => ".png"))
    println("Summary figure is plotted in ", replace(filename, ".spj" => ".png"))

    p = Vector{}()
    for (i, par) in enumerate(params)
        if i < n + 1
            row, col = i ÷ ncols + 1, i - (i ÷ ncols) * ncols
            push!(p, scatter(chain[:, i, end], llhoodvals[:, end], xtickfontsize=100, ytickfontsize=100, label=par, ms=20, legendfontsize=30, bottom_margin=40mm))
        end
    end
    #println(p)
    plot(p..., layout=grid(ncols, nrows), size=(4000 * ncols, 2000 * nrows), dpi=20)
    savefig(replace(filename, ".spj" => "_lns.png"))
    println("Likelihoods are plotted in ", replace(filename, ".spj" => "_lns.png"))

    p = Vector{}()
    for (i, par) in enumerate(params)
        if i < n + 1
            row, col = i ÷ ncols + 1, i - (i ÷ ncols) * ncols
            push!(p, histogram(vec(chain[:, i, (end - div(end, 4)):end]), xtickfontsize=100, ytickfontsize=100, label=par, ms=20, legendfontsize=30, bottom_margin=40mm))
        end
    end
    #println(p)
    plot(p..., layout=grid(ncols, nrows), size=(4000 * ncols, 2000 * nrows), dpi=20)
    savefig(replace(filename, ".spj" => "_1d.png"))
    println("Marginalized distributions are plotted in ", replace(filename, ".spj" => "_1d.png"))
end

function readMCMC(filename)
	parnames, chain, llhoodvals = deserialize(filename)
	return chain, llhoodvals
end

function fitMCMC(spec, pars, add, parnames; sampler="Affine", prior=nothing, nwalkers=100, nsteps=1000, nthreads=1, thinning=1, init=nothing, opts=0, filename="mcmc")

    #COUNTERS["num"] = nwalkers

	#println("init: ", init)
	#println(parnames)
    pars = make_pars(pars, parnames=parnames)
	#println(pars)
	println("pars length: ", length(pars))
    priors = make_priors(prior)
	#println("priors: ", priors)
    params = [p.val for (k, p) in pars if p.vary == 1]
	#println(init)

	#lnlike = p->begin
	function lnlike(p)
		i = 1
		#println(pars)
		for (k, v) in pars
			if v.vary == 1
				#println(i, " ", p[i], " ", v.min)
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
				model = calc_spectrum(s, pars, out="binned")
				#println("model ", model)

				retval -= .5 * sum(((model .- s.y[s.mask]) ./ s.unc[s.mask]) .^ 2)

				#println(retval)
                #println(s.y[s.mask])
				#println(s.unc[s.mask])

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
                                retval -= (x < 0 ? x : 0) ^ 2
                            end
						end
					end
				end
			end
		end

		# constraints for H2 on on excitation diagram to be gradually increasing with J
		if opts["H2_excitation"] == true
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
                            #println(k, " ", v, " ", d[k+2], " ", (d[k+2] - v < 0 ? (d[k+2] - v) / 50 : 0) ^ 2 + (v < 0 ? v / 100 : 0) ^ 2, " ", (d[k+2] - v < 0 ? (d[k+2] / v - 1) * 10 : 0) ^ 2 + (v < 0 ? v / 100 : 0) ^ 2)
                            retval -= (d[k+op] - v < 0 ? (d[k+op] / v - 1) * 100 : 0) ^ 2 + (v < 0 ? v / 10 : 0) ^ 2
                        end
                    end
                end
            end
			#println(T)
		end

		return retval
	end

	if sampler in ["Affine", "ESS", "UltraNest"]
		if nprocs() > 1
			rmprocs(2:nprocs())
		end

		#nthreads = 1
		#open("config/options.ini") do f
		#	for l in eachline(f)
		#		if occursin("MCMC_threads", l)
		#			nthreads = parse(Int64, split(l)[3])
		#		end
		#	end
		#end

		if nthreads > 1
			#addprocs(nthreads - 1)
      		#try
			#    addprocs(SlurmManager(nthreads)) #, N = nthreads ÷ 96 + 1)
			#catch
			addprocs(nthreads - 1)
			#addprocs_slurm(nthreads, nodes=nthreads ÷ 96 + 1, exename="/home/balashev/julia-1.8.5/bin/julia")
		end

		@everywhere include("profiles.jl")

		println("procs: ", nprocs())
	end

	if sampler == "Affine"
		bounds = hcat([p.min for (k, p) in pars if p.vary], [p.max for (k, p) in pars if p.vary])
		#println(bounds)
		chain, llhoodvals = sampleAffine(lnlike, nwalkers, init, nsteps, thinning, bounds, filename=filename)

	elseif sampler == "ESS"
		bounds = hcat([p.min for (k, p) in pars if p.vary], [p.max for (k, p) in pars if p.vary])
		chain, llhoodvals = sampleESS(lnlike, nwalkers, init, nsteps, thinning, bounds)

	elseif sampler == "Hamiltonian"
		"""
		does not work yet
		"""
		#chain = sampleHMC(lnlike, init, nsteps)
		x0 = init[:,1]

		# Define a Hamiltonian system
		metric = DiagEuclideanMetric(size(x0)[1])
		hamiltonian = Hamiltonian(metric, lnlike, ForwardDiff)

		# Define a leapfrog solver, with initial step size chosen heuristically
		initial_ϵ = find_good_stepsize(hamiltonian, x0)
		integrator = Leapfrog(initial_ϵ)

		# Define an HMC sampler, with the following components
		#   - multinomial sampling scheme,
		#   - generalised No-U-Turn criteria, and
		#   - windowed adaption for step-size and diagonal mass matrix
		proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
		adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

		# Run the sampler to draw samples from the specified Gaussian, where
		#   - `samples` will store the samples
		#   - `stats` will store diagnostic statistics for each sample
		samples, stats = sampleAffine(hamiltonian, proposal, initial_θ, nsteps, adaptor, nsteps/2; progress=true)
		return samples, stats

	elseif sampler == "UltraNest"
		ultranest = pyimport("ultranest")

		mytransform = cube->begin
			params = copy(cube)
			hi, lo = [p.max for (k, p) in pars if p.vary], [p.min for (k, p) in pars if p.vary]
			for (i, c) in enumerate(cube)
				params[i] = c * (hi[i] - lo[i]) + lo[i]
			end
			return params
		end

		lnlike_vec = p-> begin
			return pmap(lnlike, mapslices(x->[x], p, dims=2)[:])
		end

		mytransform_vec = cube->begin
			return transpose(reduce(hcat, map(i->mytransform(cube[i, :]), 1:size(cube, 1))))
		end

		paramnames = [replace(p.name, "_"=>"") for (k, p) in pars if p.vary]
		sampler = ultranest.ReactiveNestedSampler(paramnames, lnlike_vec, transform=mytransform_vec, vectorized=true)
		results = sampler.run()
		print("result has these keys:", keys(results), "\n")

		println(results["samples"])
		println(results["maximum_likelihood"])

		sampler.print_results()
		#sampler.plot_run()
		#sampler.plot()
		#return results["samples"], nothing
		return results, nothing
	end

	rmprocs(workers())
	println(nprocs())

	return chain, llhoodvals
end

function check_pars(proposal, pars)
    return all([proposal[i] > p.min && proposal[i] < p.max for (i, p) in enumerate([p for p in pars if p.vary])])
end

function sampleAffine(llhood::Function, nwalkers::Int, x0::Array, nsteps::Integer, thinning::Integer, bounds::Array; a::Number=2., out::Number=1., filename::String="mcmc_last.spj")
    """
    Modified version of AffineInvariantMCMC by MADS (see copyright information in initial module)
    """
	@assert length(size(x0)) == 2
	print(filename)
	#println(thinning)
	#println("x0: ", x0)
	x = copy(x0)
	println(size(x))
	chain = Array{Float64}(undef, nwalkers, size(x0, 2), div(nsteps, thinning))
	llhoodvals = Array{Float64}(undef, nwalkers, div(nsteps, thinning))
	lastllhoodvals = pmap(llhood, map(i->x[i, :], 1:size(x, 1)))
	chain[:, :, 1] = x0
	llhoodvals[:, 1] = lastllhoodvals
	for i = 2:nsteps
		println(i)
		s = shuffle(collect(1:nwalkers))
		for ensembles in [(s[1:div(nwalkers, 2)], s[div(nwalkers, 2) + 1:nwalkers]), (s[div(nwalkers, 2) + 1:nwalkers], s[1:div(nwalkers, 2)])]
			active, inactive = ensembles
			zs = map(u->((a - 1) * u + 1)^2 / a, rand(length(active)))
			proposals = map(i-> min.(max.(zs[i] * x[active[i], :] + (1 - zs[i]) * x[rand(inactive), :], bounds[:,1]), bounds[:,2]), 1:length(active))
			newllhoods = pmap(llhood, proposals)
			for (j, walkernum) in enumerate(active)
				logratio = (size(x, 2) - 1) * log(zs[j]) + newllhoods[j] - lastllhoodvals[walkernum]
				if log(rand()) < logratio
					lastllhoodvals[walkernum] = newllhoods[j]
					x[walkernum, :] = proposals[j]
			    end
				if i % thinning == 0
					chain[walkernum, :, div(i, thinning)] = x[walkernum, :]
					llhoodvals[walkernum, div(i, thinning)] = lastllhoodvals[walkernum]
				end
			end
		end
		if (out > 0) && (minimum(lastllhoodvals) < 4 * median(lastllhoodvals) - 3 * maximum(lastllhoodvals))
 		    imin = findmin(lastllhoodvals)[2]
		    println(imin, " ", lastllhoodvals[imin], " ", median(lastllhoodvals))
		    ind = rand(1:length(lastllhoodvals))
            println(ind)
            x[imin, :], lastllhoodvals[imin] = x[ind, :], lastllhoodvals[ind]
        end
        if i % thinning == 0
            serialize(replace(filename, ".spj" => ".spl"), chain[:, :, div(i, thinning)])
        end
	end
	return chain, llhoodvals
end

function sampleESS(llhood::Function, nwalkers::Int, x0::Array, nsteps::Integer, thinning::Integer, bounds::Array; mu::Number=1.)
	"""
	Ensamble Slice Sampler. Rewritten based on zeus python package
	"""
	function get_directions(x, mu)
		nsamples = size(x, 1)
		inds = hcat(rand(collect(permutations((1:nsamples), 2)), nsamples)...)
		return 2. .* mu .* (x[inds[1,:], :] .- x[inds[2,:], :])
	end

	maxsteps, maxiters = 100, 100
	#println(bounds)
	@assert length(size(x0)) == 2
	ndims = size(x0, 2)
	x = copy(x0)
	chain = Array{Float64}(undef, nwalkers, ndims, div(nsteps, thinning))
	llhoodvals = Array{Float64}(undef, nwalkers, div(nsteps, thinning))
	lastllhoodvals = pmap(llhood, map(i->x[i, :], 1:nwalkers))
	chain[:, :, 1] = x0
	llhoodvals[:, 1] = lastllhoodvals
	ncalls = 0

	for i = 2:nsteps
		println(i)
		# Initialise number of Log prob calls
		nexp, ncon, ncall = 0, 0, 0

		s = shuffle(collect(1:nwalkers))
		for ensembles in [(s[1:div(nwalkers, 2)], s[div(nwalkers, 2) + 1:nwalkers]), (s[div(nwalkers, 2) + 1:nwalkers], s[1:div(nwalkers, 2)])]
			# Define active-inactive ensembles
			active, inactive = ensembles
			#println(active, " ", inactive)

			# Compute directions
			directions = get_directions(x[inactive, :], mu)
			#println("direct ", directions)

			# Get Z0 = LogP(x0)
			z0 = lastllhoodvals[active] .- randexp(div(nwalkers, 2))

			# Set Initial Interval Boundaries
			L = - rand(Float64, div(nwalkers, 2))
			R = L .+ 1.0
			#println(L, " ", R)

			# Parallel stepping-out
			J = rand(0:maxsteps-1, div(nwalkers, 2))
			K = (maxsteps .- 1) .- J
			#println(J, " ", K)

			# Stepping-out initialisation
			mask_J, Z_L, X_L = trues(div(nwalkers, 2)), Array{Float64}(undef, div(nwalkers, 2)), Array{Float64}(undef, div(nwalkers, 2), ndims)
			mask_K, Z_R, X_R = trues(div(nwalkers, 2)), Array{Float64}(undef, div(nwalkers, 2)), Array{Float64}(undef, div(nwalkers, 2), ndims)
			#println(mask_K, Z_R, X_R)

			cnt = 0
			# Stepping-Out procedure
			while (sum(mask_J) > 0) | (sum(mask_K) > 0)
				cnt = sum(mask_J) > 0 ? cnt + 1 : cnt
				cnt = sum(mask_K) > 0 ? cnt + 1 : cnt
				if cnt > maxiters
					throw(DomainError("Number of expansion exceed limit"))
				end

				mask_J[mask_J] = J[mask_J] .> 0
				mask_K[mask_K] = K[mask_K] .> 0
				#println(cnt, " mask ", mask_J, " ", mask_K)

				X_L[mask_J, :] = min.(max.(directions[mask_J, :] .* L[mask_J] .+ x[active, :][mask_J, :], transpose(bounds[:, 1])), transpose(bounds[:, 2]))
				X_R[mask_K, :] = min.(max.(directions[mask_K, :] .* R[mask_K] .+ x[active, :][mask_K, :], transpose(bounds[:, 1])), transpose(bounds[:, 2]))
				#println("X: ", X_L, " ", X_R)

				if sum(mask_J) + sum(mask_K) < 1
					cnt -= 1
				else
					Z_LR_masked = pmap(llhood, map(i->vcat(X_L[mask_J, :], X_R[mask_K, :])[i, :], 1:sum(mask_J)+sum(mask_K)))
					Z_L[mask_J] .= Z_LR_masked[begin:sum(mask_J)]
					Z_R[mask_K] .= Z_LR_masked[sum(mask_J)+1:end]
					ncall += sum(mask_J) + sum(mask_K)
				end

				m = z0[mask_J] .< Z_L[mask_J]
				mask_J[mask_J] .= m
				if sum(m) > 0
					L[mask_J] .-= 1
					J[mask_J] .-= 1
				end
				nexp += sum(m)
				m = z0[mask_K] .< Z_R[mask_K]
				mask_K[mask_K] .= m
				if sum(m) > 0
					R[mask_K] .+= 1
					K[mask_K] .-= 1
				end
				nexp += sum(m)
			end
			#println(X_L, " ", X_R)

			# Shrinking procedure
			Widths, z_prime, x_prime = Array{Float64}(undef, div(nwalkers, 2)), Array{Float64}(undef, div(nwalkers, 2)), Array{Float64}(undef, div(nwalkers, 2), ndims)
			mask = trues(div(nwalkers, 2))

			cnt = 0
			while sum(mask) > 0
				# Update Widths of intervals
				Widths[mask] = L[mask] .+ rand(Float64, sum(mask)) .* (R[mask] .- L[mask])

				# Compute New Positions
				x_prime[mask, :] .= min.(max.(directions[mask, :] .* Widths[mask] .+ x[active, :][mask, :], transpose(bounds[:, 1])), transpose(bounds[:, 2]))

				# Calculate LogP of New Positions
				z_prime[mask] = pmap(llhood, map(i->x_prime[mask, :][i, :], 1:sum(mask)))

				ncall += sum(mask)

				# Shrink slices
				mask[mask] .= z0[mask] .> z_prime[mask]
				#println(mask)
				#println(L, " ", R)
				for (j, w) in enumerate(Widths)
					if mask[j] == 1
						if Widths[j] < 0
							L[j] = Widths[j]
						else
							R[j] = Widths[j]
						end
					end
				end
				#println(L, " ", R)
				ncon += sum(mask)

				cnt += 1
				if cnt > maxiters
					throw(DomainError("Number of contractions exceeded maximum limit!"))
				end
			end
			x[active, :] = x_prime[:, :]
			lastllhoodvals[active] = z_prime[:]
			if i % thinning == 0
				chain[active, :, i] .= x_prime[:, :]
				llhoodvals[active, i] .= z_prime[:]
			end
		end
		#println(mu)
		nexp = max(1, nexp)
		mu *= 2.0 * nexp / (nexp + ncon)
		ncalls += ncall
	end
	return chain, llhoodvals
end


function sampleHMC(llhood::Function, x0::Array, nsteps::Integer)
    """
    Hamiltonian MCMC sampling using Turing.jl package
    """
	init = x0[:,1]
	#lastllhoodvals = RobustPmap.rpmap(llhood, map(i->x[:, i], 1:size(x, 2)))

	# Define a Hamiltonian system
	metric = DiagEuclideanMetric(size(init)[1])
	hamiltonian = Hamiltonian(metric, llhood, ForwardDiff)

	# Define a leapfrog solver, with initial step size chosen heuristically
	initial_ϵ = find_good_stepsize(hamiltonian, init)
	integrator = Leapfrog(initial_ϵ)

	# Define an HMC sampler, with the following components
	#   - multinomial sampling scheme,
	#   - generalised No-U-Turn criteria, and
	#   - windowed adaption for step-size and diagonal mass matrix
	proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
	adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

	# Run the sampler to draw samples from the specified Gaussian, where
	#   - `samples` will store the samples
	#   - `stats` will store diagnostic statistics for each sample
	samples, stats = sample(hamiltonian, proposal, initial_θ, nsteps, adaptor, nsteps/2; progress=true)
	return samples
end

@everywhere spectrum_comp(spectrum, s, x; comp=0) = p->spectrum(p, s, x, comp=comp)
@everywhere spectrum_cheb(spectrum, s, x) = p->spectrum(p, s, x, cheb=1)

function fit_disp(x, samples, spec, ppar, add; sys=1, tieds=Dict(), nums=100, nthreads=1)
	"""
	Calculate the dispersion of the fit
	"""
	pars = make_pars(ppar, tieds=tieds)
	inds = hcat(rand(1:size(samples, 1), nums), rand(1:size(samples, 2), nums))
	pp = map(i->samples[i[1], i[2], :], eachrow(inds))

    function spectrum(p, s, x; comp=0, cheb=0)
        i = 1
        for (k, v) in pars
            if v.vary == 1
                pars[k].val = p[i]
                i += 1
            end
        end

        update_pars(pars, spec, add)

        if cheb == 0
            w, f = calc_spectrum(s, pars, comp=comp)
        else
            w, f = x, correct_continuum(s.cont, pars, x)
        end
        if size(f)[1] > 2
            return LinearInterpolation(w, f, extrapolation_bc=Flat())(x)
        else
            return ones(size(x))
        end
	end

	if 1 == 1
		ntheards = 1
		if nprocs() > 1
			rmprocs(2:nprocs())
		end
		if nthreads > 1
			addprocs(nthreads - 1)
		end
		@everywhere include("profiles.jl")
		println("procs: ", nprocs())
	end

    fit = Any[]
    cheb = Any[]
    for (i, s) in enumerate(spec)
        push!(fit, pmap(spectrum_comp(spectrum, s, x[i], comp=0), pp))
        push!(cheb, pmap(spectrum_cheb(spectrum, s, x[i]), pp))
    end

    fit_comps = Any[]
    for (i, s) in enumerate(spec)
        push!(fit_comps, Any[])
        for k in 1:sys
            push!(fit_comps[i], reduce(vcat, transpose.(pmap(spectrum_comp(spectrum, s, x[i], comp=k), pp))))
        end
    end

    rmprocs(workers())

    return fit, fit_comps, cheb
end