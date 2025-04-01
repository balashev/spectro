using Distributed
println(ARGS)
if size(ARGS)[1] >= 3 || endswith(strip(ARGS[end]), "plot")
    @everywhere include("MCMC.jl")
    @everywhere include("profiles.jl")

    if endswith(strip(ARGS[end]), "plot")
        plotChain(ARGS[1])
    else
        sample_type = "Affine"
        if Bool(sum([occursin("ESS", a) for a in ARGS]))
            sampler_type = "ESS"
        end
        if Bool(sum([occursin("Affine", a) for a in ARGS]))
            sampler_type = "Affine"
        end
        runMCMC(ARGS[1], parse(Int, ARGS[2]), nstep=parse(Int, ARGS[3]), cont=Bool(sum([occursin("ontinue", a) for a in ARGS])), last=Bool(sum([occursin("last", a) for a in ARGS])), sampler_type=sampler_type)
    end
else
    println("provide an input in format: julia runMCMC.jl <filename> plot")
    println("                        or: julia runMCMC.jl <filename> <number of treads> <number of steps>")
    println("                        or: julia runMCMC.jl <filename> <number of treads> <number of steps> continue")
    println("                        or: julia runMCMC.jl <filename> <number of treads> <number of steps> continue from last")
end