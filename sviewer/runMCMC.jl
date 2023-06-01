using Distributed
println(ARGS)
if size(ARGS)[1] >= 3 || endswith(strip(ARGS[end]), "plot")
    @everywhere include("MCMC.jl")
    @everywhere include("profiles.jl")

    if endswith(strip(ARGS[end]), "plot")
        plotChain(ARGS[1])
    else
        runMCMC(ARGS[1], ARGS[2], nstep=ARGS[3], cont=Bool(sum([occursin("ontinue", a) for a in ARGS])), last=Bool(sum([occursin("last", a) for a in ARGS])))
    end
else
    println("provide an input in format: julia runMCMC.jl <filename> plot")
    println("                        or: julia runMCMC.jl <filename> <number of treads> <number of steps>")
    println("                        or: julia runMCMC.jl <filename> <number of treads> <number of steps> continue")
    println("                        or: julia runMCMC.jl <filename> <number of treads> <number of steps> continue from last")
end