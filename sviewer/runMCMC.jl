using Distributed
if size(ARGS)[1] == 2
    @everywhere include("MCMC.jl")
    @everywhere include("profiles.jl")
    runMCMC(ARGS[1], ARGS[2])
else
    println("provide an input in format: julia runMCMC.jl <filename> <number of treads>")
end
