using Distributed
println(ARGS)
if size(ARGS)[1] >= 2
    @everywhere include("MCMC.jl")
    @everywhere include("profiles.jl")

    if endswith(strip(ARGS[end]), "plot")
        plotChain(ARGS[1])
    else
        cont = endswith(strip(ARGS[end]), "ontinue")
        if size(ARGS)[1] - cont == 3
            nstep = ARGS[3]
        else
            nstep = nothing
        end
        runMCMC(ARGS[1], ARGS[2], nstep=nstep, cont=cont)
    end
else
    println("provide an input in format: julia runMCMC.jl <filename> <number of treads>")
    println("                        or: julia runMCMC.jl <filename> <number of treads> <number of steps>")
    println("                        or: julia runMCMC.jl <filename> <number of treads> <number of steps> continue")
end