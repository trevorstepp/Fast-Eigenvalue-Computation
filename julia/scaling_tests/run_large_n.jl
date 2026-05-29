include("../BlockEig.jl")
using .BlockEig
using Printf
using Statistics
using LinearAlgebra
using BenchmarkTools

#BenchmarkTools.DEFAULT_PARAMETERS.samples = 5
#BenchmarkTools.DEFAULT_PARAMETERS.evals = 1

#const N_VALUES = [100, 250, 500, 750, 1000, 1500, 2000]
const N_VALUES = [100, 250, 500, 750, 1000, 1500, 2000, 3000, 4500, 6000, 10000]
const K_VAL = 3

function run_experiments()
    block_time = Float64[]

    for n in N_VALUES
        println("\nRunning n = $n")
        # build matrix
        M = build_block_matrix(K_VAL, n; seed=0)

        # timing
        t_block = @benchmark eig_KxK_diagblocks($K_VAL, $n, $M)
        push!(block_time, median(t_block).time / 1e9)
    end

    return block_time
end

if abspath(PROGRAM_FILE) == @__FILE__
    block_time = run_experiments()

    # save results for plotting
    using CSV, DataFrames
    df = DataFrame(
        K = K_VAL,
        n = N_VALUES,
        block_time = block_time,
    )
    CSV.write("julia_large_n_timings.csv", df)
end