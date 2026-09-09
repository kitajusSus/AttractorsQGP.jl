"""
    Master Driver Script: Run all LPCA publication experiments
    Executes:
    1. 01_k_dependency.jl
    2. 02_normalization_tests.jl
    3. 03_coordinate_invariance.jl
    4. 04_dimension_statistics.jl
"""

if !isdefined(Main, :PublicationLPCA)
    include(joinpath(@__DIR__, "..", "src", "PublicationLPCA.jl"))
end
using .PublicationLPCA

include("01_k_dependency.jl")
include("02_normalization_tests.jl")
include("03_coordinate_invariance.jl")
include("04_dimension_statistics.jl")
include("05_phase_space_grids.jl")
include("06_parameterization_focus_analysis.jl")

function run_all_lpca_publication_experiments()
    println("=================================================================")
    println("      STARTING FULL LOCAL PCA SYSTEMATIC STUDY PIPELINE          ")
    println("=================================================================")

    start_time = time()

    println("\n>>> [STEP 1/6] Running K-Dependency Analysis...")
    run_k_dependency_experiment()

    println("\n>>> [STEP 2/6] Running Data Normalization Comparisons...")
    run_normalization_experiment()

    println("\n>>> [STEP 3/6] Running Coordinate & Scale Invariance Tests...")
    run_coordinate_invariance_experiment()

    println("\n>>> [STEP 4/6] Running Dimension Statistics & dims() Evaluation...")
    run_dimension_statistics_experiment()

    println("\n>>> [STEP 5/6] Generating 2D & 3D Phase Space Evolution Grids...")
    run_phase_space_grids_experiment()

    println("\n>>> [STEP 6/6] Running Parameterization Focus Analysis (Dense K Sweep)...")
    run_parameterization_focus_experiment()

    elapsed = time() - start_time
    println("\n=================================================================")
    @printf("  ALL EXPERIMENTS COMPLETED SUCCESSFULLY IN %.2f SECONDS!\n", elapsed)
    println("  Figures saved to: publication_lpca/plots/")
    println("  Data tables saved to: publication_lpca/results/")
    println("=================================================================")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_all_lpca_publication_experiments()
end
