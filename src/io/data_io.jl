using CSV
using DataFrames

"""
Save dataset matrix [tau, T, A] to CSV.
"""
function save_dataset_csv(path::AbstractString, data::AbstractMatrix{<:Real})
    @assert size(data, 2) == 3 "Dataset must have columns [tau, T, A]."
    df = DataFrame(tau=data[:, 1], T=data[:, 2], A=data[:, 3])
    CSV.write(path, df)
    return path
end

"""
Load dataset matrix [tau, T, A] from CSV.
"""
function load_dataset_csv(path::AbstractString)
    df = CSV.read(path, DataFrame)
    @assert all([:tau, :T, :A] .∈ Ref(names(df))) "CSV must contain tau, T, A columns."
    return Matrix{Float64}(select(df, :tau, :T, :A))
end
