using CSV
using DataFrames
using HDF5
using Serialization

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
    cols = names(df)

    name_map = Dict(lowercase(strip(String(c))) => c for c in cols)
    tau_col = get(name_map, "tau", nothing)
    t_col = get(name_map, "t", nothing)
    a_col = get(name_map, "a", nothing)

    if tau_col === nothing || t_col === nothing || a_col === nothing
        @assert size(df, 2) >= 3 "CSV must contain tau, T, A columns or at least three columns."
        return Matrix{Float64}(df[:, 1:3])
    end

    return Matrix{Float64}(select(df, tau_col, t_col, a_col))
end

"""
Save dataset matrix [tau, T, A] to HDF5.
"""
function save_dataset_h5(path::AbstractString, data::AbstractMatrix{<:Real})
    @assert size(data, 2) == 3 "Dataset must have columns [tau, T, A]."
    h5open(path, "w") do f
        f["dataset"] = Matrix{Float64}(data)
    end
    return path
end

"""
Load dataset matrix [tau, T, A] from HDF5.
"""
function load_dataset_h5(path::AbstractString)
    h5open(path, "r") do f
        @assert haskey(f, "dataset") "HDF5 file must contain /dataset"
        data = read(f["dataset"])
        @assert size(data, 2) == 3 "Dataset must have columns [tau, T, A]."
        return Matrix{Float64}(data)
    end
end

"""
Save dataset using native Julia serialization (.jls).
"""
function save_dataset_jls(path::AbstractString, data::AbstractMatrix{<:Real})
    @assert size(data, 2) == 3 "Dataset must have columns [tau, T, A]."
    open(path, "w") do io
        serialize(io, Matrix{Float64}(data))
    end
    return path
end

"""
Load dataset from native Julia serialization (.jls).
"""
function load_dataset_jls(path::AbstractString)
    open(path, "r") do io
        data = deserialize(io)
        @assert data isa AbstractMatrix "Serialized object must be a matrix."
        @assert size(data, 2) == 3 "Dataset must have columns [tau, T, A]."
        return Matrix{Float64}(data)
    end
end

"""
Save dataset by extension: .csv, .h5/.hdf5, .jls
"""
function save_dataset(path::AbstractString, data::AbstractMatrix{<:Real})
    lower = lowercase(path)
    if endswith(lower, ".csv")
        return save_dataset_csv(path, data)
    elseif endswith(lower, ".h5") || endswith(lower, ".hdf5")
        return save_dataset_h5(path, data)
    elseif endswith(lower, ".jls")
        return save_dataset_jls(path, data)
    else
        error("Unsupported format. Use .csv, .h5/.hdf5, or .jls")
    end
end

"""
Load dataset by extension: .csv, .h5/.hdf5, .jls
"""
function load_dataset(path::AbstractString)
    lower = lowercase(path)
    if endswith(lower, ".csv")
        return load_dataset_csv(path)
    elseif endswith(lower, ".h5") || endswith(lower, ".hdf5")
        return load_dataset_h5(path)
    elseif endswith(lower, ".jls")
        return load_dataset_jls(path)
    else
        error("Unsupported format. Use .csv, .h5/.hdf5, or .jls")
    end
end
