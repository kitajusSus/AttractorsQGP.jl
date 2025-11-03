
using CSV, DataFrames, HDF5, Random, Dates


const N_POINTS = 10000
const T_RANGE = (200.0, 1800.0)
const A_RANGE = (-12.0, 25.0)
const output_filename_h = "initial_conditions.h5"
const output_filename_csv = "initial_conditions.csv"
const rng = Xoshiro(5)



function generate_data(rng, n_points, t_range, a_range)
    initial_T = t_range[1] .+ (t_range[2] - t_range[1]) .* rand(rng, n_points)
    initial_A = a_range[1] .+ (a_range[2] - a_range[1]) .* rand(rng, n_points)
    df = DataFrame(
        Run_ID = 1:n_points,
        T_0 = initial_T,
        A_0 = initial_A
    )
    return df
end

function save_to_csv(df::DataFrame, filename::String)
    println("Zapisywanie danych do pliku CSV: $filename")
    CSV.write(filename, df)
    println("Zapisano pomyślnie.")
end

function save_to_hdf5(df::DataFrame, filename::String, metadata::Dict)
    println("Zapisywanie danych do pliku HDF5: $filename")
    h5open(filename, "w") do file
        g = create_group(file, "initial_conditions")

        g["Run_ID"] = df.Run_ID
        g["T_0"] = df.T_0
        g["A_0"] = df.A_0

        a = attrs(g)
        for (key, value) in metadata
            a[string(key)] = value
        end
    end
    println("Zapisano pomyślnie.")
end

function main()
    metadata = Dict(
        "description" => "Zestaw losowych warunków początkowych.",
        "seed" => 5,
        "n_points" => N_POINTS,
        "timestamp" => string(now())
    )
    df = generate_data(rng, N_POINTS, T_RANGE, A_RANGE)
    save_to_csv(df, output_filename_csv)
    save_to_hdf5(df, output_filename_h, metadata)

end


main()
