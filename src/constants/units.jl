"""
Physical constants and simple unit conversion helpers.

The hydro equations are solved in natural units with temperature in `fm`.
Use conversion helpers when you need a different unit at I/O boundaries.
"""
const HBARC_MEV_FM = 197.3269804
const MEV_PER_FM = HBARC_MEV_FM
const FM_PER_MEV = inv(HBARC_MEV_FM)

"""
Convert temperature from fm to requested unit.

Supported units:
- `:fm` (default internal unit)
- `:MeV`
"""
function to_temperature_unit(T::Real, unit::Symbol)
    if unit === :fm
        return Float64(T)
    elseif unit === :MeV
        return Float64(T) * MEV_PER_FM
    else
        throw(ArgumentError("Unsupported temperature unit: $(unit). Use :fm or :MeV."))
    end
end

"""
Convert temperature from a given unit to fm.

Supported units:
- `:fm`
- `:MeV`
"""
function temperature_to_fm(T::Real, unit::Symbol)
    if unit === :fm
        return Float64(T)
    elseif unit === :MeV
        return Float64(T) * FM_PER_MEV
    else
        throw(ArgumentError("Unsupported temperature unit: $(unit). Use :fm or :MeV."))
    end
end
