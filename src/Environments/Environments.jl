module Environments

import ..reset!

export AbstractEnvironment,
    step!,
    set_render_mode!,
    render,
    close,
    env,
    observation_type,
    observation_extrema,
    observation_length,
    action_type,
    action_length,
    action_set,
    make_env

abstract type AbstractEnvironment{O <: AbstractVector,A} end

function step! end
function set_render_mode! end
function render end
function close end # Close render window. Gym-specific
function observation_type end
function observation_extrema end
function observation_length end
function action_type end
function action_length end
function action_set end

include("GymWrappers.jl")
using .GymWrappers:
    GymCartPoleV1,
    GymMountainCarDiscrete,
    GymAcrobot,
    GymBreakoutRAM,
    GymInvertedPendulum,
    GymHopperBulletV0
export GymCartPoleV1,
    GymMountainCarDiscrete,
    GymAcrobot,
    GymBreakoutRAM,
    GymInvertedPendulum,
    GymHopperBulletV0

include("CartPole.jl")
export CartPole, CartPoleV1

const environment_table = Dict{Symbol,Type{<:AbstractEnvironment}}(
    :GymCartPoleV1 => GymCartPoleV1,
    :CartPole => CartPole,
    :GymMountainCarDiscrete => GymMountainCarDiscrete,
    :GymAcrobot => GymAcrobot,
    :GymBreakoutRAM => GymBreakoutRAM,
    :GymInvertedPendulum => GymInvertedPendulum,
)

function env(name::String; kargs...)
    environment_table[Symbol(name)](kargs...)
end

# TODO: Check for the presence of required keys, default values for non-essential ones
function env(spec::Dict{Symbol,Any})
    env(spec[:name]; spec[:args]...)
end

function make_env(envspec::Dict{Symbol,Any})
    environment_table[Symbol(envspec[:name])]()
end

end
