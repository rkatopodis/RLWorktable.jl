module Environments

import ..reset!

export AbstractEnvironment, 
step!,
render,
close,
env,
observation_type,
observation_extrema,
observation_length,
action_type,
action_set

abstract type AbstractEnvironment{O <: AbstractVector,A <: Real} end

function step! end
function render end
function close end # Close render window. Gym-specific
function observation_type end
function observation_extrema end
function observation_length end
function action_type end
function action_set end

include("GymWrappers.jl")
using .GymWrappers: GymCartPoleV1, GymMountainCarDiscrete, GymAcrobot, GymBreakoutRAM, GymInvertedPendulum
export GymCartPoleV1, GymMountainCarDiscrete, GymAcrobot, GymBreakoutRAM, GymInvertedPendulum

include("CartPole.jl")
export CartPole, CartPoleV1

const environment_table = Dict{Symbol,Type{<:AbstractEnvironment}}(
    :GymCartPoleV1 => GymCartPoleV1,
    :CartPole => CartPole,
    :GymMountainCarDiscrete => GymMountainCarDiscrete,
    :GymAcrobot => GymAcrobot,
    :GymBreakoutRAM => GymBreakoutRAM,
    :GymInvertedPendulum => GymInvertedPendulum
)

function env(name::String; kargs...)
    environment_table[Symbol(name)](kargs...)
end

# TODO: Check for the presence of required keys, default values for non-essential ones
function env(spec::Dict{Symbol,Any})
    env(spec[:name]; spec[:args]...)
end

end