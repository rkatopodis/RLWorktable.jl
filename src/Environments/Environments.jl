module Environments

import ..reset!

export AbstractEnvironment, step!, render, close

abstract type AbstractEnvironment end

function step! end
# function reset! end
function render end
function close end # Close render window. Gym-specific

include("GymWrappers.jl")
using .GymWrappers: GymCartPoleV1, GymMountainCarDiscrete
export GymCartPoleV1, GymMountainCarDiscrete

include("CartPole.jl")
export CartPole, CartPoleV1
end