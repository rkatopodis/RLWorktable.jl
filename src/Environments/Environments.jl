module Environments

import ..reset!

export AbstractEnvironment, step!, render, close, GymCartPoleV1

abstract type AbstractEnvironment end

function step! end
# function reset! end
function render end
function close end # Close render window. Gym-specific

include("GymWrappers.jl")
using .GymWrappers: GymCartPoleV1

include("CartPole.jl")
export CartPole, CartPoleV1
end