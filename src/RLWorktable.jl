module RLWorktable

include("Environments/Environments.jl")
using .Environments
export GymCartPoleV1

include("Agents/Agents.jl")
using .Agents
export RandomAgent

include("Simulations/Simulations.jl")
using .Simulations
export simulate_episode

end
