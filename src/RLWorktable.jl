module RLWorktable

include("ExperienceBuffers/ExperienceBuffers.jl")
using .ExperienceBuffers
export DynamicExperienceBuffer, add!, reset!

include("Environments/Environments.jl")
using .Environments
export GymCartPoleV1

include("Agents/Agents.jl")
using .Agents
export RandomAgent, MonteCarloDiscountedDiscriminatorAgent, select_action

include("Simulations/Simulations.jl")
using .Simulations
export simulate_episode, EnvironmentLoop, simulate

end
