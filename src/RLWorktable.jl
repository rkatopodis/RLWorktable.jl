module RLWorktable

function reset! end

include("Buffers/Buffers.jl")
using .Buffers
export DynamicBuffer, add!, reset!

include("Environments/Environments.jl")
using .Environments
export env, step!, render, close, GymCartPoleV1, GymMountainCarDiscrete, GymAcrobot, CartPoleV1

include("Agents/Agents.jl")
using .Agents
export agent,
  RandomAgent,
  MonteCarloDiscountedDiscriminatorAgent,
  SARSADiscountedDiscriminatorAgent,
  ExpectedSARSADiscriminatorAgent,
  QLearningDiscountedDiscriminatorAgent

include("Simulations/Simulations.jl")
using .Simulations
export simulate_episode, EnvironmentLoop, simulate, mean_total_reward, experiment

include("Tools/Tools.jl")

end
