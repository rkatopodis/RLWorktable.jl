module RLWorktable

function update! end
function reset! end
function select_action end

include("Buffers/Buffers.jl")
using .Buffers
export DynamicBuffer, add!, reset!

include("Environments/Environments.jl")
using .Environments
export env, step!, render, close, GymCartPoleV1, GymMountainCarDiscrete, GymAcrobot, GymBreakoutRAM, CartPoleV1

include("Approximators/Approximators.jl")

include("Agents/Agents.jl")
using .Agents
export agent,
  RandomAgent,
  MonteCarloDiscountedDiscriminatorAgent,
  SARSADiscountedDiscriminatorAgent,
  ExpectedSARSADiscriminatorAgent,
  QLearningDiscountedDiscriminatorAgent,
  MCDiscriminatorAgent,
  MCDifferentialDiscriminatorAgent,
  FunctionalPG

include("Simulations/Simulations.jl")
using .Simulations
export simulate_episode, EnvironmentLoop, simulate, mean_total_reward, experiment

include("Tools/Tools.jl")

end
