module Agents

export AbstractAgent, select_action, observe!, update!

abstract type AbstractAgent end

# I can make use of multiple dispatch here so that both observe_first! and observe!
# are called observe!. The "observe_first!" equivalent only takes a observation while
# the "observe!" equivalent takes an action, reward and next observation
# function observe_first! end
function select_action end
function observe! end
function update! end

include("RandomAgent.jl")
export RandomAgent

include("WNN/MonteCarloDiscountedDiscriminatorAgent.jl")
export MonteCarloDiscountedDiscriminatorAgent

end