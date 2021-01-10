module Agents

using StatsBase: mean, sample, pweights

import ..reset!

export AbstractAgent, select_action!, observe!, update!

# TODO: Make this type parametric. All agents should know the types of its observations,
#       actions and encoder
abstract type AbstractAgent end

# I can make use of multiple dispatch here so that both observe_first! and observe!
# are called observe!. The "observe_first!" equivalent only takes a observation while
# the "observe!" equivalent takes an action, reward and next observation
# function observe_first! end
# function select_action end
function observe! end
function update! end

function q_values!(agent::A, observation, dest::AbstractVector{Float64}) where {A <: AbstractAgent}
    q_max = typemin(Float64)
    q = 0.0
    encoded_obs = encode(agent.encoder, observation) # TODO: Don't have to create a bunch of temp arrays here
    for (i, action) in Iterators.zip(eachindex(dest), agent.actions)
        q = predict(agent.Q̂[action], encoded_obs)
        dest[i] = q
        q_max = q > q_max ? q : q_max
    end

    return q_max
end

# TODO: Make a version that takes a destination vector 
# TODO: This implementation is wrong. It must take into account the epsilon prob.
#       Refer to the Alberta MOOC exercise.
function expected_q_value(agent::A, observation) where {A <: AbstractAgent}
    values = similar(agent.actions, Float64)
    q_max = q_values!(agent, observation, values)

    return mean(values, pweights((values .== q_max) .+ agent.ϵ))
end

# TODO: Make this a function of the Q-value approximation, not the agent.
#       (Define somewhere else a DiscreteQDiscriminator. Maybe in a Approximators
#        module)
function select_action!(agent::A, observation; ϵ=agent.ϵ) where {A <: AbstractAgent}
    # TODO: I could avoid creating these temporary arrays here
    values = similar(agent.actions, Float64)
    q_max = q_values!(agent, observation, values)

    if rand(agent.rng) < ϵ # Random action taken with ϵ probability
        agent.action = rand(agent.rng, agent.actions)
    else
        # Selecting action that maximizes Q̂, with ties broken randomly
        agent.action = sample(agent.rng, agent.actions, pweights(values .== q_max))
    end

    return agent.action
end

include("RandomAgent.jl")
export RandomAgent

include("WNN/MonteCarloDiscountedDiscriminatorAgent.jl")
export MonteCarloDiscountedDiscriminatorAgent

include("WNN/SARSADiscountedDiscriminatorAgent.jl")
export SARSADiscountedDiscriminatorAgent

include("WNN/ExpectedSARSADiscriminatorAgent.jl")
export ExpectedSARSADiscriminatorAgent

include("WNN/QLearningDiscountedDiscriminatorAgent.jl")
export QLearningDiscountedDiscriminatorAgent

end