module Agents

using Random
using StatsBase: mean, sample, pweights
using Ramnet.Encoders

using ..Environments: AbstractEnvironment, observation_type, observation_length, observation_extrema, action_type, action_set, action_length
import ..reset!, ..update!, ..select_action

export AbstractAgent, select_action!, observe!, update!, agent, encoding, make_agent

# TODO: Make this type parametric. All agents should know the types of its observations,
#       actions and encoder
abstract type AbstractAgent{O <: AbstractVector,A} end

# I can make use of multiple dispatch here so that both observe_first! and observe!
# are called observe!. The "observe_first!" equivalent only takes a observation while
# the "observe!" equivalent takes an action, reward and next observation
# function observe_first! end
# function select_action end
function observe! end
# function update! end

# NOTE: This version assumes discriminators take in as input binary patterns, already encoded
# function q_values!(agent::G, observation::O, dest::AbstractVector{Float64}) where {O <: AbstractVector,A <: Real,G <: AbstractAgent{O,A}}
#     q_max = typemin(Float64)
#     q = 0.0
#     encoded_obs = encode(agent.encoder, observation) # TODO: Don't have to create a bunch of temp arrays here
#     for (i, action) in Iterators.zip(eachindex(dest), agent.actions)
#         q = predict(agent.Q̂[action], encoded_obs)
#         dest[i] = q
#         q_max = q > q_max ? q : q_max
#     end

#     return q_max
# end

# TODO: Make a version that takes a destination vector
# TODO: This implementation is wrong. It must take into account the epsilon prob.
#       Refer to the Alberta MOOC exercise.
function expected_q_value(agent::G, observation::O) where {O <: AbstractVector,A <: Real,G <: AbstractAgent{O,A}}
    values = similar(agent.actions, Float64)
    q_max = q_values!(agent, observation, values)

    return mean(values, pweights((values .== q_max) .+ agent.ϵ))
end

include("RandomAgent.jl")
export RandomAgent

# include("WNN/MonteCarloDiscountedDiscriminatorAgent.jl")
# export MonteCarloDiscountedDiscriminatorAgent

include("WNN/SARSADiscountedDiscriminatorAgent.jl")
export SARSADiscountedDiscriminatorAgent

# include("WNN/ExpectedSARSADiscriminatorAgent.jl")
# export ExpectedSARSADiscriminatorAgent

# include("WNN/QLearningDiscountedDiscriminatorAgent.jl")
# export QLearningDiscountedDiscriminatorAgent

include("WNN/MCDiscriminatorAgent.jl")
export MCDiscriminatorAgent

# include("WNN/MCDifferentialDiscriminatorAgent.jl")
# export MCDifferentialDiscriminatorAgent

include("WNN/PG/FunctionalPG.jl")
export FunctionalPG

include("WNN/PG/ContinousFunctionalPG.jl")
export ContinousFunctionalPG

include("WNN/PG/FunctionalAC.jl")
export FunctionalAC

include("WNN/PG/ContinousFunctionalAC.jl")
export ContinousFunctionalAC

const agent_table = Dict{Symbol,Type{<:AbstractAgent}}(
    :MCDiscriminatorAgent => MCDiscriminatorAgent,
    # :MonteCarloDiscountedDiscriminatorAgent => MonteCarloDiscountedDiscriminatorAgent,
    :SARSADiscountedDiscriminatorAgent => SARSADiscountedDiscriminatorAgent,
    # :ExpectedSARSADiscriminatorAgent => ExpectedSARSADiscriminatorAgent,
    # :QLearningDiscountedDiscriminatorAgent => QLearningDiscountedDiscriminatorAgent
)

const encoding_table = Dict{Symbol,Type{<:AbstractEncoder}}(
    :Thermometer => Thermometer,
    :CircularThermometer => CircularThermometer
)

function encoding(name::String, minimum, maximum, resolution)
    encoding_table[Symbol(name)](eltype(minimum), minimum, maximum, resolution)
end

function agent(name::String, ::Type{O}, ::Type{A}, actions, obs_size, encoder::E; kargs...) where {O,A,E <: AbstractEncoder}
    agent_table[Symbol(name)]{O,A,E}(actions, obs_size, encoder; kargs...)
end

function agent(agent_spec::Dict{Symbol,Any}, env::AbstractEnvironment)
    enc_spec = agent_spec[:encoding]
    enc = encoding(
        enc_spec[:name],
        observation_extrema(env)...,
        enc_spec[:resolution]
    )

    agent(
       agent_spec[:name],
       observation_type(env),
       action_type(env),
       action_set(env),
       observation_length(env),
       enc;
       agent_spec[:args]...
    )
end

function make_agent(env::AbstractEnvironment, agentspec::Dict{Symbol,Any})
    encoder_name = agentspec[:encoding][:name] |> Symbol
    encoder_resolution = agentspec[:encoding][:resolution]

    enc = encoding_table[encoder_name](observation_extrema(env)..., encoder_resolution)

    agent_name = agentspec[:name] |> Symbol

    agent_table[agent_name](env, enc; agentspec[:args]...)
end

# TODO: Make this a function of the Q-value approximation, not the agent.
#       (Define somewhere else a DiscreteQDiscriminator. Maybe in a Approximators
#        module)
# function select_action!(agent::MonteCarloDiscountedDiscriminatorAgent{O,A,E}, observation::O; ϵ::Float64=agent.ϵ) where {O <: AbstractVector,A <: Real,E <: AbstractEncoder}
#     # TODO: I could avoid creating these temporary arrays here
#     values = similar(agent.actions, Float64)
#     q_max = q_values!(agent, observation, values)

#     if rand(agent.rng) < ϵ # Random action taken with ϵ probability
#         agent.action = rand(agent.rng, agent.actions)
#     else
#         # Selecting action that maximizes Q̂, with ties broken randomly
#         agent.action = sample(agent.rng, agent.actions, pweights(values .== q_max))
#     end

#     return agent.action::A
# end

end
