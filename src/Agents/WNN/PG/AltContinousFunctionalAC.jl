using Ramnet.Encoders
using Ramnet.Models.AltDiscriminators:RegressionDiscriminator

using LinearAlgebra:Diagonal

using ..Approximators:AltContinousActionPolicy
using ..Buffers: MultiStepDynamicBuffer, add!

mutable struct AltContinousFunctionalAC{OS,AS,T <: Real,O <: StaticArray{Tuple{OS},T,1},A <: StaticArray{Tuple{AS},T,1},E <: AbstractEncoder{T}} <: AbstractAgent{O,A}
    steps::Int
    γ::Float64
    actor::AltContinousActionPolicy{OS,AS,T,O,A,E}
    critic::RegressionDiscriminator{1}
    buffer::MultiStepDynamicBuffer{O,A}
    observation::Union{Nothing,O}
    done::Bool
    action::Union{Nothing,A}
    min_obs::Union{Nothing,O}
    max_obs::Union{Nothing,O}
    # rng::MersenneTwister
end

function AltContinousFunctionalAC(::Type{O}, ::Type{A}, steps, n, η, epochs, discount, forgetting_factor, encoder::E; seed::Union{Nothing,Int}=nothing) where {OS,AS,T <: Real,O <: StaticArray{Tuple{OS},T,1},A <: StaticArray{Tuple{AS},T,1},E <: AbstractEncoder{T}}
    # !isnothing(seed) && seed < 0 && throw(DomainError(seed, "Seed must be non-negative"))
    # rng = isnothing(seed) ? MersenneTwister() : MersenneTwister(seed)

    n < 1 && throw(DomainError(n, "Tuple size must be at least 1"))

    actor = AltContinousActionPolicy(O, A, n, encoder; η, epochs, partitioner=:uniform_random, seed)
    critic = RegressionDiscriminator{1}(OS, 32, encoder; seed, γ=forgetting_factor)
    buffer = MultiStepDynamicBuffer{O,A}()

    AltContinousFunctionalAC{OS,AS,T,O,A,E}(steps, discount, actor, critic, buffer, nothing, false, nothing, nothing, nothing)
end

# function AltContinousFunctionalAC(env, encoder::E; steps, tuple_size, learning_rate, epochs, discount, forgetting_factor=1.0, cov::C, seed=nothing) where {T <: Real,E <: AbstractEncoder{T},C <: AbstractMatrix{T}}
#     AltContinousFunctionalAC(observation_type(env), action_type(env), steps, tuple_size, learning_rate, epochs, discount, forgetting_factor, encoder, cov; seed)
# end

function AltContinousFunctionalAC(env, encoder::E; steps, tuple_size, learning_rate, epochs, discount, forgetting_factor=1.0, seed=nothing) where {T <: Real,E <: AbstractEncoder{T}}
    # AltContinousFunctionalAC(env, encoder; steps, tuple_size, learning_rate, epochs, discount, forgetting_factor, cov=Diagonal(cov), seed=nothing)
    AltContinousFunctionalAC(observation_type(env), action_type(env), steps, tuple_size, learning_rate, epochs, discount, forgetting_factor, encoder; seed)
end

function observe!(agent::AltContinousFunctionalAC{OS,AS,T,O,A,E}, observation::O) where {OS,AS,T,O,A,E}
    agent.observation = observation
    agent.done = false

    agent.action = _select_action(agent, observation)

    # Finding the edges of the domain
    if isnothing(agent.min_obs)
        agent.min_obs = observation
    else
        agent.min_obs = min.(agent.min_obs, observation)
    end

    if isnothing(agent.max_obs)
        agent.max_obs = observation
    else
        agent.max_obs = max.(agent.max_obs, observation)
    end

    nothing
end

# TODO: All observe! methods are the same for all agents. Generalize.
# TODO: This method does not need to take in the action
function observe!(agent::AltContinousFunctionalAC{OS,AS,T,O,A,E}, action::A, reward::Float64, observation::O, done::Bool) where {OS,AS,T,O,A,E}
    add!(agent.buffer, agent.observation, agent.action, reward)

    if !done
        agent.action = _select_action(agent, observation)
    end

    agent.observation = observation
    agent.done = done

  # Finding the edges of the domain
    agent.min_obs = min.(agent.min_obs)
    agent.max_obs = max.(agent.max_obs)

    nothing
end

function _select_action(agent::AltContinousFunctionalAC{OS,AS,T,O,A,E}, observation::O) where {OS,AS,T,O,A,E}
    select_action(agent.actor, observation)
end

function select_action!(agent::AltContinousFunctionalAC{OS,AS,T,O,A,E}, observation::O) where {OS,AS,T,O,A,E}
    if !isnothing(agent.action)
        return agent.action
    end
  
    return _select_action(agent, observation)
end

function update!(agent::AltContinousFunctionalAC)
    if agent.done
        G = 0.0
        for transition in Iterators.reverse(agent.buffer)
            G = transition.reward + agent.γ * G
  
            δ = G - (predict(agent.critic, transition.observation) |> first)
            update!(agent.actor, transition.observation, transition.action, δ)
            train!(agent.critic, transition.observation, G)
        end
  
        reset!(agent.buffer)
    elseif length(agent.buffer) == agent.steps
        G = predict(agent.critic, agent.observation) |> first
  
        for transition in Iterators.reverse(agent.buffer)
            G = transition.reward + agent.γ * G
        end
  
        t = popfirst!(agent.buffer)
  
        δ = G - (predict(agent.critic, t.observation) |> first)
        update!(agent.actor, t.observation, t.action, δ)
        train!(agent.critic, t.observation, G)
    end
  
    nothing
end

function reset!(agent::AltContinousFunctionalAC; seed::Union{Nothing,Int}=nothing)
    if !isnothing(seed)
        if seed ≥ 0
            seed!(agent.rng, seed)
        else
            throw(DomainError(seed, "Seed must be non-negative"))
        end
    end
    
    reset!(agent.actor)
    Ramnet.reset!(agent.critic)
    reset!(agent.buffer)

    nothing
end
