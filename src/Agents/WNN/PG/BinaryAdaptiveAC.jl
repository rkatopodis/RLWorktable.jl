using Ramnet.Encoders

using ..Approximators:AdaptiveBinaryPolicy, VDiscriminator
using ..Buffers: MultiStepDynamicBuffer, add!

mutable struct BinaryAdaptiveAC{OS,T <: Real,O <: StaticArray{Tuple{OS},T,1},E <: AbstractEncoder{T}} <: AbstractAgent{O,Int}
    steps::Int
    γ::Float64
    actor::AdaptiveBinaryPolicy{OS,T,O,E}
    critic::VDiscriminator
    buffer::MultiStepDynamicBuffer{O,Int}
    observation::Union{Nothing,O}
    done::Bool
    action::Union{Nothing,Int}
    min_obs::Union{Nothing,O}
    max_obs::Union{Nothing,O}
    # rng::MersenneTwister
end

function BinaryAdaptiveAC(::Type{O}, steps, n, λ, μ, η, epochs, discount, forgetting_factor, encoder::E; seed::Union{Nothing,Int}=nothing) where {OS,T <: Real,O <: StaticArray{Tuple{OS},T,1},E <: AbstractEncoder{T}}
    # !isnothing(seed) && seed < 0 && throw(DomainError(seed, "Seed must be non-negative"))
    # rng = isnothing(seed) ? MersenneTwister() : MersenneTwister(seed)

    n < 1 && throw(DomainError(n, "Tuple size must be at least 1"))

    actor = AdaptiveBinaryPolicy(O, n, encoder; λ, μ, η, epochs, partitioner=:uniform_random, seed)
    critic = VDiscriminator(O, 8, forgetting_factor, encoder, seed)
    buffer = MultiStepDynamicBuffer{O,Int}()

    BinaryAdaptiveAC{OS,T,O,E}(steps, discount, actor, critic, buffer, nothing, false, nothing, nothing, nothing)
end

function BinaryAdaptiveAC(env, encoder::E; steps, tuple_size, lambda, mu, learning_rate, epochs, discount, forgetting_factor=1.0, seed=nothing) where {T <: Real,E <: AbstractEncoder{T}}
    BinaryAdaptiveAC(observation_type(env), steps, tuple_size, lambda, mu, learning_rate, epochs, discount, forgetting_factor, encoder; seed)
end

function observe!(agent::BinaryAdaptiveAC{OS,T,O,E}, observation::O) where {OS,T,O,E}
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
function observe!(agent::BinaryAdaptiveAC{OS,T,O,E}, action::Int, reward::Float64, observation::O, done::Bool) where {OS,T,O,E}
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

function _select_action(agent::BinaryAdaptiveAC{OS,T,O,E}, observation::O) where {OS,T,O,E}
    select_action(agent.actor, observation)
end

function select_action!(agent::BinaryAdaptiveAC{OS,T,O,E}, observation::O) where {OS,T,O,E}
    if !isnothing(agent.action)
        return agent.action
    end
  
    return _select_action(agent, observation)
end

function update!(agent::BinaryAdaptiveAC)
    if agent.done
        G = 0.0
        for transition in Iterators.reverse(agent.buffer)
            G = transition.reward + agent.γ * G
  
            δ = G - agent.critic(transition.observation)
            update!(agent.actor, transition.observation, transition.action, δ)
            update!(agent.critic, transition.observation, G)
        end
  
        reset!(agent.buffer)
    elseif length(agent.buffer) == agent.steps
        G = agent.critic(agent.observation)
  
        for transition in Iterators.reverse(agent.buffer)
            G = transition.reward + agent.γ * G
        end
  
        t = popfirst!(agent.buffer)
  
        δ = G - agent.critic(t.observation)
        update!(agent.actor, t.observation, t.action, δ)
        update!(agent.critic, t.observation, G)
    end
  
    nothing
end

function reset!(agent::BinaryAdaptiveAC; seed::Union{Nothing,Int}=nothing)
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
