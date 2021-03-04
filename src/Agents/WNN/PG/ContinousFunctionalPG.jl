using Ramnet.Encoders

using ..Approximators:ContinousActionPolicy
using ..Buffers: MultiStepDynamicBuffer, add!

mutable struct ContinousFunctionalPG{D,T <: Real,O <: AbstractVector{T},A <: AbstractVector{<:Real},E <: AbstractEncoder{T},C <: AbstractMatrix{T}} <: AbstractAgent{O,A}
    γ::Float64
    policy::ContinousActionPolicy{D,T,O,E,C}
    buffer::MultiStepDynamicBuffer{O,A}
    observation::Union{Nothing,O}
    done::Bool
    action::Union{Nothing,A}
    min_obs::Union{Nothing,O}
    max_obs::Union{Nothing,O}
    # rng::MersenneTwister

    function ContinousFunctionalPG{D,T,O,A,E,C}(n, obs_size, η, γ, encoder::E, cov::C; seed::Union{Nothing,Int}=nothing) where {D,T <: Real,O <: AbstractVector{T},A <: AbstractVector{<:Real},E <: AbstractEncoder{T},C <: AbstractMatrix{T}}
        # !isnothing(seed) && seed < 0 && throw(DomainError(seed, "Seed must be non-negative"))
        # rng = isnothing(seed) ? MersenneTwister() : MersenneTwister(seed)

        n < 1 && throw(DomainError(n, "Tuple size must be at least 1"))

        policy = ContinousActionPolicy{D,T,O,E,C}(obs_size, n, encoder;cov, η, partitioner=:uniform, seed)
        buffer = MultiStepDynamicBuffer{O,A}()

        new(γ, policy, buffer, nothing, false, nothing, nothing, nothing)
    end
end

function ContinousFunctionalPG(env, encoder::E; n, η, γ, cov::C, seed=nothing) where {T <: Real,E <: AbstractEncoder{T},C <: AbstractMatrix{T}}
    ContinousFunctionalPG{action_length(env),T,observation_type(env),action_type(env),E,C}(n, observation_length(env), η, γ, encoder, cov; seed)
end

function observe!(agent::ContinousFunctionalPG{D,T,O,A,E,C}, observation::O) where {D,T <: Real,O <: AbstractVector{T},A <: AbstractVector{<:Real},E <: AbstractEncoder{T},C <: AbstractMatrix{T}}
    agent.observation = observation
    agent.done = false

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
function observe!(agent::ContinousFunctionalPG{D,T,O,A,E,C}, action::A, reward::Float64, observation::O, done::Bool) where {D,T <: Real,O <: AbstractVector{T},A <: AbstractVector{<:Real},E <: AbstractEncoder{T},C <: AbstractMatrix{T}}
    add!(agent.buffer, agent.observation, agent.action, reward)

    agent.observation = observation
    agent.done = done

  # Finding the edges of the domain
    agent.min_obs = min.(agent.min_obs)
    agent.max_obs = max.(agent.max_obs)

    nothing
end

function select_action!(agent::ContinousFunctionalPG{D,T,O,A,E,C}, observation::O) where {D,T <: Real,O <: AbstractVector{T},A <: AbstractVector{<:Real},E <: AbstractEncoder{T},C <: AbstractMatrix{T}}
    agent.action = select_action(agent.policy, observation)

    return agent.action
end

function update!(agent::ContinousFunctionalPG)
    if agent.done
        G = 0.0
        for transition in Iterators.reverse(agent.buffer)
            G = transition.reward + agent.γ * G

            update!(agent.policy, transition.observation, transition.action, G)
        end

        reset!(agent.buffer)
    end

    nothing
end

function Agents.reset!(agent::ContinousFunctionalPG)
    reset!(agent.policy)
    reset!(agent.buffer)

    nothing
end
