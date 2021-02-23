using Ramnet.Encoders
using Ramnet

using Random

using ..Buffers: MultiStepDynamicBuffer, add!
using ..Environments

mutable struct MCDiscriminatorAgent{O <: AbstractVector,A <: Real,S,T <: Real,E <: AbstractEncoder{T}} <: AbstractAgent{O,A}
    actions::UnitRange{A}
    n::Int
    obs_size::Int
    α::Float64
    γ::Float64
    ϵ::Float64
    # episodes::Int
    encoder::E
    Q̂::Dict{A,RegressionDiscriminator{S,T,E}}
    buffer::MultiStepDynamicBuffer{O,A}
    observation::Union{Nothing,O}
    done::Bool
    action::Union{Nothing,A}
    rng::MersenneTwister

    # TODO: Validate actions and encoder and ϵ
    # TODO: Do I need the size if the encoder is given? Not if a knew the observation length (static array?)
    function MCDiscriminatorAgent{O,A,S,T,E}(actions, n, obs_size, α, γ, ϵ, encoder::E; seed::Union{Nothing,Int}=nothing) where {O,A <: Real,S,T <: Real,E <: AbstractEncoder{T}}
        !isnothing(seed) && seed < 0 && throw(DomainError(seed, "Seed must be non-negative"))
        rng = isnothing(seed) ? MersenneTwister() : MersenneTwister(seed)

        n < 1 && throw(DomainError(n, "Tuple size must be at least 1"))

        Q̂ = Dict(a => RegressionDiscriminator(obs_size, n, encoder; style=S, α) for a in actions)
        buffer = MultiStepDynamicBuffer{O,A}()

        new(actions, n, obs_size, α, γ, ϵ, encoder, Q̂, buffer, nothing, false, nothing, rng)
    end
end

# TODO: Return type can only be determined at runtime. Make AbstractEnvironment parametric
function MCDiscriminatorAgent(
    env, encoder::E; n, style::Symbol=:original, α::Float64=1.0, gamma::Float64=1.0, epsilon::Float64=0.01, seed::Union{Nothing,Int}=nothing) where {T <: Real,E <: AbstractEncoder{T}}
    MCDiscriminatorAgent{observation_type(env),action_type(env),style,T,E}(action_set(env), n, observation_length(env), α, gamma, epsilon, encoder; seed)
end

function observe!(agent::MCDiscriminatorAgent{O,A,S,T,E}, observation::O) where {O,A <: Real,S,T <: Real,E <: AbstractEncoder{T}}
    agent.observation = observation
    agent.done = false
    
    nothing
end

# TODO: All observe! methods are the same for all agents. Generalize.
# TODO: This method does not need to take in the action
function observe!(agent::MCDiscriminatorAgent{O,A,S,T,E}, action::A, reward::Float64, observation::O, done::Bool) where {O,A <: Real,S,T <: Real,E <: AbstractEncoder{T}}
    add!(agent.buffer, agent.observation, agent.action, reward)

    agent.observation = observation
    agent.done = done

    nothing
end

function update!(agent::MCDiscriminatorAgent{O,A,S,T,E}) where {O,A <: Real,S,T <: Real,E <: AbstractEncoder{T}}
    if agent.done
        G = 0.0
        for transition in Iterators.reverse(agent.buffer)
            G = transition.reward + agent.γ * G
            train!(
                agent.Q̂[transition.action],
                transition.observation,
                G
            )
        end

        reset!(agent.buffer)
    end

    nothing
end

function Agents.reset!(agent::MCDiscriminatorAgent{O,A,S,T,E}) where {O,A <: Real,S,T <: Real,E <: AbstractEncoder{T}}
    for action in agent.actions
        agent.Q̂[action] = RegressionDiscriminator(agent.obs_size, agent.n, agent.encoder; style=s, α=agent.α)
    end

    reset!(agent.buffer)

    nothing
end

# function q_values!(agent::MCDiscriminatorAgent{O,A,S,T,E}, observation::O, dest::AbstractVector{Float64}) where {O,A <: Real,S,T <: Real,E <: AbstractEncoder{T}}
#     q_max = typemin(Float64)
#     q = 0.0
#     for (i, action) in Iterators.zip(eachindex(dest), agent.actions)
#         q = predict(agent.Q̂[action], observation)
#         dest[i] = q
#         q_max = q > q_max ? q : q_max
#     end

#     return q_max
# end