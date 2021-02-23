using Ramnet.Encoders
using Ramnet

using Random

using ..Buffers: MultiStepDynamicBuffer, add!
using ..Environments

mutable struct MCDifferentialDiscriminatorAgent{O <: AbstractVector,A <: Real,T <: Real,E <: AbstractEncoder{T}} <: AbstractAgent{O,A}
    actions::UnitRange{A}
    n::Int
    obs_size::Int
    α::Float64
    γ::Float64
    ϵ::Float64
    epochs::Int
    encoder::E
    Q̂::Dict{A,DifferentialDiscriminator{T,E}}
    buffer::MultiStepDynamicBuffer{O,A}
    observation::Union{Nothing,O}
    done::Bool
    action::Union{Nothing,A}
    rng::MersenneTwister

  # TODO: Validate actions and encoder and ϵ
    function MCDifferentialDiscriminatorAgent{O,A,T,E}(actions, n, obs_size, α, γ, ϵ, epochs, encoder::E; seed::Union{Nothing,Int}=nothing) where {O <: AbstractVector,A <: Real,T <: Real,E <: AbstractEncoder{T}}
        !isnothing(seed) && seed < 0 && throw(DomainError(seed, "Seed must be non-negative"))
        rng = isnothing(seed) ? MersenneTwister() : MersenneTwister(seed)

        n < 1 && throw(DomainError(n, "Tuple size must be at least 1"))
        epochs < 1 && throw(DomainError(epochs, "Number of epochs must be at least 1"))

        Q̂ = Dict(a => DifferentialDiscriminator(obs_size, n, encoder; α) for a in actions)
        buffer = MultiStepDynamicBuffer{O,A}()

        new(actions, n, obs_size, α, γ, ϵ, epochs, encoder, Q̂, buffer, nothing, false, nothing, rng)
    end
end

function MCDifferentialDiscriminatorAgent(
  env::AbstractEnvironment{O,A}, encoder::E; n, α::Float64=1.0, γ::Float64=1.0, ϵ::Float64=0.01, epochs::Int=1, seed::Union{Nothing,Int}=nothing) where {O <: AbstractVector,A <: Real,T <: Real,E <: AbstractEncoder{T}}
    MCDifferentialDiscriminatorAgent{O,A,T,E}(action_set(env), n, observation_length(env), α, γ, ϵ, epochs, encoder; seed)
end

function observe!(agent::MCDifferentialDiscriminatorAgent{O,A,T,E}, observation::O) where {O,A <: Real,T <: Real,E <: AbstractEncoder{T}}
    agent.observation = observation
    agent.done = false
  
    nothing
end

# TODO: All observe! methods are the same for all agents. Generalize.
# TODO: This method does not need to take in the action
function observe!(agent::MCDifferentialDiscriminatorAgent{O,A,T,E}, action::A, reward::Float64, observation::O, done::Bool) where {O <: AbstractVector,A <: Real,T <: Real,E <: AbstractEncoder{T}}
    add!(agent.buffer, agent.observation, agent.action, reward)

    agent.observation = observation
    agent.done = done

    nothing
end

function update!(agent::MCDifferentialDiscriminatorAgent{O,A,T,E}) where {O <: AbstractVector,A <: Real,T <: Real,E <: AbstractEncoder{T}}
    if agent.done
        G = 0.0
        len = length(agent.buffer)
        Gs = Vector{Float64}(undef, len)
        obs = O[]
        idx_by_action = Dict{A,Vector{Int}}()

        for (i, transition) in Iterators.enumerate(Iterators.reverse(agent.buffer))
            G = transition.reward + agent.γ * G
            Gs[i] = G
            # push!(idx_by_action[transition.action], i)
            push!(get!(idx_by_action, transition.action, Int[]), i)
            push!(obs, transition.observation)
          # train!(
          #     agent.Q̂[transition.action],
          #     transition.observation,
          #     G
          # )
        end

        # @show Gs
        # @show idx_by_action
        for a in keys(idx_by_action)
            train!(
              agent.Q̂[a],
              hcat(obs[idx_by_action[a]]...),
              Gs[idx_by_action[a]];
              epochs=agent.epochs
            )
        end

        reset!(agent.buffer)
    end

    nothing
end

function Agents.reset!(agent::MCDifferentialDiscriminatorAgent{O,A,T,E}) where {O <: AbstractVector,A <: Real,T <: Real,E <: AbstractEncoder{T}}
    for action in agent.actions
        agent.Q̂[action] = DifferentialDiscriminator(agent.obs_size, agent.n, agent.encoder; α=agent.α)
    end

    reset!(agent.buffer)

    nothing
end