using Ramnet.Encoders
using Ramnet

using Random
using StaticArrays

using ..Approximators:QDiscriminator
using ..Buffers: MultiStepDynamicBuffer, add!
using ..Environments

mutable struct MCDiscriminatorAgent{AS,OS,T <: Real,O <: StaticArray{Tuple{OS},T,1},A <: Real,E <: AbstractEncoder{T}} <: AbstractAgent{O,A}
    actions::SVector{AS,A}
    discount::Float64
    ϵ::Float64
    Q̂::QDiscriminator{AS,OS,T,O,A,E}
    buffer::MultiStepDynamicBuffer{O,A}
    observation::Union{Nothing,O}
    done::Bool
    action::Union{Nothing,A}
    rng::MersenneTwister
end

# TODO: Validate actions and encoder and ϵ
function MCDiscriminatorAgent(::Type{O}, actions::StaticArray{Tuple{AS},A,1}, n, forgetting_factor, discount, ϵ, encoder::E; seed::Union{Nothing,Int}=nothing) where {AS,OS,T,O <: StaticArray{Tuple{OS},T,1},A,E <: AbstractEncoder{T}}
    !isnothing(seed) && seed < 0 && throw(DomainError(seed, "Seed must be non-negative"))
    rng = isnothing(seed) ? MersenneTwister() : MersenneTwister(seed)

    n < 1 && throw(DomainError(n, "Tuple size must be at least 1"))

    Q̂ = QDiscriminator(O, actions, n, forgetting_factor, encoder, seed)
    buffer = MultiStepDynamicBuffer{O,A}()

    MCDiscriminatorAgent{AS,OS,T,O,A,E}(actions, discount, ϵ, Q̂, buffer, nothing, false, nothing, rng)
end

# TODO: Return type can only be determined at runtime. Make AbstractEnvironment parametric
function MCDiscriminatorAgent(
    env, encoder::E; n, forget::Float64=1.0, gamma::Float64=1.0, epsilon::Float64=0.01, seed::Union{Nothing,Int}=nothing) where {T <: Real,E <: AbstractEncoder{T}}
    MCDiscriminatorAgent{observation_type(env),action_type(env),T,E}(action_set(env), n, observation_length(env), forget, gamma, epsilon, encoder; seed)
end

function MCDiscriminatorAgent(env, encoder::E, n, forgetting_factor=1.0, discount=1.0, ϵ=0.01, seed=nothing) where {T <: Real,E <: AbstractEncoder{T}}
    MCDiscriminatorAgent(observation_type(env), action_set(env), n, forgetting_factor, discount, ϵ, encoder; seed)
end

function observe!(agent::MCDiscriminatorAgent{AS,OS,T,O,A,E}, observation::O) where {AS,OS,T,O,A,E} # {O,A <: Real,T <: Real,E <: AbstractEncoder{T}}
    agent.observation = observation
    agent.done = false
    
    nothing
end

# TODO: All observe! methods are the same for all agents. Generalize.
# TODO: This method does not need to take in the action
function observe!(agent::MCDiscriminatorAgent{AS,OS,T,O,A,E}, action::A, reward::Float64, observation::O, done::Bool) where {AS,OS,T,O,A,E} # {O,A <: Real,T <: Real,E <: AbstractEncoder{T}}
    add!(agent.buffer, agent.observation, agent.action, reward)

    agent.observation = observation
agent.done = done

    nothing
end

function select_action!(agent::MCDiscriminatorAgent{AS,OS,T,O,A,E}, observation::O) where {AS,OS,T,O,A,E} # {O,A <: Real,T <: Real,E <: AbstractEncoder{T}}
    if rand(agent.rng) < agent.ϵ
        agent.action = rand(agent.rng, agent.actions)
    else
        agent.action = select_action(agent.Q̂, observation)
    end

    return agent.action
end

function update!(agent::MCDiscriminatorAgent{AS,OS,T,O,A,E}) where {AS,OS,T,O,A,E} # {O,A <: Real,T <: Real,E <: AbstractEncoder{T}}
    if agent.done
        G = 0.0
        for transition in Iterators.reverse(agent.buffer)
            G = transition.reward + agent.discount * G

            update!(agent.Q̂, transition.observation, transition.action, G)
        end

        reset!(agent.buffer)
    end

    nothing
end

function reset!(agent::MCDiscriminatorAgent{AS,OS,T,O,A,E}; seed::Union{Nothing,Int}=nothing) where {AS,OS,T,O,A,E} # {O,A <: Real,S,T <: Real,E <: AbstractEncoder{T}}
    if !isnothing(seed)
        if seed ≥ 0
        seed!(agent.rng, seed)
        else
            throw(DomainError(seed, "Seed must be non-negative"))
        end
    end
    
    reset!(agent.Q̂; seed)
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