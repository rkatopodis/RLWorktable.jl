using Ramnet.Encoders
using Ramnet

using Random
using StaticArrays

using ..Approximators:QDiscriminator
using ..Buffers: MultiStepDynamicBuffer, add!
using ..Environments

mutable struct SARSADiscountedDiscriminatorAgent{AS,OS,T <: Real,O <: StaticArray{Tuple{OS},T,1},A <: Real,E <: AbstractEncoder{T}} <: AbstractAgent{O,A}
    steps::Int
    actions::SVector{AS,A}
    discount::Float64
    ϵ_start::Float64
    ϵ_end::Float64
    ϵ_decay::Int
    update_count::Int
    Q̂::QDiscriminator{AS,OS,T,O,A,E}
    buffer::MultiStepDynamicBuffer{O,A}
    observation::Union{Nothing,O}
    done::Bool
    action::Union{Nothing,A}
    rng::MersenneTwister
end

function SARSADiscountedDiscriminatorAgent(::Type{O}, actions::StaticArray{Tuple{AS},A,1}, steps, n, forgetting_factor, discount, ϵ_start, ϵ_end, ϵ_decay, encoder::E; seed::Union{Nothing,UInt}=nothing) where {AS,OS,T,O <: StaticArray{Tuple{OS},T,1},A,E <: AbstractEncoder{T}}
    !isnothing(seed) && seed < 0 && throw(DomainError(seed, "Seed must be non-negative"))
    rng = isnothing(seed) ? MersenneTwister() : MersenneTwister(seed)

    n < 1 && throw(DomainError(n, "Tuple size must be at least 1"))
    # size < n && throw(DomainError(size, "Input size may not be smaller then tuple size"))
    !(0 ≤ discount ≤ 1) && throw(DomainError(regressor_discount, "Discount must lie in the [0, 1] interval"))

    Q̂ = QDiscriminator(O, actions, n, forgetting_factor, encoder, seed)
    buffer = MultiStepDynamicBuffer{O,A}()

    SARSADiscountedDiscriminatorAgent{AS,OS,T,O,A,E}(steps, actions, discount, ϵ_start, ϵ_end, ϵ_decay, 0, Q̂, buffer, nothing, false, nothing, rng)
end

function SARSADiscountedDiscriminatorAgent(
    env, encoder::E; steps, tuple_size, forgetting_factor::Float64=1.0, discount::Float64=1.0, epsilon_start::Float64=0.01, epsilon_end::Float64=0.0, epsilon_decay::Int=1000, seed::Union{Nothing,UInt}=nothing) where {T <: Real,E <: AbstractEncoder{T}}
    # SARSADiscountedDiscriminatorAgent{observation_type(env),action_type(env),T,E}(steps, action_set(env), n, observation_length(env), forget, gamma, epsilon, encoder; seed)
    SARSADiscountedDiscriminatorAgent(observation_type(env), action_set(env), steps, tuple_size, forgetting_factor, discount, epsilon_start, epsilon_end, epsilon_decay, encoder; seed)
end

function observe!(agent::SARSADiscountedDiscriminatorAgent{AS,OS,T,O,A,E}, observation::O) where {AS,OS,T,O,A,E} # {O,A <: Real,T <: Real,E <: AbstractEncoder{T}}
    agent.observation = observation
    agent.done = false
    agent.action = _select_action(agent, observation)
    
    nothing
end

# TODO: All observe! methods are the same for all agents. Generalize.
# TODO: This method does not need to take in the action
function observe!(agent::SARSADiscountedDiscriminatorAgent{AS,OS,T,O,A,E}, action::A, reward::Float64, observation::O, done::Bool) where {AS,OS,T,O,A,E} # {O,A <: Real,T <: Real,E <: AbstractEncoder{T}}
    add!(agent.buffer, agent.observation, agent.action, reward)

    if !done
        agent.action = _select_action(agent, observation)
    end

    agent.observation = observation
    agent.done = done
    nothing
end

function epsilon(agent::SARSADiscountedDiscriminatorAgent)
    α = agent.update_count / agent.ϵ_decay
    (1 - α) * agent.ϵ_start + α * agent.ϵ_end
end

function _select_action(agent::SARSADiscountedDiscriminatorAgent{AS,OS,T,O,A,E}, observation::O) where {AS,OS,T,O,A,E}
    if rand(agent.rng) < epsilon(agent)
        return rand(agent.rng, agent.actions)
    end
        
    return select_action(agent.Q̂, observation)
end


function select_action!(agent::SARSADiscountedDiscriminatorAgent{AS,OS,T,O,A,E}, observation::O) where {AS,OS,T,O,A,E}
    if !isnothing(agent.action)
        return agent.action
    end

    return _select_action(agent, observation)
end

function update!(agent::SARSADiscountedDiscriminatorAgent)
    if agent.done
        G = 0.0
        for transition in Iterators.reverse(agent.buffer)
            G = transition.reward + agent.discount * G

            update!(agent.Q̂, transition.observation, transition.action, G)
        end

        reset!(agent.buffer)
    elseif length(agent.buffer) == agent.steps
        # next_action = select_action!(agent, agent.observation)

        G = agent.Q̂(agent.observation, agent.action)

        for transition in Iterators.reverse(agent.buffer)
            G = transition.reward + agent.discount * G
        end

        t = popfirst!(agent.buffer)

        update!(agent.Q̂, t.observation, t.action, G)
    end

    # agent.action = nothing

    nothing
end

function reset!(agent::SARSADiscountedDiscriminatorAgent; seed::Union{Nothing,UInt}=nothing)
    if !isnothing(seed)
        if seed ≥ 0
        Random.seed!(agent.rng, seed)
        else
            throw(DomainError(seed, "Seed must be non-negative"))
        end
    end
    
    reset!(agent.Q̂; seed)
    reset!(agent.buffer)

    nothing
end