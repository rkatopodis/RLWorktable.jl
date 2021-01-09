using Ramnet.Encoders
using Ramnet

using ..Buffers: SingleStepBuffer, add!, ready

mutable struct QLearningDiscountedDiscriminatorAgent{O,A <: Real,E <: AbstractEncoder} <: AbstractAgent
    actions::UnitRange{A}
    n::Int
    size::Int
    regressor_discount::Float64
    rl_discount::Float64
    ϵ::Float64
    encoder::E
    Q̂::Dict{A,RegressionDiscriminator}
    buffer::SingleStepBuffer{O,A}
    past_obs::Union{Nothing,O}
    done::Bool
    past_action::Union{Nothing,A}
    q_max::Float64
    rng::MersenneTwister

    function QLearningDiscountedDiscriminatorAgent{O,A,E}(actions, n, size, regressor_discount, rl_discount, ϵ, encoder::E; seed=Union{Nothing,Int} = nothing) where {O,A <: Real,E <: AbstractEncoder}
        !isnothing(seed) && seed < 0 && throw(DomainError(seed, "Seed must be non-negative"))
        rng = isnothing(seed) ? MersenneTwister() : MersenneTwister(seed)

        n < 1 && throw(DomainError(n, "Tuple size must be at least 1"))
        size < n && throw(DomainError(size, "Input size may not be smaller then tuple size"))
        !(0 ≤ regressor_discount ≤ 1) && throw(DomainError(regressor_discount, "Discount must lie in the [0, 1] interval"))

        Q̂ = Dict(a => RegressionDiscriminator(size, n; γ=regressor_discount) for a in actions)
        buffer = SingleStepBuffer{O,A}()

        new(
            actions,
            n,
            size,
            regressor_discount,
            rl_discount,
            ϵ,
            encoder,
            Q̂,
            buffer,
            nothing,
            false,
            nothing,
            0.0,
            rng
        )
    end
end

function _select_action(agent::QLearningDiscountedDiscriminatorAgent, observation)
    # Random action taken with ϵ probability
    # rand(agent.rng) < agent.ϵ && return rand(agent.rng, agent.actions)

    # TODO: I could avoid creating this temporary arrays here
    # Selecting action that maximizes Q̂, with ties broken randomly
    q_values = similar(agent.actions, Float64)
    q_max = typemin(Float64)
    q = 0.0
    encoded_obs = encode(agent.encoder, observation) # TODO: Don't have to create a bunch of temp arrays here
    for (i, action) in Iterators.zip(eachindex(q_values), agent.actions)
        q = predict(agent.Q̂[action], encoded_obs)
        q_values[i] = q
        q_max = q > q_max ? q : q_max
    end

    # println("Q-values: $q_values")

    action_probs = q_values .== q_max
    selected_action = rand(agent.rng) < agent.ϵ ? rand(agent.rng, agent.actions) : sample(agent.rng, agent.actions, pweights(action_probs))
    return selected_action, q_max
end

select_action(agent::QLearningDiscountedDiscriminatorAgent, observation) = agent.past_action

function observe!(agent::QLearningDiscountedDiscriminatorAgent{O,A,E}, observation::O) where {O,A <: Real,E <: AbstractEncoder}
    agent.past_obs = observation
    agent.done = false
    agent.past_action, agent.q_max = _select_action(agent, observation)
  
    nothing
end

function observe!(agent::QLearningDiscountedDiscriminatorAgent{O,A,E}, action::A, reward::Float64, observation::O, done::Bool) where {O,A <: Real,E <: AbstractEncoder}
    add!(agent.buffer, agent.past_obs, agent.past_action, reward)

    agent.past_obs = observation
    agent.done = done
    # agent.past_action, agent.q_max = _select_action(agent, observation)

    nothing
end

function update!(agent::QLearningDiscountedDiscriminatorAgent)
    !ready(agent.buffer) && return

    for transition in agent.buffer
        if agent.done
            train!(
              agent.Q̂[transition.action],
              encode(agent.encoder, transition.observation),
              transition.reward
            )
        else
            train!(
              agent.Q̂[transition.action],
              encode(agent.encoder, transition.observation),
              transition.reward + agent.rl_discount * agent.q_max
            )
        end
    end

    if !agent.done
        # @show t = _select_action(agent, agent.past_obs)
        # agent.past_action, agent.q_max = t
        agent.past_action, agent.q_max = _select_action(agent, agent.past_obs)
    end
    # (agent.past_action, agent.q_max) = _select_action(agent, agent.past_obs)

    nothing
end

function Agents.reset!(agent::QLearningDiscountedDiscriminatorAgent)
    for action in agent.actions
        agent.Q̂[action] = RegressionDiscriminator(agent.size, agent.n; γ=agent.discount)
    end

    agent.past_obs = nothing
    agent.past_action = nothing

    nothing
end