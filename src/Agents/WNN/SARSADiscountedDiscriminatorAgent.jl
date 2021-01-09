using Ramnet.Encoders
using Ramnet

using ..Buffers: MultiStepDynamicBuffer, add!

mutable struct SARSADiscountedDiscriminatorAgent{O,A <: Real,E <: AbstractEncoder} <: AbstractAgent
    steps::Int
    actions::UnitRange{A}
    n::Int
    size::Int
    regressor_discount::Float64
    rl_discount::Float64
    ϵ::Float64
    encoder::E
    Q̂::Dict{A,RegressionDiscriminator}
    buffer::MultiStepDynamicBuffer{O,A}
    observation::Union{Nothing,O}
    done::Bool
    action::Union{Nothing,A}
    rng::MersenneTwister

    function SARSADiscountedDiscriminatorAgent{O,A,E}(steps, actions, n, size, regressor_discount, rl_discount, ϵ, encoder::E; seed::Union{Nothing,Int}=nothing) where {O,A <: Real,E <: AbstractEncoder}
        !isnothing(seed) && seed < 0 && throw(DomainError(seed, "Seed must be non-negative"))
        rng = isnothing(seed) ? MersenneTwister() : MersenneTwister(seed)

        n < 1 && throw(DomainError(n, "Tuple size must be at least 1"))
        size < n && throw(DomainError(size, "Input size may not be smaller then tuple size"))
        !(0 ≤ regressor_discount ≤ 1) && throw(DomainError(regressor_discount, "Discount must lie in the [0, 1] interval"))

        Q̂ = Dict(a => RegressionDiscriminator(size, n; γ=regressor_discount) for a in actions)
        buffer = MultiStepDynamicBuffer{O,A}()

        new(
            steps,
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
            rng
        )
    end
end

# TODO: This is the same as the MC select_action. Consolidate.
function _select_action(agent::SARSADiscountedDiscriminatorAgent, observation)
    # Random action taken with ϵ probability
    rand(agent.rng) < agent.ϵ && return rand(agent.rng, agent.actions)

    # TODO: I could avoid creating these temporary arrays here
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

    return sample(agent.rng, agent.actions, pweights(action_probs))
end

function select_action(agent::SARSADiscountedDiscriminatorAgent, observation)
    # return agent.action
    agent.action = _select_action(agent, observation)
    return agent.action
end

function observe!(agent::SARSADiscountedDiscriminatorAgent{O,A,E}, observation::O) where {O,A <: Real,E <: AbstractEncoder}
    agent.observation = observation
    agent.done = false
    # agent.action = _select_action(agent, observation)
    
    nothing
end

function observe!(agent::SARSADiscountedDiscriminatorAgent{O,A,E}, action::A, reward::Float64, observation::O, done::Bool) where {O,A <: Real,E <: AbstractEncoder}
    add!(agent.buffer, agent.observation, agent.action, reward)

    agent.observation = observation
    agent.done = done

    # if !done
    #     agent.action = _select_action(agent, observation)
    # end

    nothing
end

function update!(agent::SARSADiscountedDiscriminatorAgent)
    if agent.done
        # println("Monte Carlo update")
        G = 0.0
        # println("$(length(agent.buffer)) transitions")
        for transition in Iterators.reverse(agent.buffer)
            G = transition.reward + agent.rl_discount * G
            train!(
                agent.Q̂[transition.action],
                encode(agent.encoder, transition.observation),
                G
            )
        end

        reset!(agent.buffer)
    elseif length(agent.buffer) == agent.steps
        # println("Bootstrapping update")
        next_action = _select_action(agent, agent.observation)
        G = predict(agent.Q̂[next_action], encode(agent.encoder, agent.observation))
        for transition in Iterators.reverse(agent.buffer)
            G = transition.reward + agent.rl_discount * G
        end

        t = popfirst!(agent.buffer)

        train!(
            agent.Q̂[t.action],
            encode(agent.encoder, t.observation),
            G
        )
    else
        # println("No update!")
    end

    # for transition in agent.buffer
    #     if agent.done
    #         train!(
    #             agent.Q̂[transition.action],
    #             encode(agent.encoder, transition.observation),
    #             transition.reward
    #         )
    #     else
    #         train!(
    #             agent.Q̂[transition.action],
    #             encode(agent.encoder, transition.observation),
    #             transition.reward + agent.rl_discount * predict(
    #                 agent.Q̂[agent.past_action],
    #                 encode(agent.encoder, agent.past_obs)
    #             )
    #         )
    #     end
    # end

    nothing
end

function Agents.reset!(agent::SARSADiscountedDiscriminatorAgent)
    for action in agent.actions
        agent.Q̂[action] = RegressionDiscriminator(agent.size, agent.n; γ=agent.regressor_discount)
    end

    # agent.past_obs = nothing
    # agent.past_action = nothing

    nothing
end