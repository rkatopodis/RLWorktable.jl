using Ramnet.Encoders
using Ramnet

using ..Buffers: MultiStepDynamicBuffer, add!

mutable struct ExpectedSARSADiscriminatorAgent{O <: AbstractVector,A <: Real,E <: AbstractEncoder} <: AbstractAgent{O,A}
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

    function ExpectedSARSADiscriminatorAgent{O,A,E}(steps, actions, n, size, regressor_discount, rl_discount, ϵ, encoder::E; seed::Union{Nothing,Int}=nothing) where {O <: AbstractVector,A <: Real,E <: AbstractEncoder}
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

function observe!(agent::ExpectedSARSADiscriminatorAgent{O,A,E}, observation::O) where {O <: AbstractVector,A <: Real,E <: AbstractEncoder}
    agent.observation = observation
    agent.done = false
    
    nothing
end

function observe!(agent::ExpectedSARSADiscriminatorAgent{O,A,E}, action::A, reward::Float64, observation::O, done::Bool) where {O <: AbstractVector,A <: Real,E <: AbstractEncoder}
    add!(agent.buffer, agent.observation, agent.action, reward)

    agent.observation = observation
    agent.done = done

    nothing
end

function update!(agent::ExpectedSARSADiscriminatorAgent)
    if agent.done
        G = 0.0
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
        G = expected_q_value(agent, agent.observation)
        for transition in Iterators.reverse(agent.buffer)
            G = transition.reward + agent.rl_discount * G
        end

        t = popfirst!(agent.buffer)

        train!(
            agent.Q̂[t.action],
            encode(agent.encoder, t.observation),
            G
        )
    end

    nothing
end

function Agents.reset!(agent::ExpectedSARSADiscriminatorAgent)
    for action in agent.actions
        agent.Q̂[action] = RegressionDiscriminator(agent.size, agent.n; γ=agent.regressor_discount)
    end

    nothing
end