using Ramnet.Encoders
using Ramnet

using Random

using ..Buffers: MultiStepDynamicBuffer, add!

mutable struct MonteCarloDiscountedDiscriminatorAgent{O <: AbstractVector,A <: Real,E <: AbstractEncoder} <: AbstractAgent{O,A}
    actions::UnitRange{A}
    n::Int
    size::Int
    regressor_discount::Float64
    γ::Float64
    ϵ::Float64
    # episodes::Int
    encoder::E
    Q̂::Dict{A,FastRegressionDiscriminator}
    buffer::MultiStepDynamicBuffer{O,A}
    observation::Union{Nothing,O}
    done::Bool
    action::Union{Nothing,A}
    rng::MersenneTwister

    # TODO: Validate actions and encoder and ϵ
    # TODO: Do I need the size if the encoder is given? Not if a knew the observation length (static array?)
    function MonteCarloDiscountedDiscriminatorAgent{O,A,E}(actions, n, obs_size, regressor_discount, γ, ϵ, encoder::E; seed::Union{Nothing,Int}=nothing) where {O,A <: Real,E <: AbstractEncoder}
        !isnothing(seed) && seed < 0 && throw(DomainError(seed, "Seed must be non-negative"))
        rng = isnothing(seed) ? MersenneTwister() : MersenneTwister(seed)

        n < 1 && throw(DomainError(n, "Tuple size must be at least 1"))
        # size < n && throw(DomainError(size, "Input size may not be smaller then tuple size"))
        size = obs_size * resolution(encoder)
        # !(0 ≤ discount ≤ 1) && throw(DomainError(discount, "Discount must lie in the [0, 1] interval"))
        # episodes < 1 && throw(DomainError(episodes, "Number of episodes must be greater then one"))

        Q̂ = Dict(a => FastRegressionDiscriminator(size, n; γ=regressor_discount) for a in actions)
        buffer = MultiStepDynamicBuffer{O,A}()

        new(actions, n, size, regressor_discount, γ, ϵ, encoder, Q̂, buffer, nothing, false, nothing, rng)
    end
end

function MonteCarloDiscountedDiscriminatorAgent{O,A,E}(
    actions, obs_size, encoder::E; n, regressor_discount, gamma, epsilon, seed) where {O,A <: Real,E <: AbstractEncoder}
    MonteCarloDiscountedDiscriminatorAgent{O,A,E}(actions, n, obs_size, regressor_discount, gamma, epsilon, encoder; seed)
end

function observe!(agent::MonteCarloDiscountedDiscriminatorAgent{O,A,E}, observation::O) where {O,A <: Real,E <: AbstractEncoder}
    agent.observation = observation
    agent.done = false
    
    nothing
end

# TODO: All observe! methods are the same for all agents. Generalize.
# TODO: This method does not need to take in the action
function observe!(agent::MonteCarloDiscountedDiscriminatorAgent{O,A,E}, action::A, reward::Float64, observation::O, done::Bool) where {O,A <: Real,E <: AbstractEncoder}
    add!(agent.buffer, agent.observation, agent.action, reward)

    agent.observation = observation
    agent.done = done

    nothing
end

function update!(agent::MonteCarloDiscountedDiscriminatorAgent)
    if agent.done
        G = 0.0
        for transition in Iterators.reverse(agent.buffer)
            G = transition.reward + agent.γ * G
            train!(
                agent.Q̂[transition.action],
                encode(agent.encoder, transition.observation),
                G
            )
        end

        reset!(agent.buffer)
    end

    nothing
end

function Agents.reset!(agent::MonteCarloDiscountedDiscriminatorAgent)
    for action in agent.actions
        agent.Q̂[action] = FastRegressionDiscriminator(agent.size, agent.n; γ=agent.regressor_discount)
    end

    reset!(agent.buffer)

    nothing
end