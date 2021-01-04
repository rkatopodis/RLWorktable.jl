mutable struct SARSADiscountedDiscriminatorAgent{O,A <: Real,E <: AbstractEncoder} <: AbstractAgent
    actions::UnitRange{A}
    n::Int
    size::Int
    discount::Float64
    γ::Float64
    ϵ::Float64
    encoder::E
    Q̂::Dict{A,RegressionDiscriminator}
    s::Union{Nothing,O}
    a::Union{Nothing,A}
    r::Float64
    s′::Union{Nothing,O}
    a′::Union{Nothing,A}
    done::Bool
    rng::MersenneTwister

    function SARSADiscountedDiscriminatorAgent{O,A,E}(actions, n, size, discount, γ, ϵ, encoder::E; seed=Union{Nothing,Int} = nothing) where {O,A <: Real,E <: AbstractEncoder}
        !isnothing(seed) && seed < 0 && throw(DomainError(seed, "Seed must be non-negative"))
        rng = isnothing(seed) ? MersenneTwister() : MersenneTwister(seed)

        n < 1 && throw(DomainError(n, "Tuple size must be at least 1"))
        size < n && throw(DomainError(size, "Input size may not be smaller then tuple size"))
        !(0 ≤ discount ≤ 1) && throw(DomainError(discount, "Discount must lie in the [0, 1] interval"))

        Q̂ = Dict(a => RegressionDiscriminator(size, n; γ=discount) for a in actions)
        buffer = DynamicBinaryBuffer{A}()

        new(
            actions,
            n,
            size,
            discount,
            γ,
            ϵ,
            encoder,
            Q̂,
            nothing,
            nothing,
            zero(Float64),
            nothing,
            nothing,
            false,
            rng
        )
    end
end

# function SARSADiscountedDiscriminatorAgent(actions::UnitRange{A}, n, size, discount, γ, ϵ, encoder::E; seed=nothing) where {A <: Real,E <: AbstractEncoder}
#     SARSADiscountedDiscriminatorAgent{A,E}(actions, n, size, discount, γ, ϵ, encoder; seed)
# end

# TODO: This is the same as the MC select_action. Consolidate.
function _select_action(agent::SARSADiscountedDiscriminatorAgent, observation)
    # Random action taken with ϵ probability
    rand(agent.rng) < agent.ϵ && return rand(agent.rng, agent.actions)

    # TODO: I could avoid creating this temporary arrays here
    # Selecting action that maximizes Q̂, with ties broken randomly
    q_values = similar(agent.actions, Float64)
    q_max = typemin(Float64)
    q = 0.0
    for (i, action) in Iterators.zip(eachindex(q_values), agent.actions)
        q = predict(agent.Q̂[action], encode(agent.encoder, observation)) # TODO: Don't have to create a bunch of temp arrays here
        q_values[i] = q
        q_max = q > q_max ? q : q_max
    end

    action_probs = q_values .== q_max

    return sample(agent.rng, agent.actions, pweights(action_probs))
end

function select_action(agent::SARSADiscountedDiscriminatorAgent, observation)
    if !isnothing(agent.a′)
        return agent.a′
    else
        _select_action(agent, observation)
    end
end

function observe!(agent::SARSADiscountedDiscriminatorAgent{O,A,E}, observation::O) where {O,A <: Real,E <: AbstractEncoder}
    agent.s = observation

    nothing
end

function observe!(agent::SARSADiscountedDiscriminatorAgent{O,A,E}, action::A, reward::Float64, observation::O, done::Bool) where {O,A <: Real,E <: AbstractEncoder}
    agent.a = action
    agent.r = reward
    agent.s′ = observation
    agent.done = done

    nothing
end

function update!(agent::SARSADiscountedDiscriminatorAgent)
    if agent.done
        train!(agent.Q̂[agent.a], encode(agent.encoder, agent.s), agent.r)
    else
        agent.a′ = _select_action(agent, agent.s′)

        train!(
            agent.Q̂[agent.a],
            encode(agent.encoder, agent.s),
            agent.r + agent.γ * predict(
                agent.Q̂[agent.a′],
                encode(agent.encoder, agent.s′)
            )
        )

        agent.s = agent.s′
    end

    nothing
end

function Agents.reset!(agent::SARSADiscountedDiscriminatorAgent)
    for action in agent.actions
        agent.Q̂[action] = RegressionDiscriminator(agent.size, agent.n; γ=agent.discount)
    end

    agent.s = nothing
    agent.s′ = nothing
    agent.a = nothing
    agent.a′ = nothing

    nothing
end