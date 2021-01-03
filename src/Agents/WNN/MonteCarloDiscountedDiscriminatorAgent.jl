using Ramnet.Encoders
using Ramnet

using Random
using StatsBase: sample, pweights

using ..ExperienceBuffers: DynamicBinaryBuffer, add!, reset!

struct MonteCarloDiscountedDiscriminatorAgent{A <: Real,E <: AbstractEncoder} <: AbstractAgent
    actions::UnitRange{A}
    n::Int
    size::Int
    discount::Float64
    ϵ::Float64
    encoder::E
    Q̂::Dict{A,RegressionDiscriminator}
    buffer::DynamicBinaryBuffer{A}
    rng::MersenneTwister

    # TODO: Validate actions and encoder and ϵ
    # TODO: Do I need the size if the encoder is given?
    function MonteCarloDiscountedDiscriminatorAgent{A,E}(actions, n, size, discount, ϵ, encoder::E; seed=Union{Nothing,Int} = nothing) where {A <: Real,E <: AbstractEncoder}
        !isnothing(seed) && seed < 0 && throw(DomainError(seed, "Seed must be non-negative"))
        rng = isnothing(seed) ? MersenneTwister() : MersenneTwister(seed)

        n < 1 && throw(DomainError(n, "Tuple size must be at least 1"))
        size < n && throw(DomainError(size, "Input size may not be smaller then tuple size"))
        !(0 ≤ discount ≤ 1) && throw(DomainError(discount, "Discount must lie in the [0, 1] interval"))

        Q̂ = Dict(a => RegressionDiscriminator(size, n; γ=discount) for a in actions)
        buffer = DynamicBinaryBuffer{A}()

        new(actions, n, size, discount, ϵ, encoder, Q̂, buffer, rng)
    end
end

function MonteCarloDiscountedDiscriminatorAgent(actions::UnitRange{A}, n, size, discount, ϵ, encoder::E; seed) where {A <: Real,E <: AbstractEncoder}
    MonteCarloDiscountedDiscriminatorAgent{A,E}(actions, n, size, discount, ϵ, encoder; seed)
end


function select_action(agent::MonteCarloDiscountedDiscriminatorAgent{A,E}, observation) where {A <: Real,E <: AbstractEncoder}
    # Random action taken with ϵ probability
    rand(agent.rng) < agent.ϵ && return rand(agent.rng, agent.actions)

    # TODO: I could avoid creating this temporary arrays here
    # Selecting action ta maximizes Q̂, with ties broken randomly
    q_values = similar(agent.actions, Float64)
    # action_probs = similar(agent.actions, Float64)
    q_max = typemin(Float64)
    q = 0.0
    for (i, action) in Iterators.zip(eachindex(q_values), agent.actions)
        q = predict(agent.Q̂[action], encode(agent.encoder, observation)) # TODO: Don't have to create a bunch of temp arrays here
        q_values[i] = q
        q_max = q > q_max ? q : q_max
    end

    action_probs = q_values .== q_max
    # action_probs /= sum(action_probs)

    return sample(agent.rng, agent.actions, pweights(action_probs))
end

function observe!(agent::MonteCarloDiscountedDiscriminatorAgent, observation)
    add!(agent.buffer, encode(agent.encoder, observation))  

    nothing
end

function observe!(agent::MonteCarloDiscountedDiscriminatorAgent{A,E}, action::A, reward::Float64, observation, done::Bool) where {A <: Real,E <: AbstractEncoder}
    add!(agent.buffer, action, reward, encode(agent.encoder, observation), done)

    nothing
end

function update!(agent::MonteCarloDiscountedDiscriminatorAgent)
    if agent.buffer.terminals[end] == true
        target = 0.0
        for (obs, action, reward, next_obs, done) in Iterators.reverse(agent.buffer)
            target += reward
            train!(agent.Q̂[action], obs, target)
        end

        reset!(agent.buffer)
    end

    nothing
end