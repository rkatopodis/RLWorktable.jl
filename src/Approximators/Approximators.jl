module Approximators

using Random
using StatsBase: mean, pweights

using Ramnet
using Ramnet.Encoders

include("PolicyApproximators/BinaryActionPolicy.jl")
export BinaryActionPolicy

struct DiscreteQDiscriminator{O <: AbstractVector,A <: Real,E <: AbstractEncoder}
    action_range::UnitRange{A}
    discount::Float64
    encoder::E
    Q::Dict{A,RegressionDiscriminator}

    # TODO: It's annoying to have to specify the input size. It requires knowledge
    #       of the encoder's resolution and the length of the state of the environment.
    #       Maybe it would be better to have a type that specifies the characteristics
    #       of the environment, such as action and observation spaces.
    function DiscreteQDiscriminator{O,A,E}(action_range::UnitRange{A}, size::Int, n::Int, discount::Float64, encoder::E) where {O <: AbstractVector,A <: Real,E <: AbstractEncoder}
        n < 1 && throw(DomainError(n, "Tuple size must be at least 1"))
        size < n && throw(DomainError(size, "Input size may not be smaller then tuple size"))
        !(0 ≤ discount ≤ 1) && throw(DomainError(discount, "Discount must lie in the [0,1] interval"))

        Q = Dict(a => RegressionDiscriminator(size, n; γ=discount) for a in action_range)

        new(action_range, discount, encoder, Q)
    end
end

# TODO: Properly check if observation and action are valid
function (M::DiscreteQDiscriminator{O,A,E})(observation::O, action::A) where {O <: AbstractVector,A <: Real,E <: AbstractEncoder}
    return predict(M.Q[action], encode(M.encoder, observation))
end

# TODO: Properly check if observation is valid
function q_values!(M::DiscreteQDiscriminator{O,A,E}, observation::O, q_values::Vector{Float64}) where {O <: AbstractVector,A <: Real,E <: AbstractEncoder}
    length(q_values) != length(M.action_range) && throw(
      DimensionMismatch("Expected vector of length $(length(M.action_range)), got $(length(q_values))"))

    q_max = typemin(Float64)
    q = 0.0
    encoded_obs = encode(M.encoder, observation) # TODO: Don't have to create a bunch of temp arrays here
    for (i, action) in Iterators.zip(eachindex(q_values), M.action_range)
        q = predict(M.QQ[action], encoded_obs)
        dest[i] = q
        q_max = q > q_max ? q : q_max
    end

    return q_max
end

function (M::DiscreteQDiscriminator{O,A,E})(observation::O) where {O <: AbstractVector,A <: Real,E <: AbstractEncoder}
    values = similar(M.action_range, Float64)
    q_values!(M, observation, values)

return values
end

function expected_q_value(M::DiscreteQDiscriminator{O,A,E}, observation::O; ϵ::Float64=0.0) where {O <: AbstractVector,A <: Real,E <: AbstractEncoder}
    values = similar(M.action_range, Float64)
    q_max = q_values!(M, observation, values)

    return mean(values, pweights((values .== q_max) .+ ϵ))
end

# function select_greedy_action(M::DiscreteQDiscriminator{O,A,E}, observation, rng::MersenneTwister; ϵ::Float64=0.0 ) where {A <: AbstractAgent}
#     # TODO: I could avoid creating these temporary arrays here
#     values = similar(M.action_range, Float64)
#     q_max = q_values!(M, observation, values)

#     if rand(agent.rng) < ϵ # Random action taken with ϵ probability
#         agent.action = rand(agent.rng, agent.actions)
#     else
#       # Selecting action that maximizes Q̂, with ties broken randomly
#         agent.action = sample(agent.rng, agent.actions, pweights(values .== q_max))
#     end

#     return agent.action
# end


end
