import ..update!, ..select_action, ..reset!

using Random
using StaticArrays

using Ramnet
using Ramnet.Models.AltDiscriminators: FunctionalDiscriminator
using Ramnet.Optimizers
import Ramnet.Loss

struct DiscreteActionPolicyLoss <: Loss.AbstractLoss end

struct DiscreteActionPolicy{OS,AS,T<:Real,O<:StaticArray{Tuple{OS},T,1},E<:AbstractEncoder{T}}
    opt::FunctionalOptimizer{DiscreteActionPolicyLoss}
    f::FunctionalDiscriminator{AS,T,E}
    rng::MersenneTwister
end

function DiscreteActionPolicy{AS}(::Type{O}, n::Int, encoder::E; start_learning_rate::Float64 = 0.1, end_learning_rate::Float64 = 1e5, learning_rate_decay::Int = 1000, epochs::Int = 1, partitioner::Symbol = :uniform_random, seed::Union{Nothing,UInt} = nothing) where {AS,OS,T<:Real,O<:StaticArray{Tuple{OS},T,1},E<:AbstractEncoder{T}}
    DiscreteActionPolicy{OS,AS,T,O,E}(
        FunctionalOptimizer(DiscreteActionPolicyLoss(); start_learning_rate, end_learning_rate, learning_rate_decay, epochs),
        FunctionalDiscriminator{AS}(OS, n, encoder, partitioner; seed),
        isnothing(seed) ? MersenneTwister() : MersenneTwister(seed)
    )
end

function Loss.grad(::DiscreteActionPolicyLoss, p::DiscreteActionPolicy{OS,AS,T,O,E}, observation::O, action::Int, G) where {OS,AS,T,O,E}
    probs = probabilities(p, observation)
    probs[action+1] -= 1
    return probs * G
end

function probabilities(p::DiscreteActionPolicy{OS,AS,T,O,E}, observation::O) where {OS,AS,T,O,E}
    logits = predict(p.f, observation)
    return exp.(logits) / sum(exp.(logits))
end

function probability(p::DiscreteActionPolicy{OS,AS,T,O,E}, action::Int, observation::O) where {OS,AS,T,O,E}
    return probabilities(p, observation)[action+1]
end

function mode(p::DiscreteActionPolicy{OS,AS,T,O,E}, observation::O) where {OS,AS,T,O,E}
    nothing
end

function update!(p::DiscreteActionPolicy{OS,AS,T,O,E}, observation::O, action::Int, G) where {OS,AS,T,O,E}
    gradient = Optimizers.grad(p.opt, p, observation, action, G)

    train!(p.f, observation, learning_rate(p.opt) * gradient)

    return nothing
end

function select_action(p::DiscreteActionPolicy{OS,AS,T,O,E}, observation::O) where {OS,AS,T,O,E}
    cdf = cumsum(probabilities(p, observation))
    s = rand(p.rng)

    # Linear search. Might be too expensive if number of possible actions is big
    for i in eachindex(cdf)
        if s < cdf[i]
            return i - 1
        end
    end
end

function reset!(p::DiscreteActionPolicy; seed::Union{Nothing,UInt} = nothing)
    Ramnet.reset!(p.opt)
    Ramnet.reset!(p.f; seed)
    Random.seed!(p.rng, seed)

    return nothing
end