import ..update!, ..select_action, ..reset!

using Random
using StaticArrays

using Ramnet
using Ramnet.Models.AltDiscriminators:FunctionalDiscriminator
using Ramnet.Optimizers
import Ramnet.Loss

struct BinaryActionPolicyLoss <: Loss.AbstractLoss end

# Policy approximator with only two admissible actions: {-1, 1}.
struct BinaryActionPolicy{OS,T <: Real,O <: StaticArray{Tuple{OS},T,1},E <: AbstractEncoder{T}}
    opt::FunctionalOptimizer{BinaryActionPolicyLoss}
    f::FunctionalDiscriminator{1,T,E}
    rng::MersenneTwister
end

function Loss.grad(::BinaryActionPolicyLoss, p::BinaryActionPolicy{OS,T,O,E}, observation::O, action::Int, G) where {OS,T,O,E}
    sign(action) * (1 - probability(p, action, observation)) * G
end

function BinaryActionPolicy(::Type{O}, n::Int, encoder::E; start_learning_rate::Float64=0.1, end_learning_rate::Float64=1e5, learning_rate_decay::Int=1000, epochs::Int=1, partitioner::Symbol=:uniform_random, seed::Union{Nothing,UInt}=nothing) where {OS,T <: Real,O <: StaticArray{Tuple{OS},T,1},E <: AbstractEncoder{T}}
    BinaryActionPolicy{OS,T,O,E}(
        FunctionalOptimizer(BinaryActionPolicyLoss(); start_learning_rate, end_learning_rate, learning_rate_decay, epochs),
        FunctionalDiscriminator{1}(OS, n, encoder, partitioner; seed),
        isnothing(seed) ? MersenneTwister() : MersenneTwister(seed)
    )
end

# Compute π(action | observation)
function probability(p::BinaryActionPolicy{OS,T,O,E}, action::Int, observation::O) where {OS,T <: Real,O <: StaticArray{Tuple{OS},T,1},E <: AbstractEncoder{T}}
    return 1 / (1 + exp(sign(action) * predict(p.f, observation)[1]))
end

# TODO: Validate action (must be ether -1 or +1)
function update!(p::BinaryActionPolicy{OS,T,O,E}, observation::O, action::Int, G) where {OS,T <: Real,O <: StaticArray{Tuple{OS},T,1},E <: AbstractEncoder{T}}
    gradient = Optimizers.grad(p.opt, p, observation, action, G)

    train!(p.f, observation, learning_rate(p.opt) * gradient)

    return nothing
end

function select_action(p::BinaryActionPolicy{OS,T,O,E}, observation::O) where {OS,T <: Real,O <: StaticArray{Tuple{OS},T,1},E <: AbstractEncoder{T}}
    return rand(p.rng) < probability(p, 1, observation) ? 1 : -1
    end

function reset!(p::BinaryActionPolicy; seed::Union{Nothing,UInt}=nothing)
    Ramnet.reset!(p.opt)
    Ramnet.reset!(p.f; seed)
    Random.seed!(p.rng, seed)

    return nothing
end
