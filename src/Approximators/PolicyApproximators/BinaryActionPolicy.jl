import ..update!, ..select_action, ..reset!

using StaticArrays

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

function BinaryActionPolicy(::Type{O}, n::Int, encoder::E; η::Float64=0.1, epochs::Int=1, partitioner::Symbol=:uniform_random, seed::Union{Nothing,Int}=nothing) where {OS,T <: Real,O <: StaticArray{Tuple{OS},T,1},E <: AbstractEncoder{T}}
    BinaryActionPolicy{OS,T,O,E}(
        FunctionalOptimizer(BinaryActionPolicyLoss(); learning_rate=η, epochs),
        FunctionalDiscriminator{1}(OS, n, encoder, partitioner; seed),
        isnothing(seed) ? MersenneTwister() : MersenneTwister(seed)
    )
end

# Compute π(action | observation)
function probability(p::BinaryActionPolicy{OS,T,O,E}, action::Int, observation::O) where {OS,T <: Real,O <: StaticArray{Tuple{OS},T,1},E <: AbstractEncoder{T}}
    prob = 1 / (1 + exp(predict(p.f, observation)[1]))

    return action == 1 ? prob : 1 - prob
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

function reset!(p::BinaryActionPolicy{OS,T,O,E}) where {OS,T <: Real,O <: StaticArray{Tuple{OS},T,1},E <: AbstractEncoder{T}}
    Ramnet.reset!(p.f)

    return nothing
end
