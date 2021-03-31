import ..update!, ..select_action, ..reset!

using StaticArrays
using Random
using LinearAlgebra:Diagonal

using Ramnet.Models.AltDiscriminators:FunctionalDiscriminator
using Ramnet.Optimizers
import Ramnet.Loss

struct ContinousActionPolicyLoss <: Loss.AbstractLoss end

# Policy approximator with only two admissible actions: {-1, 1}.
struct ContinousActionPolicy{OS,AS,T <: Real,O <: StaticArray{Tuple{OS},T,1},A <: StaticArray{Tuple{AS},T,1},E <: AbstractEncoder{T},C <: AbstractMatrix{T}}
    opt::FunctionalOptimizer{ContinousActionPolicyLoss}
    f::FunctionalDiscriminator{AS,T,E}
    # Covariance diagonal matrix
    cov::C
    inv_cov::C
    rng::MersenneTwister
end

function ContinousActionPolicy(::Type{O}, ::Type{A}, n::Int, encoder::E; cov::C, η::Float64=0.1, epochs::Int=1, partitioner::Symbol=:uniform_random, seed::Union{Nothing,Int}=nothing) where {AS,OS,T <: Real,O <: StaticArray{Tuple{OS},T,1},A <: StaticArray{Tuple{AS},T,1},E <: AbstractEncoder{T},C <: AbstractMatrix{T}}
    ContinousActionPolicy{OS,AS,T,O,A,E,C}(
        FunctionalOptimizer(ContinousActionPolicyLoss(); learning_rate=η, epochs),
        FunctionalDiscriminator{AS}(OS, n, encoder, partitioner; seed),
        cov,
        inv(cov),
        isnothing(seed) ? MersenneTwister() : MersenneTwister(seed)
    )
end

function Loss.grad(::ContinousActionPolicyLoss, p::ContinousActionPolicy{OS,AS,T,O,A,E,C}, observation::O, action::A, G) where {OS,AS,T,O,A,E,C}
    # p.inv_cov * (action - predict(p.f, observation)) * G
    p.inv_cov * (predict(p.f, observation) - action) * G
end

# TODO: Validate action (must be ether -1 or +1)
function update!(p::ContinousActionPolicy{OS,AS,T,O,A,E,C}, observation::O, action::A, G) where {OS,AS,T,O,A,E,C}
    gradient = Optimizers.grad(p.opt, p, observation, action, G)

    train!(p.f, observation, learning_rate(p.opt) * gradient)
    
    return nothing
end

# TODO: The clamping of the action is specific to the hopper agent!
function select_action(p::ContinousActionPolicy{OS,AS,T,O,A,E,C}, observation::O) where {OS,AS,T,O,A,E,C}
    return A(p.cov * randn(p.rng, AS) + predict(p.f, observation))
end

function reset!(p::ContinousActionPolicy)
    Ramnet.reset!(p.f)

    return nothing
end
