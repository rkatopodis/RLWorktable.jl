import ..update!, ..select_action, ..reset!

using StaticArrays
using Random
using LinearAlgebra: Diagonal

using Ramnet.Models.AltDiscriminators: FunctionalDiscriminator
using Ramnet.Optimizers
import Ramnet.Loss

mutable struct ContinousActionPolicyLoss <: Loss.AbstractLoss
    start_cov::Float64
    end_cov::Float64
    cov_decay::Int
    update_count::Int
end

function ContinousActionPolicyLoss(start_cov::Float64, end_cov::Float64, cov_decay::Int)
    ContinousActionPolicyLoss(start_cov, end_cov, cov_decay, 0)
end

# Policy approximator with only two admissible actions: {-1, 1}.
struct ContinousActionPolicy{OS,AS,T<:Real,O<:StaticArray{Tuple{OS},T,1},A<:StaticArray{Tuple{AS},T,1},E<:AbstractEncoder{T}}
    opt::FunctionalOptimizer{ContinousActionPolicyLoss}
    f::FunctionalDiscriminator{AS,T,E}
    rng::MersenneTwister
end

function ContinousActionPolicy(::Type{O}, ::Type{A}, n::Int, encoder::E; start_cov::Float64, end_cov::Float64, cov_decay::Int, start_learning_rate::Float64 = 0.1, end_learning_rate::Float64 = 1e5, learning_rate_decay::Int = 1000, epochs::Int = 1, partitioner::Symbol = :uniform_random, seed::Union{Nothing,UInt} = nothing) where {AS,OS,T<:Real,O<:StaticArray{Tuple{OS},T,1},A<:StaticArray{Tuple{AS},T,1},E<:AbstractEncoder{T}}
    ContinousActionPolicy{OS,AS,T,O,A,E}(
        FunctionalOptimizer(ContinousActionPolicyLoss(start_cov, end_cov, cov_decay); start_learning_rate, end_learning_rate, learning_rate_decay, epochs),
        FunctionalDiscriminator{AS}(OS, n, encoder, partitioner; seed),
        isnothing(seed) ? MersenneTwister() : MersenneTwister(seed)
    )
end

function cov(l::ContinousActionPolicyLoss)
    α = l.update_count / l.cov_decay
    UniformScaling((1 - α) * l.start_cov + α * l.end_cov)
end

function Loss.grad(l::ContinousActionPolicyLoss, action, pred_action)
    inv(cov(l)) * (pred_action - action)
end

function mode(p::ContinousActionPolicy{OS,AS,T,O,A,E}, observation::O) where {OS,AS,T,O,A,E}
    # println(typeof(predict(p.f, observation)))
    return SVector{AS,T}(predict(p.f, observation))
end

function update!(p::ContinousActionPolicy{OS,AS,T,O,A,E}, observation::O, action::A, G) where {OS,AS,T,O,A,E}
    # gradient = Optimizers.grad(p.opt, p, observation, action, G)

    gradient = Optimizers.grad(p.opt, action, predict(p.f, observation))
    train!(p.f, observation, learning_rate(p.opt) * gradient * G)

    p.opt.loss.update_count += 1

    return nothing
end

# TODO: The clamping of the action is specific to the hopper agent!
function select_action(p::ContinousActionPolicy{OS,AS,T,O,A,E}, observation::O; deterministic::Bool = False) where {OS,AS,T,O,A,E}
    if deterministic
        return mode(p, observation)
    end

    return A(cov(p.opt.loss) * randn(p.rng, AS) + predict(p.f, observation))
end

function reset!(p::ContinousActionPolicy; seed::Union{Nothing,UInt} = nothing)
    Ramnet.reset!(p.opt)
    p.opt.loss.update_count = 0
    Ramnet.reset!(p.f; seed)
    Random.seed!(p.rng, seed)

    return nothing
end
