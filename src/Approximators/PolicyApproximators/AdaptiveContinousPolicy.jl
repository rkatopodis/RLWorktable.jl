import ..update!, ..select_action, ..reset!

using ..Buffers:Experience

using StaticArrays
using Random
using LinearAlgebra:Diagonal, UniformScaling

using Ramnet.Models.AltDiscriminators:AdaptiveFunctionalDiscriminator
using Ramnet.Optimizers
import Ramnet.Loss

mutable struct AdaptiveContinousPolicyLoss <: Loss.AbstractLoss
    start_cov::Float64
    end_cov::Float64
    cov_decay::Int
    update_count::Int
end

function AdaptiveContinousPolicyLoss(start_cov::Float64, end_cov::Float64, cov_decay::Int)
    AdaptiveContinousPolicyLoss(start_cov, end_cov, cov_decay, 0)
end

struct AdaptiveContinousPolicy{OS,AS,T <: Real,O <: StaticArray{Tuple{OS},T,1},A <: StaticArray{Tuple{AS},T,1},E <: AbstractEncoder{T}}
    opt::AdaptiveOptimizer{AS,AdaptiveContinousPolicyLoss}
    f::AdaptiveFunctionalDiscriminator{AS,T,E}
    rng::MersenneTwister
end

function AdaptiveContinousPolicy(::Type{O}, ::Type{A}, n::Int, encoder::E; λ::Float64=1.0, start_cov::Float64, end_cov::Float64, cov_decay::Int, μ::Float64=0.1, η::Float64=0.1, epochs::Int=1, partitioner::Symbol=:uniform_random, seed::Union{Nothing,Int}=nothing) where {AS,OS,T <: Real,O <: StaticArray{Tuple{OS},T,1},A <: StaticArray{Tuple{AS},T,1},E <: AbstractEncoder{T}}
    AdaptiveContinousPolicy{OS,AS,T,O,A,E}(
        AdaptiveOptimizer{AS}(AdaptiveContinousPolicyLoss(start_cov, end_cov, cov_decay); λ, meta_rate=μ, initial_rate=η),
        AdaptiveFunctionalDiscriminator{AS}(OS, n, λ, encoder, partitioner; seed),
        isnothing(seed) ? MersenneTwister() : MersenneTwister(seed)
    )
end

function cov(l::AdaptiveContinousPolicyLoss)
    α = l.update_count / l.cov_decay
    UniformScaling((1 - α) * l.start_cov + α * l.end_cov)
end

function Loss.grad(l::AdaptiveContinousPolicyLoss, action, pred_action)
    inv(cov(l)) * (pred_action - action)
end

function Loss.hessian(l::AdaptiveContinousPolicyLoss, y_true, y_pred)
    inv(cov(l))
end

# TODO: Validate action (must be ether -1 or +1)
function update!(p::AdaptiveContinousPolicy{OS,AS,T,O,A,E}, observation::O, action::A, G) where {OS,AS,T,O,A,E}
    gradient, trace_update = Optimizers.grad(p.opt, p.f, observation, action)
    rate = SizedVector{AS}(learning_rate(p.opt))

    # @show rate
    # @show gradient
    # @show typeof(gradient)
    # @show rate .* G * gradient
    # @show rate .* trace_update
    # @show typeof(gradient)
    train!(p.f, observation, G * rate .* gradient, rate .* SizedVector{AS}(trace_update))
    p.opt.loss.update_count = p.opt.loss.update_count == p.opt.loss.cov_decay ? p.opt.loss.cov_decay : p.opt.loss.update_count + 1

    return nothing
end


# function update!(p::AdaptiveContinousPolicy{OS,AS,T,O,A,E}, experiences::AbstractVector{Experience{O,A}}, critic) where {OS,AS,T,O,A,E}
#     gradients = Matrix{Float64}(undef, AS, length(experiences))
#     observations = Matrix{T}(undef, OS, length(experiences))
        
#     for i in eachindex(experiences)
        
#         observation = experiences[i].observation
#         action = experiences[i].action
#         value = experiences[i].value
#         advantage = value - critic(observation)
#         gradients[:, i] = Optimizers.grad(p.opt, p, observation, action, advantage)
#         observations[:, i] = observation
#     end

#     train!(p.f, observations, SizedMatrix{AS,length(experiences)}(learning_rate(p.opt) * gradients))
# p.update_count = p.update_count == p.cov_decay ? p.cov_decay : p.update_count + 1

#     nothing
# end

function select_action(p::AdaptiveContinousPolicy{OS,AS,T,O,A,E}, observation::O) where {OS,AS,T,O,A,E}
return SizedVector{AS,T}(cov(p.opt.loss) * randn(p.rng, AS) + predict(p.f, observation)[1])
end

function reset!(p::AdaptiveContinousPolicy)
    Ramnet.reset!(p.f)

    return nothing
end
