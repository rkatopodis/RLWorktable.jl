import ..update!, ..select_action, ..reset!

using ..Buffers:Experience

using StaticArrays
using Random
using LinearAlgebra:Diagonal, UniformScaling

using Ramnet.Models.AltDiscriminators:AdaptiveFunctionalDiscriminator
using Ramnet.Optimizers
import Ramnet.Loss

struct AdaptiveBinaryPolicyLoss <: Loss.AbstractLoss end

# Policy approximator with only two admissible actions: {-1, 1}.
struct AdaptiveBinaryPolicy{OS,T <: Real,O <: StaticArray{Tuple{OS},T,1},E <: AbstractEncoder{T}}
    opt::AdaptiveOptimizer{1,AdaptiveBinaryPolicyLoss}
    f::AdaptiveFunctionalDiscriminator{1,T,E}
    rng::MersenneTwister
end

function AdaptiveBinaryPolicy(::Type{O}, n::Int, encoder::E; λ::Float64=1.0, μ::Float64=0.1, η::Float64=0.1, epochs::Int=1, partitioner::Symbol=:uniform_random, seed::Union{Nothing,Int}=nothing) where {OS,T <: Real,O <: StaticArray{Tuple{OS},T,1},E <: AbstractEncoder{T}}
    AdaptiveBinaryPolicy{OS,T,O,E}(
        AdaptiveOptimizer{1}(AdaptiveBinaryPolicyLoss(); λ, meta_rate=μ, initial_rate=η),
        AdaptiveFunctionalDiscriminator{1}(OS, n, λ, encoder, partitioner; seed),
        isnothing(seed) ? MersenneTwister() : MersenneTwister(seed)
    )
end

# Compute π(action | observation)
function probability(action::Int, pred)
    # @show pred
    # @show typeof(pred)
    1 / (1 + exp(sign(action) * first(pred)))
end

function Loss.grad(::AdaptiveBinaryPolicyLoss, action, pred)
    # @show pred
    # @show typeof(pred)
    sign(action) * (1 - probability(action, pred))
end

function Loss.hessian(::AdaptiveBinaryPolicyLoss, action, pred)
    p = probability(action, pred)
    return -1 * p * (1 - p)
end

# TODO: Validate action (must be ether -1 or +1)
function update!(p::AdaptiveBinaryPolicy{OS,T,O,E}, observation::O, action::Int, G) where {OS,T,O,E}
    gradient, trace_update = Optimizers.grad(p.opt, p.f, observation, action)
    rate = SizedVector{1}(learning_rate(p.opt))

    train!(p.f, observation, G * rate .* gradient, rate .* SizedVector{1}(trace_update))
    # p.opt.loss.update_count = p.opt.loss.update_count == p.opt.loss.cov_decay ? p.opt.loss.cov_decay : p.opt.loss.update_count + 1

    return nothing
end


# function update!(p::AdaptiveBinaryPolicy{OS,AS,T,O,A,E}, experiences::AbstractVector{Experience{O,A}}, critic) where {OS,AS,T,O,A,E}
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

function select_action(p::AdaptiveBinaryPolicy{OS,T,O,E}, observation::O) where {OS,T,O,E}
    pred, _ = predict(p.f, observation)
    return rand(p.rng) < probability(1, pred) ? 1 : -1
end

function reset!(p::AdaptiveBinaryPolicy)
    Ramnet.reset!(p.f)

    return nothing
end
