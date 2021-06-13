import ..update!, ..select_action, ..reset!

using ..Buffers:Experience

using StaticArrays
using Random
using LinearAlgebra:Diagonal, UniformScaling

using Ramnet.Models.AltDiscriminators:RegularizedFunctionalDiscriminator
using Ramnet.Optimizers
import Ramnet.Loss

struct RegularizedContinousPolicyLoss <: Loss.AbstractLoss end

# Policy approximator with only two admissible actions: {-1, 1}.
mutable struct RegularizedContinousPolicy{OS,AS,T <: Real,O <: StaticArray{Tuple{OS},T,1},A <: StaticArray{Tuple{AS},T,1},E <: AbstractEncoder{T}}
    opt::FunctionalOptimizer{RegularizedContinousPolicyLoss}
    f::RegularizedFunctionalDiscriminator{AS,T,E}
    # Covariance diagonal matrix
    start_cov::Float64
    end_cov::Float64
    cov_decay::Int
    update_count::Int
    rng::MersenneTwister
end

function RegularizedContinousPolicy(::Type{O}, ::Type{A}, n::Int, encoder::E; α::Float64=1.0, start_cov::Float64, end_cov::Float64, cov_decay::Int, η::Float64=0.1, epochs::Int=1, partitioner::Symbol=:uniform_random, seed::Union{Nothing,Int}=nothing) where {AS,OS,T <: Real,O <: StaticArray{Tuple{OS},T,1},A <: StaticArray{Tuple{AS},T,1},E <: AbstractEncoder{T}}
    RegularizedContinousPolicy{OS,AS,T,O,A,E}(
        FunctionalOptimizer(RegularizedContinousPolicyLoss(); learning_rate=η, epochs),
        RegularizedFunctionalDiscriminator{AS}(OS, n, α, encoder, partitioner; seed),
        start_cov,
        end_cov,
        cov_decay,
        0,
        isnothing(seed) ? MersenneTwister() : MersenneTwister(seed)
    )
end

function cov(p::RegularizedContinousPolicy{OS,AS,T,O,A,E}) where {OS,AS,T,O,A,E}
    α = p.update_count / p.cov_decay
    UniformScaling((1 - α) * p.start_cov + α * p.end_cov)
end

function Loss.grad(::RegularizedContinousPolicyLoss, p::RegularizedContinousPolicy{OS,AS,T,O,A,E}, observation::O, action::A, G) where {OS,AS,T,O,A,E}
    inv(cov(p)) * (predict(p.f, observation) - action) * G
end

# TODO: Validate action (must be ether -1 or +1)
function update!(p::RegularizedContinousPolicy{OS,AS,T,O,A,E}, observation::O, action::A, G) where {OS,AS,T,O,A,E}
    gradient = Optimizers.grad(p.opt, p, observation, action, G)

    train!(p.f, observation, learning_rate(p.opt) * gradient)
    p.update_count = p.update_count == p.cov_decay ? p.cov_decay : p.update_count + 1

    return nothing
end

function update!(p::RegularizedContinousPolicy{OS,AS,T,O,A,E}, experiences::AbstractVector{Experience{O,A}}, critic) where {OS,AS,T,O,A,E}
    gradients = Matrix{Float64}(undef, AS, length(experiences))
    observations = Matrix{T}(undef, OS, length(experiences))
        
    for i in eachindex(experiences)

        observation = experiences[i].observation
        action = experiences[i].action
        value = experiences[i].value
        advantage = value - critic(observation)
        gradients[:, i] = Optimizers.grad(p.opt, p, observation, action, advantage)
        observations[:, i] = observation
    end

    train!(p.f, observations, SizedMatrix{AS,length(experiences)}(learning_rate(p.opt) * gradients))
    p.update_count = p.update_count == p.cov_decay ? p.cov_decay : p.update_count + 1

    nothing
end

function select_action(p::RegularizedContinousPolicy{OS,AS,T,O,A,E}, observation::O) where {OS,AS,T,O,A,E}
    return SizedVector{AS}(cov(p) * randn(p.rng, AS) + predict(p.f, observation))
end

function reset!(p::RegularizedContinousPolicy)
    Ramnet.reset!(p.f)

    return nothing
end
