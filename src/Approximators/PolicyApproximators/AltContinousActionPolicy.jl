import ..update!, ..select_action, ..reset!

using StaticArrays
using Random
using LinearAlgebra:Diagonal

using Ramnet.Models.AltDiscriminators:FunctionalDiscriminator
using Ramnet.Optimizers
import Ramnet.Loss

struct AltContinousActionPolicyLoss <: Loss.AbstractLoss end

# Policy approximator with only two admissible actions: {-1, 1}.
struct AltContinousActionPolicy{OS,AS,T <: Real,O <: StaticArray{Tuple{OS},T,1},A <: StaticArray{Tuple{AS},T,1},E <: AbstractEncoder{T}}
    opt::FunctionalOptimizer{AltContinousActionPolicyLoss}
    f::FunctionalDiscriminator{6,T,E}
    rng::MersenneTwister
end

function AltContinousActionPolicy(::Type{O}, ::Type{A}, n::Int, encoder::E; η::Float64=0.1, epochs::Int=1, partitioner::Symbol=:uniform_random, seed::Union{Nothing,Int}=nothing) where {AS,OS,T <: Real,O <: StaticArray{Tuple{OS},T,1},A <: StaticArray{Tuple{AS},T,1},E <: AbstractEncoder{T}}
    AltContinousActionPolicy{OS,AS,T,O,A,E}(
        FunctionalOptimizer(AltContinousActionPolicyLoss(); learning_rate=η, epochs),
        FunctionalDiscriminator{2 * AS}(OS, n, encoder, partitioner; seed),
        isnothing(seed) ? MersenneTwister() : MersenneTwister(seed)
    )
    end

function gaussian_params(p::AltContinousActionPolicy{OS,AS,T,O,A,E}, observation::O) where {OS,AS,T,O,A,E}
    params = predict(p.f, observation)
    # @show params
    μ = params[1:AS]
    σ = exp.(params[AS + 1:end]) .+ eps()

    return (μ, σ)
end

function Loss.grad(::AltContinousActionPolicyLoss, p::AltContinousActionPolicy{OS,AS,T,O,A,E}, observation::O, action::A, G) where {OS,AS,T,O,A,E}
    μ, σ = gaussian_params(p, observation)

    # @show μ, σ
    inv_cov = inv(Diagonal(σ)^2)
    rel_action = action - μ

    gradient = [-1 * inv_cov * rel_action; 1 .- inv_cov * rel_action.^2] * G

    @assert !any(isnan.(gradient)) "Something wrong with the gradient"

    return gradient
end

# TODO: Validate action (must be ether -1 or +1)
function update!(p::AltContinousActionPolicy{OS,AS,T,O,A,E}, observation::O, action::A, G) where {OS,AS,T,O,A,E}
    gradient = Optimizers.grad(p.opt, p, observation, action, G)

    train!(p.f, observation, learning_rate(p.opt) * SVector{2 * AS}(gradient))
    
    return nothing
end

# TODO: The clamping of the action is specific to the hopper agent!
function select_action(p::AltContinousActionPolicy{OS,AS,T,O,A,E}, observation::O) where {OS,AS,T,O,A,E}
    μ, σ = gaussian_params(p, observation)

# @show μ
    # @show σ
    
    cov = Diagonal(σ)^2

    return A(clamp.(cov * randn(p.rng, AS) + μ, -1.0f0, 1.0f0))
end

function reset!(p::AltContinousActionPolicy)
    Ramnet.reset!(p.f)

    return nothing
end
