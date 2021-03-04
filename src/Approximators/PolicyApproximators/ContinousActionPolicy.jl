import ..update!, ..select_action, ..reset!

using StaticArrays: StaticArray, SizedVector
using Random
using LinearAlgebra:Diagonal

# Policy approximator with only two admissible actions: {-1, 1}.
struct ContinousActionPolicy{D,T <: Real,O <: AbstractVector{T},E <: AbstractEncoder{T},C <: AbstractMatrix{T}}
    f::MultiFunctionalDiscriminator{D,T,E}
    # Covariance diagonal matrix
    cov::C
    inv_cov::C
    rng::MersenneTwister
end

function ContinousActionPolicy{D,T,O,E,C}(input_len::Int, n::Int, encoder::E; cov::C, η::Float64=0.1, partitioner::Symbol=:uniform, seed::Union{Nothing,Int}=nothing) where {D,T <: Real,O <: AbstractVector{T},E <: AbstractEncoder{T},C <: AbstractMatrix{T}}
    ContinousActionPolicy{D,T,O,E,C}(
        MultiFunctionalDiscriminator{D}(input_len, n, encoder; η, partitioner, seed),
        cov,
        inv(cov),
        isnothing(seed) ? MersenneTwister() : MersenneTwister(seed)
    )
end

# TODO: Validate action (must be ether -1 or +1)
function update!(p::ContinousActionPolicy{D,T,O,E,C}, observation::O, action::StaticArray{Tuple{D},Float32,1}, G) where {D,T <: Real,O <: AbstractVector{T},E <: AbstractEncoder{T},C <: AbstractMatrix{T}}
    weight = SizedVector{D}(p.inv_cov * (action - predict(p.f, observation)) * G)
    
    # @show typeof(predict(p.f, observation))
    # @show typeof(action - predict(p.f, observation))
    # @show typeof(p.inv_cov * (action - predict(p.f, observation)))
    # @show typeof(weight)
    add_kernel!(p.f, observation, p.f.η * weight)
    
    return nothing
end

# TODO: The clamping of the action is specific to the hopper agent!
function select_action(p::ContinousActionPolicy{D,T,O,E,C}, observation::O) where {D,T <: Real,O <: AbstractVector{T},E <: AbstractEncoder{T},C <: AbstractMatrix{T}}
    return clamp.(p.cov * randn(p.rng, D) + predict(p.f, observation), -1, 1)
end

function reset!(p::ContinousActionPolicy)
    Ramnet.reset!(p.f)

    return nothing
end
