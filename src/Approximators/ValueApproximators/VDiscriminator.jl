import ..update!, ..reset!

using Ramnet.Models.AltDiscriminators:RegressionDiscriminator
using StaticArrays

struct VDiscriminator{OS,T <: Real,O <: StaticArray{Tuple{OS},T,1},E <: AbstractEncoder{T}}
    V::RegressionDiscriminator{1}
end

function VDiscriminator(::Type{O}, n::Int, γ::Float64, encoder::E, seed::Union{Nothing,Int}) where {OS,T <: Real,O <: StaticArray{Tuple{OS},T,1},E <: AbstractEncoder{T}}
    VDiscriminator{OS,T,O,E}(RegressionDiscriminator{1}(OS, n, encoder; seed, γ))
end

function update!(v::VDiscriminator{OS,T,O,E}, observation::O, value) where {OS,T,O,E}
    train!(v.V, observation, value)
end

function (v::VDiscriminator{OS,T,O,E})(observation::O) where {OS,T,O,E}
    predict(v.V, observation) |> first
end

function reset!(v::VDiscriminator; seed::Union{Nothing,Int}=nothing)
    if !isnothing(seed)
        if seed >= 0
            Random.seed!(q.rng, seed)
        else
            throw(DomainError(seed, "Seed must be non-negative"))
        end
    end

    reset!(v.V; seed)
end