import ..update!, ..select_action, ..reset!

using Ramnet.Models.AltDiscriminators:RegressionDiscriminator
using StatsBase: mean, sample, pweights
using StaticArrays

struct QDiscriminator{AS,OS,T <: Real,O <: StaticArray{Tuple{OS},T,1},A <: Real,E <: AbstractEncoder{T}}
    actions::SVector{AS,A}
    Q::Dict{A,RegressionDiscriminator{1}}
    last_q_values::Vector{Float64}
    rng::MersenneTwister
end

function QDiscriminator(::Type{O}, actions::StaticArray{Tuple{AS},A,1}, n::Int, γ::Float64, encoder::E, seed::Union{Nothing,UInt}) where {AS,OS,T <: Real,O <: StaticArray{Tuple{OS},T,1},A <: Real,E <: AbstractEncoder{T}}
    !isnothing(seed) && seed < 0 && throw(DomainError(seed, "Seed must be non-negative"))
    rng = isnothing(seed) ? MersenneTwister() : MersenneTwister(seed)

    Q = Dict(a => RegressionDiscriminator{1}(OS, n, encoder; seed, γ) for a in actions)
    QDiscriminator{AS,OS,T,O,A,E}(actions, Q, zeros(Float64, AS), rng)
end

function update!(q::QDiscriminator{AS,OS,T,O,A,E}, observation::O, action::A, G) where {AS,OS,T <: Real,O <: StaticArray{Tuple{OS},T,1},A <: Real,E <: AbstractEncoder{T}}
    train!(q.Q[action], observation, G)
end

function (q::QDiscriminator{AS,OS,T,O,A,E})(observation::O, action::A) where {AS,OS,T,O,A,E}
    predict(q.Q[action], observation) |> first
end

function q_values!(q::QDiscriminator{AS,OS,T,O,A,E}, observation::O) where {AS,OS,T,O,A,E}
    q_max = typemin(Float64)
    v = 0.0
    for (i, action) in Iterators.zip(eachindex(q.last_q_values), q.actions)
        v = q(observation, action)
        q.last_q_values[i] = v
        q_max = v > q_max ? v : q_max
    end

    return q_max
end

function select_action(q::QDiscriminator{AS,OS,T,O,A,E}, observation::O) where {AS,OS,T,O,A,E}
    q_max = q_values!(q, observation)

    return sample(q.rng, q.actions, pweights(q.last_q_values .== q_max))
end

function reset!(q::QDiscriminator; seed::Union{Nothing,UInt}=nothing)
    if !isnothing(seed)
        if seed >= 0
            Random.seed!(q.rng, seed)
        else
            throw(DomainError(seed, "Seed must be non-negative"))
        end
    end

    # reset!(q.Q; seed)
    for (_, v) in q.Q
        Ramnet.reset!(v; seed)
    end
    fill!(q.last_q_values, 0.0)

    nothing
end