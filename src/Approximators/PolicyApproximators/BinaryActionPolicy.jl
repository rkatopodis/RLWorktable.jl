import ..update!, ..select_action, ..reset!

# Policy approximator with only two admissible actions: {-1, 1}.
struct BinaryActionPolicy{T <: Real,O <: AbstractVector{T},E <: AbstractEncoder{T}}
    f::FunctionalDiscriminator{:mse_loss,T,E}
    rng::MersenneTwister
end

function BinaryActionPolicy{T,O,E}(input_len::Int, n::Int, encoder::E; η::Float64=0.1, partitioner::Symbol=:uniform, seed::Union{Nothing,Int}=nothing) where {T <: Real,O <: AbstractVector{T},E <: AbstractEncoder{T}}
    BinaryActionPolicy{T,O,E}(
        FunctionalDiscriminator(input_len, n, encoder; loss=:mse_loss, η, partitioner, seed),
        isnothing(seed) ? MersenneTwister() : MersenneTwister(seed)
    )
end

# Compute π(action | observation)
function probability(p::BinaryActionPolicy{T,O,E}, action::Int, observation::O) where {T <: Real,O <: AbstractVector{T},E <: AbstractEncoder{T}}
    prob = 1 / (1 + exp(predict(p.f, observation)))

    return action == 1 ? prob : 1 - prob
end

# TODO: Validate action (must be ether -1 or +1)
function update!(p::BinaryActionPolicy{T,O,E}, observation::O, action::Int, G) where {T <: Real,O <: AbstractVector{T},E <: AbstractEncoder{T}}
    weight = -1*sign(action) * (1 - probability(p, action, observation)) * G

    add_kernel!(p.f, observation, p.f.η * weight)

    return nothing
end

function select_action(p::BinaryActionPolicy{T,O,E}, observation::O) where {T <: Real,O <: AbstractVector{T},E <: AbstractEncoder{T}}
    return rand(p.rng) < probability(p, 1, observation) ? 1 : -1
end

function reset!(p::BinaryActionPolicy{T,O,E}) where {T <: Real,O <: AbstractVector{T},E <: AbstractEncoder{T}}
    reset!(p.f)

    return nothing
end
