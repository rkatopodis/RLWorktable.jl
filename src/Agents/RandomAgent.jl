using Random

struct RandomAgent{O <: AbstractVector,A <: Real} <: AbstractAgent{O,A}
    actions::AbstractUnitRange
    rng::AbstractRNG

    function RandomAgent{O,A}(actions; seed::Union{Nothing,Int}=nothing) where {O <: AbstractVector,A <: Real}
        !isnothing(seed) && seed < 0 && throw(DomainError(seed, "Seed must be non-negative"))
        rng = isnothing(seed) ? MersenneTwister() : MersenneTwister(seed)

        new(actions, rng)
    end
end

# observe_first!(::RandomAgent) = nothing
select_action(agent::RandomAgent, observation) = rand(agent.rng, agent.actions)
observe!(::RandomAgent, observation) = nothing
observe!(::RandomAgent, action, reward, observation, terminal) = nothing
update!(::RandomAgent) = nothing