using Random

struct RandomAgent <: AbstractAgent
    actions::AbstractUnitRange
    rng::AbstractRNG

    function RandomAgent(actions; seed::Union{Nothing,Int}=nothing)
        !isnothing(seed) && seed < 0 && throw(DomainError(seed, "Seed must be non-negative"))
        rng = isnothing(seed) ? MersenneTwister() : MersenneTwister(seed)

        new(actions, rng)
    end
end

observe_first!(::RandomAgent) = nothing
select_action(agent::RandomAgent, observation) = rand(agent.rng, agent.actions)
observe!(::RandomAgent) = nothing
update!(::RandomAgent) = nothing