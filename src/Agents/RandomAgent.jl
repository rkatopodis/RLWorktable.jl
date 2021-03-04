using Random

using StaticArrays:SizedVector

struct RandomAgent{O,A,E <: AbstractEnvironment{O,A}} <: AbstractAgent{O,A}
    env::E
end

# observe_first!(::RandomAgent) = nothing
function select_action!(agent::RandomAgent, args...; kargs...)
    SizedVector{action_length(agent.env)}(agent.env.env.action_space.sample())
end
observe!(::RandomAgent, observation) = nothing
observe!(::RandomAgent, action, reward, observation, terminal) = nothing
update!(::RandomAgent) = nothing
