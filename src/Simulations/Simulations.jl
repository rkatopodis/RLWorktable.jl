module Simulations

import ..reset!

using ..Environments: AbstractEnvironment, env, step!, render, close, observation_type, observation_extrema, action_type, action_set
using ..Agents: AbstractAgent, agent, encoding, select_action!, observe!, update!

using ProgressMeter: @showprogress, Progress, next!
using StatsBase: mean

export EnvironmentLoop, simulate, simulate_episode, mean_total_reward, experiment

struct EnvironmentLoop
    env::AbstractEnvironment
    agent::AbstractAgent
end

EnvironmentLoop(; env, agent) = EnvironmentLoop(env, agent)

function simulate!(loop::EnvironmentLoop, total_rewards::AbstractVector{Float64}, episodes::Int, show_progress::Bool)
    episodes <= 0 && throw(
      DomainError(episodes, "Number of episodes must be greater than zero"))

    if show_progress
        prog = Progress(episodes)
    end

    for epi in 1:episodes
        reward, obs, done = reset!(loop.env)
        observe!(loop.agent, obs)

        total_reward = reward
        while !done
            action = select_action!(loop.agent, obs)
            reward, obs, done = step!(loop.env, action)

            total_reward += reward
            observe!(loop.agent, action, reward, obs, done)
            update!(loop.agent)
        end

        total_rewards[epi] = total_reward

        show_progress && next!(prog)
    end

    nothing
end

function simulate(loop::EnvironmentLoop; episodes::Int, show_progress::Bool=true)
    total_rewards = Vector{Float64}(undef, episodes)

    simulate!(loop, total_rewards, episodes, show_progress)

    total_rewards
end

function simulate_episode(env::AbstractEnvironment, agent::AbstractAgent; viz=false, fps=50)
    reward, obs, done = reset!(env)

    total_reward = zero(Float64)
    while !done
        action = select_action!(agent, obs; ϵ=0.0)
        reward, obs, done = step!(env, action)

        total_reward += reward
        viz && (render(env); sleep(1 / fps))
    end

    viz && close(env)

    return total_reward
end

function mean_total_reward(env::AbstractEnvironment, agent::AbstractAgent; episodes::Int=100)
    mean([simulate_episode(env, agent) for _ in 1:episodes])
end

# Run a RL experiment with a given environment, learning algorithm and parametrization
# Returns a matrix where each column contains the cumulative rewards obtaining in the respective
# training replicaton
function experiment(replications::Int, env::E, agent::A; episodes=100, show_progress::Bool=true) where {E <: AbstractEnvironment,A <: AbstractAgent}
    loop = EnvironmentLoop(env, agent)

    result = Array{Float64,2}(undef, episodes, replications)
    
    if show_progress
        prog = Progress(replications)
    end

    for col in eachcol(result)
        simulate!(loop, col, episodes, replications == 1) # Simulation progress is only shown when there is only one replication
        reset!(agent)

        show_progress && next!(prog)
    end

    return result
end

function experiment(spec::Dict{Symbol,Any})
    environment = env(spec[:env])
    ag = agent(spec[:agent], environment)

    exp_spec = spec[:experiment]

    replications  = get(exp_spec, :replications, 1)
    episodes      = get(exp_spec, :episodes, 100)
    show_progress = get(exp_spec, :show_progress, true)

    experiment(replications, environment, ag; episodes, show_progress)
end

end