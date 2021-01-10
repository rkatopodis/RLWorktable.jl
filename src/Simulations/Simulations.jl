module Simulations

import ..reset!

using ..Environments: AbstractEnvironment, step!, render, close
using ..Agents: AbstractAgent, select_action!, observe!, update!

using ProgressMeter: @showprogress
using StatsBase: mean

export EnvironmentLoop, simulate, simulate_episode, mean_total_reward, experiment

struct EnvironmentLoop
    env::AbstractEnvironment
    agent::AbstractAgent
end

EnvironmentLoop(; env, agent) = EnvironmentLoop(env, agent)

function simulate!(loop::EnvironmentLoop, total_rewards::AbstractVector{Float64}; episodes::Int)
    episodes <= 0 && throw(
      DomainError(episodes, "Number of episodes must be greater than zero"))

    @showprogress for epi in 1:episodes
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
    end

    nothing
end

function simulate(loop::EnvironmentLoop; episodes::Int)
    total_rewards = Vector{Float64}(undef, episodes)

    simulate!(loop, total_rewards; episodes)

    total_rewards
end

function simulate_episode(env::AbstractEnvironment, agent::AbstractAgent; viz=false)
    reward, obs, done = reset!(env)

    total_reward = zero(Float64)
    while !done
        action = select_action!(agent, obs; ϵ=0.0)
        reward, obs, done = step!(env, action)

        total_reward += reward
        viz && (render(env); sleep(1 / 50))
    end

    viz && close(env)

    return total_reward
end

function mean_total_reward(env::AbstractEnvironment, agent::AbstractAgent; episodes::Int=100)
    mean([simulate_episode(env, agent) for _ in 1:episodes])
end

# Run a RL experiment with a given environment, learning algorithm and parametrization
# Returns a matrix where each row contains the cumulative rewards obtaining in the respective
# training replicaton
function experiment(replications::Int, env::E, agent::A; episodes=100) where {E <: AbstractEnvironment,A <: AbstractAgent}
    loop = EnvironmentLoop(env, agent)

    result = Array{Float64,2}(undef, replications, episodes)
    
    for row in eachrow(result)
        simulate!(loop, row; episodes)
        reset!(agent)
    end

    return result
end

end