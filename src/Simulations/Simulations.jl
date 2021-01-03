module Simulations

using ..Environments: AbstractEnvironment, step!, reset!, render, close
using ..Agents: AbstractAgent, select_action, observe!, update!

using ProgressMeter: @showprogress

export EnvironmentLoop, simulate, simulate_episode

struct EnvironmentLoop
    env::AbstractEnvironment
    agent::AbstractAgent
end

EnvironmentLoop(; env, agent) = EnvironmentLoop(env, agent)

function simulate(loop::EnvironmentLoop; episodes::Int)
    episodes <= 0 && throw(
      DomainError(episodes, "Number of episodes must be greater than zero"))

    @showprogress for epi in 1:episodes
        reward, obs, done = reset!(loop.env)
        observe!(loop.agent, obs)

        while !done
            action = select_action(loop.agent, obs)
            reward, obs, done = step!(loop.env, action)

            observe!(loop.agent, action, reward, obs, done)
            update!(loop.agent)
        end
    end
end

function simulate_episode(env::AbstractEnvironment, agent::AbstractAgent; viz=false)
    reward, obs, done = reset!(env)

    total_reward = zero(Float64)
    while !done
        action = select_action(agent, obs)
        reward, obs, done = step!(env, action)

        total_reward += reward
        viz && (render(env); sleep(1 / 60))
    end

    viz && close(env)

    println("Total reward: $total_reward")
end

end