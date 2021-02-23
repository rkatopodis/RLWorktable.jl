module Simulations

import ..reset!

using ..Environments: AbstractEnvironment, env, step!, render, close, observation_type, observation_extrema, action_type, action_set
using ..Agents: AbstractAgent, agent, encoding, select_action!, observe!, update!

using ProgressMeter: @showprogress, Progress, next!
using StatsBase: mean

using JSON
using FileIO
using Base.Filesystem: basename, splitext
using Dates: format, now

export EnvironmentLoop, simulate, simulate_episode, mean_total_reward, experiment

struct EnvironmentLoop{O <: AbstractVector,A <: Real}
    env::AbstractEnvironment
    agent::AbstractAgent{O,A}
end

EnvironmentLoop(; env, agent::AbstractAgent{O,A}) where {O <: AbstractVector,A <: Real} = EnvironmentLoop{O,A}(env, agent)

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
            ag = loop.agent
            action = select_action!(ag, obs)
            ev = loop.env
            reward, obs, done = step!(ev, action)

            total_reward += reward
            observe!(loop.agent, action, reward, obs, done)
            update!(loop.agent)
        end

        total_rewards[epi] = total_reward

        show_progress && next!(prog)
    end

    nothing
end

function simulate(loop::EnvironmentLoop{O,A}; episodes::Int, show_progress::Bool=true) where {O <: AbstractVector,A <: Real}
    total_rewards = Vector{Float64}(undef, episodes)

    simulate!(loop, total_rewards, episodes, show_progress)

    total_rewards
end

function simulate_episode(env::AbstractEnvironment, agent::AbstractAgent; viz=false, fps=50)
    reward, obs, done = reset!(env)

    total_reward = zero(Float64)
    try
        while !done
            action = select_action!(agent, obs)
            reward, obs, done = step!(env, action)

            total_reward += reward
            viz && (render(env); sleep(1 / fps))
        end

        viz && close(env)

        return total_reward
    finally
        viz && close(env)
    end
end

function mean_total_reward(env::AbstractEnvironment, agent::AbstractAgent; episodes::Int=100)
    mean([simulate_episode(env, agent) for _ in 1:episodes])
end

# Run a RL experiment with a given environment, learning algorithm and parametrization
# Returns a matrix where each column contains the cumulative rewards obtaining in the respective
# training replicaton
function experiment(replications::Int, env::E, agent::A; episodes=100, show_progress::Bool=true) where {E <: AbstractEnvironment,A <: AbstractAgent}
    loop = EnvironmentLoop(env, agent)

    results = Dict{String,Any}()

    total_rewards = Array{Float64,2}(undef, episodes, replications)

    if show_progress
        prog = Progress(replications)
    end

    for (i, col) in Iterators.enumerate(eachcol(total_rewards))
        simulate!(loop, col, episodes, replications == 1) # Simulation progress is only shown when there is only one replication

        if i == replications
            results["agent"] = deepcopy(agent)
        end

        reset!(agent)

        show_progress && next!(prog)
    end

    results["total_rewards"] = total_rewards

    return results
end

function experiment(spec_file::String; save_results::Bool=true)
    spec = JSON.parsefile(spec_file, dicttype=Dict{Symbol,Any})

    environment = env(spec[:env])
    ag = agent(spec[:agent], environment)

    exp_spec = spec[:experiment]

    replications  = get(exp_spec, :replications, 1)
    episodes      = get(exp_spec, :episodes, 100)
    show_progress = get(exp_spec, :show_progress, true)

    results = experiment(replications, environment, ag; episodes, show_progress)

    if save_results
        base_filename = spec_file |> basename |> splitext |> first
        results_filename = base_filename * "_" * format(now(), "yyyy-mm-ddTHH-MM-SS") * ".jld2"

        save(results_filename, results)
    end

    results
end

end
