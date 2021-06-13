module Simulations

import ..reset!

using ..Environments:
    AbstractEnvironment,
    env,
    step!,
    set_render_mode!,
    render,
    close,
    observation_type,
    observation_extrema,
    action_type,
    action_set,
    fps
using ..Agents:
    AbstractAgent, agent, encoding, select_action!, observe!, update!

using ..Sessions

using Printf

using ProgressMeter: @showprogress, Progress, next!
import ProgressMeter

using StatsBase:mean

using JSON
using FileIO
using Base.Filesystem: basename, splitext
using Dates: format, now, Millisecond
using Logging
using TensorBoardLogger: TBLogger, tb_overwrite

export EnvironmentLoop,
    simulate, simulate_episode, mean_total_reward, experiment, simulate!, visualize

struct EnvironmentLoop{O,A,E <: AbstractEnvironment{O,A},AG <: AbstractAgent{O,A}}
    env::E
    agent::AG
end

using Ramnet.Optimizers:learning_rate

function simulate!(s::Session{O,A,E,AG}) where {O,A,E,AG}
    lg = TBLogger("tb_log", tb_overwrite, min_level=Logging.Info)
    env = E()

    sizehint!(s.cummulative_rewards, s.frames)
    initial_frame_count = s.elapsed_frames
    prog = Progress(s.frames - s.elapsed_frames)
    last_checkpoint_instant = now()
    checkpoint_millis = convert(Millisecond, s.checkpoint_interval.instant)

    with_logger(lg) do
        while s.elapsed_frames < s.frames
            reward, obs, done = reset!(env)
            # @show typeof(obs)
            observe!(s.agent, obs)

            total_reward = reward
            steps = 0
            while !done
                action = select_action!(s.agent, obs)
                reward, obs, done = step!(env, action)

                total_reward += reward
                steps += 1
                observe!(s.agent, action, reward, obs, done)
                update!(s.agent)

                if (s.elapsed_frames + steps) == s.frames
                    break
                end
            end

            s.elapsed_frames += steps
            push!(s.cummulative_rewards, total_reward)

            # Log to tensorboard
            @info "Cummulative Reward" r = total_reward log_step_increment = steps

            # Log learning rates
            # rates = learning_rate(s.agent.actor.opt)

            # @info "Learning rate #1" r1 = rates[1] log_step_increment = 0
            # @info "Learning rate #2" r2 = rates[2] log_step_increment = 0
            # @info "Learning rate #3" r3 = rates[3] log_step_increment = 0

            # Make checkpoint here

            if (now() - last_checkpoint_instant) > checkpoint_millis
                save_session!(s)
                last_checkpoint_instant = now()
            end
            
            ProgressMeter.update!(prog, s.elapsed_frames - initial_frame_count)
        end
    end

    save_session!(s)

    nothing
end

function simulate!(t::Trial{O,A,E,AG}) where {O,A,E,AG}
    env = E()

    # sizehint!(t.cummulative_rewards, t.frames)
    # initial_frame_count = t.elapsed_episodes
    initial_episode_count = t.episodes * t.elapsed_reps + t.elapsed_episodes
    # prog = Progress(t.frames - t.elapsed_frames)
    prog = Progress(t.episodes * t.replications - initial_episode_count)
    last_checkpoint_instant = now()
    checkpoint_millis = convert(Millisecond, t.checkpoint_interval.instant)

    for r in 1:t.replications
        t.elapsed_episodes = 0
        for epi in 1:t.episodes
            reward, obs, done = reset!(env)
            observe!(t.agent, obs)

            total_reward = reward
            # steps = 0
            while !done
                action = select_action!(t.agent, obs)
                reward, obs, done = step!(env, action)

                total_reward += reward
                # steps += 1
                observe!(t.agent, action, reward, obs, done)
                update!(t.agent)

                # if (t.elapsed_frames + steps) == t.frames
                #     break
                # end
            end

            # t.elapsed_frames += steps
            t.elapsed_episodes += 1
            t.cummulative_rewards[epi, r] = total_reward
            # push!(t.cummulative_rewards, total_reward)

            # Make checkpoint here

            if (now() - last_checkpoint_instant) > checkpoint_millis
                save_trial!(t)
                last_checkpoint_instant = now()
            end
        
            ProgressMeter.update!(
                prog,
                t.episodes * t.elapsed_reps + t.elapsed_episodes - initial_episode_count
            )
        end
        t.elapsed_reps += 1
        reset!(t.agent; seed=rand(t.rng, UInt))
    end
    save_trial!(t)

    nothing
end

function visualize(::Type{EN}, agent::AG) where {O,A,EN <: AbstractEnvironment{O,A}, AG <: AbstractAgent{O,A}}
    agent.action = nothing

    interval = 1. / fps(EN)
    en = EN()
    set_render_mode!(en, :human)

    reset!(en)

    try
        while true
            frame = 0
            score = 0
            restart_delay = 0
            # disable rendering during reset, makes loading much faster
            reward, obs, done = reset!(en)

            while true
                sleep(interval)

                # a = clamp.(select_action!(s.agent, obs), -1.0f0, 1.0f0)
                a = select_action!(agent, obs)

                reward, obs, done = step!(en, a)
                score += reward
                frame += 1
                still_open = render(en, :human)

                still_open == false && return
                !done && continue

                if restart_delay == 0
                    @printf "Score: %0.2f in %i frames\n" score frame

                    restart_delay = 60 * 1  # 2 sec at 60 fps
                else
                    restart_delay -= 1
                    restart_delay == 0 && break
                end
            end
        end
        finally
        close(en)
    end
end

function visualize(t::Trial{O,A,E,AG}) where {O,A,E,AG}
    visualize(E, t.agent)
end

function visualize(s::Session{O,A,E,AG}) where {O,A,E,AG}
    s.agent.action = nothing

    interval = 1. / fps(E)
    en = E()
    set_render_mode!(en, :human)

    reset!(en)

    try
        while true
            frame = 0
            score = 0
            restart_delay = 0
        # disable rendering during reset, makes loading much faster
            reward, obs, done = reset!(en)

            while true
                sleep(interval)

            # a = clamp.(select_action!(s.agent, obs), -1.0f0, 1.0f0)
                a = select_action!(s.agent, obs)

                reward, obs, done = step!(en, a)
                score += reward
                frame += 1
                still_open = render(en, :human)

                still_open == false && return
                !done && continue

                if restart_delay == 0
                    @printf "Score: %0.2f in %i frames\n" score frame

                    restart_delay = 60 * 1  # 2 sec at 60 fps
                else
                    restart_delay -= 1
                    restart_delay == 0 && break
                end
            end
        end
        finally
        close(en)
    end
end

function simulate!(
    loop::EnvironmentLoop{O,A,E,AG},
    total_rewards::AbstractVector{Float64},
    episodes::Int,
    checkpoint_interval::Int,
    show_progress::Bool
) where {O,A,E,AG}
        episodes <= 0 && throw(
        DomainError(episodes, "Number of episodes must be greater than zero"),
    )

        if show_progress
            prog = Progress(episodes)
        end

        models = Vector{AG}(undef, cld(episodes, checkpoint_interval))
        best_rewards = fill(typemin(Float64), cld(episodes, checkpoint_interval))

        for epi = 1:episodes
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

            checkpoint_index = cld(epi, checkpoint_interval)
            if total_reward > best_rewards[checkpoint_index]
                best_rewards[checkpoint_index] = total_reward
                models[checkpoint_index] = deepcopy(loop.agent)
            end
        
            show_progress && next!(prog)
        end

        return models
    end

    function simulate(
    loop::EnvironmentLoop{O,A};
    episodes::Int,
    checkpoint_interval::Int,
    show_progress::Bool=true,
) where {O,A}
        total_rewards = Vector{Float64}(undef, episodes)

        models = simulate!(loop, total_rewards, episodes, checkpoint_interval, show_progress)

        return NamedTuple{(:total_rewards, :models)}((total_rewards, models))
    end

    function simulate_episode(
    env::AbstractEnvironment,
    agent::AbstractAgent;
    viz=false,
    fps=50,
)
        viz && set_render_mode!(env, :human)

        reward, obs, done = reset!(env)

        agent.action = nothing
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

    function mean_total_reward(
    env::AbstractEnvironment,
    agent::AbstractAgent;
    episodes::Int=100,
)
        mean([simulate_episode(env, agent) for _ = 1:episodes])
    end

# Run a RL experiment with a given environment, learning algorithm and parametrization
# Returns a matrix where each column contains the cumulative rewards obtaining in the respective
# training replicaton
    function experiment(
    replications::Int,
    env::E,
    agent::A;
    episodes=100,
    show_progress::Bool=true,
) where {E <: AbstractEnvironment,A <: AbstractAgent}
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

        replications = get(exp_spec, :replications, 1)
        episodes = get(exp_spec, :episodes, 100)
        show_progress = get(exp_spec, :show_progress, true)

        results = experiment(replications, environment, ag; episodes, show_progress)

        if save_results
            base_filename = spec_file |> basename |> splitext |> first
            results_filename =
            base_filename * "_" * format(now(), "yyyy-mm-ddTHH-MM-SS") * ".jld2"

            save(results_filename, results)
        end

        results
    end

end
