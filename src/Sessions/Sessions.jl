module Sessions

using Random
using JLD2: save_object, load_object
using JSON
using Base.Filesystem: basename, splitext, joinpath, rm, abspath, isfile
using Dates: format, now, Time
using NPZ

using ..Agents, ..Environments

export Session, load_session, save_session!, Trial, DMTrial, load_trial, save_trial, save_trial!

mutable struct Session{O,A,E<:AbstractEnvironment{O,A},AG<:AbstractAgent{O,A}}
    name::String
    agent::AG
    frames::Int
    elapsed_frames::Int
    # replications::Int
    # elapsed_reps::Int
    cummulative_rewards::Vector{Float64}
    # steps_per_episode::Vector{Int}
    checkpoint_interval::Time
    keep_all::Bool
    last_checkpoint::String
end

function Session(name, ::Type{E}, agent::AG, frames, checkpoint_interval, keep_all::Bool = false) where {O,A,E<:AbstractEnvironment{O,A},AG<:AbstractAgent{O,A}}
    frames ≤ 0 && throw(DomainError(frames, "number of frames must be greater then 0"))
    # checkpoint_interval > episodes && throw(
    #   DomainError(checkpoint_interval, "interval may not be greater then the number of episodes"))
    Session{O,A,E,AG}(name, agent, frames, 0, Float64[], checkpoint_interval, keep_all, "")
end

function Base.show(io::IO, s::Session{O,A,E,AG}) where {O,A,E,AG}
    print(
        io,
        """ Session $(s.name)
         ├ Environment: $(E)
         ├ Agent : $(nameof(AG))
         ├ Frames: $(s.frames)
         ├ Elapsed frames: $(s.elapsed_frames)
         ├ Checkpoint interval: $(s.checkpoint_interval)
         └ Keep all: $(s.keep_all) 
        """
    )
end

function save_session!(s::Session; path::String = pwd())
    path = abspath(path)
    filename = s.name |> basename |> splitext |> first
    filename = filename * "_" * format(now(), "yyyy-mm-ddTHH-MM-SS") * ".jld2"
    absfilename = joinpath(path, filename)

    save_object(absfilename, s)

    !s.keep_all && isfile(s.last_checkpoint) && rm(s.last_checkpoint)

    s.last_checkpoint = absfilename

    nothing
end

function load_session_from_spec(specname::String)
    spec = JSON.parsefile(specname, dicttype = Dict{Symbol,Any})

    name = specname |> basename |> splitext |> first
    env_type = Environments.environment_table[Symbol(spec[:session][:env][:name])]
    agent = make_agent(env_type, spec[:session][:agent])
    frames = spec[:session][:simulation][:frames]
    checkpoint_interval = Time(spec[:session][:simulation][:checkpoint_interval])
    keep_all = spec[:session][:simulation][:keep_all]

    Session(name, env_type, agent, frames, checkpoint_interval, keep_all)
end

load_session_from_jld(name::String) = load_object(name)

function load_session(name::String)
    ext = name |> basename |> splitext |> last

    if ext == ".jld2"
        return load_session_from_jld(name)
    elseif ext == ".json"
        return load_session_from_spec(name)
    else
        throw(ArgumentError("unable to load session from $name"))
    end
end

# ----------------------------------- Trial ---------------------------------- #
mutable struct Trial{O,A,E<:AbstractEnvironment{O,A},AG<:AbstractAgent{O,A}}
    name::String
    agent::AG
    episodes::Int
    elapsed_episodes::Int
    replications::Int
    elapsed_reps::Int
    cummulative_rewards::Matrix{Float64}
    # steps_per_episode::Vector{Int}
    checkpoint_interval::Time
    keep_all::Bool
    last_checkpoint::String
    rng::MersenneTwister
end

function Trial(name, ::Type{E}, agent::AG, episodes, replications, checkpoint_interval, keep_all::Bool = false; seed::Union{Nothing,UInt} = nothing) where {O,A,E<:AbstractEnvironment{O,A},AG<:AbstractAgent{O,A}}
    episodes ≤ 0 && throw(DomainError(episodes, "number of episodes must be greater then 0"))
    # checkpoint_interval > episodes && throw(
    #   DomainError(checkpoint_interval, "interval may not be greater then the number of episodes"))
    Trial{O,A,E,AG}(name, agent, episodes, 0, replications, 0, Matrix{Float64}(undef, episodes, replications), checkpoint_interval, keep_all, "", MersenneTwister(seed))
end

function Base.show(io::IO, t::Trial{O,A,E,AG}) where {O,A,E,AG}
    print(
        io,
        """ Trial $(t.name)
         ├ Environment: $(E)
         ├ Agent : $(nameof(AG))
         ├ Episodes: $(t.episodes)
         ├ Replications: $(t.replications)
         ├ Elapsed replications: $(t.elapsed_reps)
         ├ Elapsed episodes: $(t.elapsed_episodes)
         ├ Checkpoint interval: $(t.checkpoint_interval)
         └ Keep all: $(t.keep_all) 
        """
    )
end

function save_trial!(t::Trial; path::String = pwd())
    path = abspath(path)
    filename = t.name |> basename |> splitext |> first
    filename = filename * "_" * format(now(), "yyyy-mm-ddTHH-MM-SS") * ".jld2"
    absfilename = joinpath(path, filename)

    save_object(absfilename, t)

    !t.keep_all && isfile(t.last_checkpoint) && rm(t.last_checkpoint)

    t.last_checkpoint = absfilename

    nothing
end

# ----------------------------- Deepmind-like Trial ----------------------------
mutable struct DMTrial{O,A,E<:AbstractEnvironment{O,A}}
    name::String
    agent_spec::Dict{Symbol,Any}
    total_timesteps::Int
    replications::Int
    eval_freq::Int
    eval_episodes::Int
    timesteps::Vector{Int}
    results::Dict{UInt,Matrix{Float64}}
    rng::MersenneTwister
end

function DMTrial(name, ::Type{E}, agent_spec, total_timesteps, replications, eval_freq, eval_episodes; seed::Union{Nothing,UInt} = nothing) where {O,A,E<:AbstractEnvironment{O,A}}
    total_timesteps ≤ 0 && throw(DomainError(total_timesteps, "number of timesteps must be greater then 0"))
    # checkpoint_interval > episodes && throw(
    #   DomainError(checkpoint_interval, "interval may not be greater then the number of episodes"))
    DMTrial{O,A,E}(name, agent_spec, total_timesteps, replications, eval_freq, eval_episodes, collect(eval_freq*Base.OneTo(fld(total_timesteps, eval_freq))), Dict{Int,Matrix{Float64}}(), MersenneTwister(seed))
end

function Base.show(io::IO, t::DMTrial{O,A,E}) where {O,A,E}
    print(
        io,
        """ Trial $(t.name)
         ├ Environment: $(E)
         ├ Agent spec : $(t.agent_spec)
         ├ Total timesteps: $(t.total_timesteps)
         └ Replications: $(t.replications)
        """
    )
end

# Save it in the format expected by the functions in Python
# A folder named "results"
# Inside, we have folders for each replication, name after their seeds
# Inside those, we have one "evaluations.npz" file for each folder
# The npz file has two fields: timesteps, a array with the timesteps where
# evaluations took place, results, a matrix where each column contains the
# evalutions along the aforementioned timesteps, one column for each seed.
function save_trial(t::DMTrial; path::String = pwd())
    resultsdir = mkpath(joinpath(path, "results"))

    for (seed, results) in t.results
        seeddir = mkpath(joinpath(resultsdir, "$(seed)"))
        npzwrite(
            joinpath(seeddir, "evaluations.npz"),
            Dict(
                "timesteps" => t.timesteps,
                "results" => results
            )
        )
    end

    nothing
end

# ----------------------------- General functions ------------------------------

function load_trial_from_spec(specname::String)
    spec = JSON.parsefile(specname, dicttype = Dict{Symbol,Any})
    dm_like = haskey(spec, :dmtrial)
    spec = spec |> values |> first

    name = specname |> basename |> splitext |> first
    env_type = Environments.environment_table[Symbol(spec[:env][:name])]
    seed = UInt(spec[:simulation][:seed])
    replications = spec[:simulation][:replications]

    if dm_like
        total_timesteps = spec[:simulation][:total_timesteps]
        eval_freq = spec[:simulation][:eval_freq]
        eval_episodes = spec[:simulation][:eval_episodes]

        return DMTrial(name, env_type, spec[:agent], total_timesteps, replications, eval_freq, eval_episodes; seed)
    else
        agent = make_agent(env_type, spec[:agent]; seed)
        checkpoint_interval = Time(spec[:simulation][:checkpoint_interval])
        keep_all = spec[:simulation][:keep_all]
        episodes = spec[:simulation][:episodes]
        return Trial(name, env_type, agent, episodes, replications, checkpoint_interval, keep_all; seed)
    end
end

load_trial_from_jld(name::String) = load_object(name)

function load_trial(name::String)
    ext = name |> basename |> splitext |> last

    if ext == ".jld2"
        return load_trial_from_jld(name)
    elseif ext == ".json"
        return load_trial_from_spec(name)
    else
        throw(ArgumentError("unable to load trial from $name"))
    end
end

end