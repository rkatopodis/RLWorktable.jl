module Sessions

using Random
using JLD2: save_object, load_object
using JSON
using Base.Filesystem: basename, splitext, joinpath, rm, abspath, isfile
using Dates: format, now, Time

using ..Agents, ..Environments

export Session, load_session, save_session!, Trial, load_trial, save_trial!

mutable struct Session{O,A,E <: AbstractEnvironment{O,A},AG <: AbstractAgent{O,A}}
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

function Session(name, ::Type{E}, agent::AG, frames, checkpoint_interval, keep_all::Bool=false) where {O,A,E <: AbstractEnvironment{O,A},AG <: AbstractAgent{O,A}}
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

function save_session!(s::Session; path::String=pwd())
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
    spec = JSON.parsefile(specname, dicttype=Dict{Symbol,Any})

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
mutable struct Trial{O,A,E <: AbstractEnvironment{O,A},AG <: AbstractAgent{O,A}}
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

function Trial(name, ::Type{E}, agent::AG, episodes, replications, checkpoint_interval, keep_all::Bool=false; seed::Union{Nothing,UInt}=nothing) where {O,A,E <: AbstractEnvironment{O,A},AG <: AbstractAgent{O,A}}
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

function save_trial!(t::Trial; path::String=pwd())
    path = abspath(path)
    filename = t.name |> basename |> splitext |> first
    filename = filename * "_" * format(now(), "yyyy-mm-ddTHH-MM-SS") * ".jld2"
    absfilename = joinpath(path, filename)

    save_object(absfilename, t)

    !t.keep_all && isfile(t.last_checkpoint) && rm(t.last_checkpoint)

    t.last_checkpoint = absfilename

    nothing
end

function load_trial_from_spec(specname::String)
    spec = JSON.parsefile(specname, dicttype=Dict{Symbol,Any})

    name = specname |> basename |> splitext |> first
    env_type = Environments.environment_table[Symbol(spec[:trial][:env][:name])]
    seed = UInt(spec[:trial][:simulation][:seed])
    agent = make_agent(env_type, spec[:trial][:agent]; seed)
    episodes = spec[:trial][:simulation][:episodes]
    replications = spec[:trial][:simulation][:replications]
    checkpoint_interval = Time(spec[:trial][:simulation][:checkpoint_interval])
    keep_all = spec[:trial][:simulation][:keep_all]

    Trial(name, env_type, agent, episodes, replications, checkpoint_interval, keep_all; seed)
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