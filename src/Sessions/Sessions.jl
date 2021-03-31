module Sessions

using JLD2: save_object, load_object
using JSON
using Base.Filesystem: basename, splitext, joinpath, rm, abspath, isfile
using Dates: format, now

using ..Agents, ..Environments

export Session, load_session, save_session!

mutable struct Session{O,A,E <: AbstractEnvironment{O,A},AG <: AbstractAgent{O,A}}
    name::String
    env::E
    agent::AG
    episodes::Int
    elapsed_episodes::Int
    cummulative_rewards::Vector{Float64}
    checkpoint_interval::Int
    keep_all::Bool
    last_checkpoint::String
end

function Session(name, env::E, agent::AG, episodes, checkpoint_interval, keep_all::Bool=false) where {O,A,E <: AbstractEnvironment{O,A},AG <: AbstractAgent{O,A}}
    episodes ≤ 0 && throw(DomainError(episodes, "number of episodes must be greater then 0"))
    checkpoint_interval > episodes && throw(
      DomainError(checkpoint_interval, "interval may not be greater then the number of episodes"))
    Session{O,A,E,AG}(name, env, agent, episodes, 0, Float64[], checkpoint_interval, keep_all, "")
end

function Base.show(io::IO, s::Session)
    print(
        io,
        """ Session $(s.name)
         ├ Environment: $(typeof(s.env))
         ├ Agent : $(nameof(typeof(s.agent)))
         ├ Episodes: $(s.episodes)
         ├ Elapsed episodes: $(s.elapsed_episodes)
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
    env = make_env(spec[:session][:env])
    agent = make_agent(env, spec[:session][:agent])
    episodes = spec[:session][:simulation][:episodes]
    checkpoint_interval = spec[:session][:simulation][:checkpoint_interval]
    keep_all = spec[:session][:simulation][:keep_all]

    Session(name, env, agent, episodes, checkpoint_interval, keep_all)
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

end