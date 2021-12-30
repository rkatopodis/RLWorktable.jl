using Ramnet.Encoders

using ..Approximators: ContinousActionPolicy
using ..Buffers: MultiStepDynamicBuffer, add!

mutable struct ContinousFunctionalPG{OS,AS,T<:Real,O<:StaticArray{Tuple{OS},T,1},A<:StaticArray{Tuple{AS},T,1},E<:AbstractEncoder{T}} <: AbstractAgent{O,A}
    γ::Float64
    policy::ContinousActionPolicy{OS,AS,T,O,A,E}
    buffer::MultiStepDynamicBuffer{O,A}
    observation::Union{Nothing,O}
    done::Bool
    action::Union{Nothing,A}
    min_obs::Union{Nothing,O}
    max_obs::Union{Nothing,O}
    # rng::MersenneTwister
end

function ContinousFunctionalPG(::Type{O}, ::Type{A}, n, start_learning_rate, end_learning_rate, learning_rate_decay, epochs, discount, encoder::E, start_cov::Float64, end_cov::Float64, cov_decay::Int; seed::Union{Nothing,UInt} = nothing) where {OS,AS,T<:Real,O<:StaticArray{Tuple{OS},T,1},A<:StaticArray{Tuple{AS},T,1},E<:AbstractEncoder{T}}
    # !isnothing(seed) && seed < 0 && throw(DomainError(seed, "Seed must be non-negative"))
    # rng = isnothing(seed) ? MersenneTwister() : MersenneTwister(seed)

    n < 1 && throw(DomainError(n, "Tuple size must be at least 1"))

    policy = ContinousActionPolicy(O, A, n, encoder; start_cov, end_cov, cov_decay, start_learning_rate, end_learning_rate, learning_rate_decay, epochs, partitioner = :uniform_random, seed)
    buffer = MultiStepDynamicBuffer{O,A}()

    ContinousFunctionalPG{OS,AS,T,O,A,E}(discount, policy, buffer, nothing, false, nothing, nothing, nothing)
end

function ContinousFunctionalPG(env, encoder::E; tuple_size, start_learning_rate, end_learning_rate, learning_rate_decay, epochs, discount, start_cov, end_cov, cov_decay, seed = nothing) where {T<:Real,E<:AbstractEncoder{T},C<:AbstractMatrix{T}}
    # ContinousFunctionalPG{action_length(env),T,observation_type(env),action_type(env),E,C}(n, observation_length(env), η, γ, encoder, cov; seed)
    ContinousFunctionalPG(observation_type(env), action_type(env), tuple_size, start_learning_rate, end_learning_rate, learning_rate_decay, epochs, discount, encoder, start_cov, end_cov, cov_decay; seed)
end

function observe!(agent::ContinousFunctionalPG{OS,AS,T,O,A,E}, observation::O) where {OS,AS,T,O,A,E}
    agent.observation = observation
    agent.done = false

    # Finding the edges of the domain
    if isnothing(agent.min_obs)
        agent.min_obs = observation
    else
        agent.min_obs = min.(agent.min_obs, observation)
    end

    if isnothing(agent.max_obs)
        agent.max_obs = observation
    else
        agent.max_obs = max.(agent.max_obs, observation)
    end

    nothing
end

# TODO: All observe! methods are the same for all agents. Generalize.
# TODO: This method does not need to take in the action
function observe!(agent::ContinousFunctionalPG{OS,AS,T,O,A,E}, action::A, reward::Float64, observation::O, done::Bool) where {OS,AS,T,O,A,E}
    # REMEMBER: you are forcing reward clipping here!!
    add!(agent.buffer, agent.observation, agent.action, clamp(reward, -1.0, 1.0))

    agent.observation = observation
    agent.done = done

    # Finding the edges of the domain
    agent.min_obs = min.(agent.min_obs)
    agent.max_obs = max.(agent.max_obs)

    nothing
end

function select_action(agent::ContinousFunctionalPG{OS,AS,T,O,A,E}, observation::O; deterministic::Bool = false) where {OS,AS,T,O,A,E}
    return select_action(agent.policy, observation; deterministic)
end

function select_action!(agent::ContinousFunctionalPG{OS,AS,T,O,A,E}, observation::O) where {OS,AS,T,O,A,E}
    agent.action = select_action(agent, observation)

    return agent.action
end

function update!(agent::ContinousFunctionalPG)
    if agent.done
        G = 0.0
        for transition in Iterators.reverse(agent.buffer)
            G = transition.reward + agent.γ * G

            update!(agent.policy, transition.observation, transition.action, G)
        end

        reset!(agent.buffer)
    end

    nothing
end

function Agents.reset!(agent::ContinousFunctionalPG; seed::Union{Nothing,UInt} = nothing)
    reset!(agent.policy; seed)
    reset!(agent.buffer)

    agent.observation = nothing
    agent.action = nothing
    agent.done = false

    nothing
end
