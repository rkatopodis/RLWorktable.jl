using Ramnet.Encoders

using StaticArrays

using ..Approximators:BinaryActionPolicy
using ..Buffers: MultiStepDynamicBuffer, add!

mutable struct FunctionalPG{OS,T <: Real,O <: StaticArray{Tuple{OS},T,1},E <: AbstractEncoder{T}} <: AbstractAgent{O,Int}
    γ::Float64
    policy::BinaryActionPolicy{OS,T,O,E}
    buffer::MultiStepDynamicBuffer{O,Int}
    observation::Union{Nothing,O}
    done::Bool
    action::Union{Nothing,Int}
    min_obs::Union{Nothing,O}
    max_obs::Union{Nothing,O}
    rng::MersenneTwister
end
  
function FunctionalPG(::Type{O}, n, start_learning_rate, end_learning_rate, learning_rate_decay, epochs, discount, encoder::E; seed::Union{Nothing,UInt}=nothing) where {OS,T <: Real,O <: StaticArray{Tuple{OS},T,1},E <: AbstractEncoder{T}}
    !isnothing(seed) && seed < 0 && throw(DomainError(seed, "Seed must be non-negative"))
    rng = isnothing(seed) ? MersenneTwister() : MersenneTwister(seed)

    n < 1 && throw(DomainError(n, "Tuple size must be at least 1"))

    policy = BinaryActionPolicy(O, n, encoder; start_learning_rate, end_learning_rate, learning_rate_decay, epochs, partitioner=:uniform_random, seed)

    buffer = MultiStepDynamicBuffer{O,Int}()

    FunctionalPG{OS,T,O,E}(discount, policy, buffer, nothing, false, nothing, nothing, nothing, rng)
end

function FunctionalPG(
  env, encoder::E; tuple_size, start_learning_rate, end_learning_rate, learning_rate_decay, epochs=1, discount=1, seed=nothing) where {T <: Real,E <: AbstractEncoder{T}}

  FunctionalPG(observation_type(env), tuple_size, start_learning_rate, end_learning_rate, learning_rate_decay, epochs, discount, encoder; seed)
end

function observe!(agent::FunctionalPG{OS,T,O,E}, observation::O) where {OS,T <: Real,O <: StaticArray{Tuple{OS},T,1},E <: AbstractEncoder{T}}
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
function observe!(agent::FunctionalPG{OS,T,O,E}, action::Int, reward::Float64, observation::O, done::Bool) where {OS,T <: Real,O <: StaticArray{Tuple{OS},T,1},E <: AbstractEncoder{T}}
  add!(agent.buffer, agent.observation, agent.action, reward)

  agent.observation = observation
  agent.done = done

  # Finding the edges of the domain
  agent.min_obs = min.(agent.min_obs)
  agent.max_obs = max.(agent.max_obs)
  
  nothing
end

function select_action!(agent::FunctionalPG{OS,T,O,E}, observation::O) where {OS,T <: Real,O <: StaticArray{Tuple{OS},T,1},E <: AbstractEncoder{T}}
    agent.action = select_action(agent.policy, observation)

    return agent.action
end

function update!(agent::FunctionalPG{OS,T,O,E}) where {OS,T <: Real,O <: StaticArray{Tuple{OS},T,1},E <: AbstractEncoder{T}}
  if agent.done
      for _ in 1:agent.policy.opt.epochs
          G = 0.0
          for transition in Iterators.reverse(agent.buffer)
              G = transition.reward + agent.γ * G

              update!(agent.policy, transition.observation, transition.action, G)
          end
      end

      reset!(agent.buffer)
  end

  nothing
end

function Agents.reset!(agent::FunctionalPG{OS,T,O,E}; seed::Union{Nothing,UInt}=nothing) where {OS,T <: Real,O <: StaticArray{Tuple{OS},T,1},E <: AbstractEncoder{T}}
  reset!(agent.policy; seed)
  reset!(agent.buffer)

  agent.observation = nothing
  agent.action = nothing
  agent.done = false
  
  nothing
end
