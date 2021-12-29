using Ramnet.Encoders
using Ramnet.Models.AltDiscriminators:RegressionDiscriminator

using StaticArrays
using Random

using ..Approximators:DiscreteActionPolicy
using ..Buffers: MultiStepDynamicBuffer, add!


mutable struct DiscreteFunctionalAC{OS,AS,T <: Real,O <: StaticArray{Tuple{OS},T,1},E <: AbstractEncoder{T}} <: AbstractAgent{O,Int}
    steps::Int
    γ::Float64
    actor::DiscreteActionPolicy{OS,AS,T,O,E}
    critic::RegressionDiscriminator{1}
    buffer::MultiStepDynamicBuffer{O,Int}
    observation::Union{Nothing,O}
    done::Bool
    action::Union{Nothing,Int}
    rng::MersenneTwister # Is this even used?
end

function DiscreteFunctionalAC{AS}(::Type{O}, steps, n, start_learning_rate, end_learning_rate, learning_rate_decay, epochs, discount, forgetting_factor, encoder::E; seed::Union{Nothing,UInt}=nothing) where {OS,AS,T <: Real,O <: StaticArray{Tuple{OS},T,1},E <: AbstractEncoder{T}}
    !isnothing(seed) && seed < 0 && throw(DomainError(seed, "Seed must be non-negative"))
    rng = isnothing(seed) ? MersenneTwister() : MersenneTwister(seed)

    n < 1 && throw(DomainError(n, "Tuple size must be at least 1"))

    actor = DiscreteActionPolicy{AS}(O, n, encoder; start_learning_rate, end_learning_rate, learning_rate_decay, epochs, partitioner=:uniform_random, seed)
    critic = RegressionDiscriminator{1}(OS, 32, encoder; seed, γ=forgetting_factor)

    buffer = MultiStepDynamicBuffer{O,Int}()

    DiscreteFunctionalAC{OS,AS,T,O,E}(steps, discount, actor, critic, buffer, nothing, false, nothing, rng)
end

function DiscreteFunctionalAC(
  env, encoder::E; steps, tuple_size, start_learning_rate, end_learning_rate, learning_rate_decay, epochs=1, discount=1.0, forgetting_factor=1.0, seed=nothing) where {T <: Real,E <: AbstractEncoder{T}}
  DiscreteFunctionalAC{length(action_set(env))}(observation_type(env), steps, tuple_size, start_learning_rate, end_learning_rate, learning_rate_decay, epochs, discount, forgetting_factor, encoder; seed)
end

function observe!(agent::DiscreteFunctionalAC{OS,AS,T,O,E}, observation::O) where {OS,AS,T <: Real,O <: StaticArray{Tuple{OS},T,1},E <: AbstractEncoder{T}}
  agent.observation = observation
  agent.done = false
  agent.action = _select_action(agent, observation)

  nothing
end

# TODO: All observe! methods are the same for all agents. Generalize.
# TODO: This method does not need to take in the action
function observe!(agent::DiscreteFunctionalAC{OS,AS,T,O,E}, action::Int, reward::Float64, observation::O, done::Bool) where {OS,AS,T <: Real,O <: StaticArray{Tuple{OS},T,1},E <: AbstractEncoder{T}}
    add!(agent.buffer, agent.observation, agent.action, reward)

    if !done
        agent.action = _select_action(agent, observation)
    end

    agent.observation = observation
    agent.done = done
    nothing
end

function _select_action(agent::DiscreteFunctionalAC{OS,AS,T,O,E}, observation::O) where {OS,AS,T <: Real,O <: StaticArray{Tuple{OS},T,1},E <: AbstractEncoder{T}}
  return select_action(agent.actor, observation)
end

function select_action!(agent::DiscreteFunctionalAC{OS,AS,T,O,E}, observation::O) where {OS,AS,T,O,E}
  if !isnothing(agent.action)
      return agent.action
  end

  return _select_action(agent, observation)
end

function update!(agent::DiscreteFunctionalAC)
  if agent.done
      G = 0.0
      for transition in Iterators.reverse(agent.buffer)
          G = transition.reward + agent.γ * G

          δ = G - (predict(agent.critic, transition.observation) |> first)
          update!(agent.actor, transition.observation, transition.action, δ)
          train!(agent.critic, transition.observation, G)
      end

      reset!(agent.buffer)
  elseif length(agent.buffer) == agent.steps
      G = predict(agent.critic, agent.observation) |> first

      for transition in Iterators.reverse(agent.buffer)
          G = transition.reward + agent.γ * G
      end

      t = popfirst!(agent.buffer)

      δ = G - (predict(agent.critic, t.observation) |> first)
      update!(agent.actor, t.observation, t.action, δ)
      train!(agent.critic, t.observation, G)
  end

  nothing
end

function reset!(agent::DiscreteFunctionalAC; seed::Union{Nothing,UInt}=nothing)
  if !isnothing(seed)
      if seed ≥ 0
          Random.seed!(agent.rng, seed)
      else
          throw(DomainError(seed, "Seed must be non-negative"))
      end
  end
    
  reset!(agent.actor; seed)
  Ramnet.reset!(agent.critic)
  reset!(agent.buffer)

  nothing
end