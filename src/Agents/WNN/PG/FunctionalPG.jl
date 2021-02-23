using Ramnet.Encoders

using ..Approximators: BinaryActionPolicy
using ..Buffers: MultiStepDynamicBuffer, add!

mutable struct FunctionalPG{T <: Real,O <: AbstractVector{T},A <: Real,E <: AbstractEncoder{T}} <: AbstractAgent{O,A}
    γ::Float64
    policy::BinaryActionPolicy{T,O,E}
    buffer::MultiStepDynamicBuffer{O,A}
    observation::Union{Nothing,O}
    done::Bool
    action::Union{Nothing,A}
    rng::MersenneTwister

    function FunctionalPG{T,O,A,E}(actions, n, obs_size, η, encoder::E; seed::Union{Nothing,Int}=nothing) where {T <: Real,O <: AbstractVector{T},A <: Real,E <: AbstractEncoder{T}}
        !isnothing(seed) && seed < 0 && throw(DomainError(seed, "Seed must be non-negative"))
        rng = isnothing(seed) ? MersenneTwister() : MersenneTwister(seed)

        n < 1 && throw(DomainError(n, "Tuple size must be at least 1"))

        policy = BinaryActionPolicy{T,O,E}(obs_size, n, encoder; η, partitioner=:uniform, seed)
        buffer = MultiStepDynamicBuffer{O,A}()

        new(1, policy, buffer, nothing, false, nothing, rng)
    end
end

function FunctionalPG(env, encoder::E; n, η, seed=nothing) where {T <: Real,E <: AbstractEncoder{T}}
  FunctionalPG{T,observation_type(env),action_type(env),E}(action_set(env), n, observation_length(env), η, encoder; seed)
end

function observe!(agent::FunctionalPG{T,O,A,E}, observation::O) where {T <: Real,O <: AbstractVector{T},A <: Real,E <: AbstractEncoder{T}}
    agent.observation = observation
    agent.done = false

    nothing
end

# TODO: All observe! methods are the same for all agents. Generalize.
# TODO: This method does not need to take in the action
function observe!(agent::FunctionalPG{T,O,A,E}, action::A, reward::Float64, observation::O, done::Bool) where {T <: Real,O <: AbstractVector{T},A <: Real,E <: AbstractEncoder{T}}
  add!(agent.buffer, agent.observation, agent.action, reward)

  agent.observation = observation
  agent.done = done

  nothing
end

function select_action!(agent::FunctionalPG{T,O,A,E}, observation::O) where {T <: Real,O <: AbstractVector{T},A <: Real,E <: AbstractEncoder{T}}
    agent.action = select_action(agent.policy, observation)

    return agent.action
end

function update!(agent::FunctionalPG{T,O,A,E}) where {T <: Real,O <: AbstractVector{T},A <: Real,E <: AbstractEncoder{T}}
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

function Agents.reset!(agent::FunctionalPG{T,O,A,E}) where {T <: Real,O <: AbstractVector{T},A <: Real,E <: AbstractEncoder{T}}
  reset!(agent.policy)
  reset!(agent.buffer)

  nothing
end
