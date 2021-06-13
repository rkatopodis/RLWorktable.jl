using Ramnet.Encoders
using Ramnet.Models.AltDiscriminators:RegressionDiscriminator

using LinearAlgebra:Diagonal

using ..Approximators:RegularizedContinousPolicy, VDiscriminator
# using ..Buffers: MultiStepDynamicBuffer, add!
using ..Buffers: FixedSizeBuffer, add!, sample!, take_last, get

mutable struct ContinousRegularizedAC{OS,AS,T <: Real,O <: StaticArray{Tuple{OS},T,1},A <: StaticArray{Tuple{AS},T,1},E <: AbstractEncoder{T}} <: AbstractAgent{O,A}
    steps::Int
    γ::Float64
    batchsize::Int
    actor::RegularizedContinousPolicy{OS,AS,T,O,A,E}
    critic::VDiscriminator
    # buffer::MultiStepDynamicBuffer{O,A}
    buffer::FixedSizeBuffer{O,A}
    observation::Union{Nothing,O}
    done::Bool
    action::Union{Nothing,A}
    min_obs::Union{Nothing,O}
    max_obs::Union{Nothing,O}
    unevaluated_experiences::Int
    # rng::MersenneTwister
end

function ContinousRegularizedAC(::Type{O}, ::Type{A}, steps, n, α, η, batchsize, epochs, discount, forgetting_factor, encoder::E, start_cov::Float64, end_cov::Float64, cov_decay::Int, buffersize::Int; seed::Union{Nothing,Int}=nothing) where {OS,AS,T <: Real,O <: StaticArray{Tuple{OS},T,1},A <: StaticArray{Tuple{AS},T,1},E <: AbstractEncoder{T},C <: AbstractMatrix{T}}
    # !isnothing(seed) && seed < 0 && throw(DomainError(seed, "Seed must be non-negative"))
    rng = isnothing(seed) ? MersenneTwister() : MersenneTwister(seed)

    n < 1 && throw(DomainError(n, "Tuple size must be at least 1"))

    actor = RegularizedContinousPolicy(O, A, n, encoder;α, start_cov, end_cov, cov_decay, η, epochs, partitioner=:uniform_random, seed)
    # critic = RegressionDiscriminator{1}(OS, 32, encoder; seed, γ=forgetting_factor)
    critic = VDiscriminator(O, 32, forgetting_factor, encoder, seed)

    # buffer = MultiStepDynamicBuffer{O,A}()
    buffer = FixedSizeBuffer{O,A}(buffersize, rng)

    ContinousRegularizedAC{OS,AS,T,O,A,E}(steps, discount, batchsize, actor, critic, buffer, nothing, false, nothing, nothing, nothing, 0)
end

function ContinousRegularizedAC(env, encoder::E; steps, tuple_size, alpha, learning_rate, batchsize, epochs, discount, forgetting_factor=1.0, start_cov, end_cov, cov_decay, buffersize, seed=nothing) where {T <: Real,E <: AbstractEncoder{T}}
    ContinousRegularizedAC(observation_type(env), action_type(env), steps, tuple_size, alpha, learning_rate, batchsize, epochs, discount, forgetting_factor, encoder, start_cov, end_cov, cov_decay, buffersize; seed)
end

function observe!(agent::ContinousRegularizedAC{OS,AS,T,O,A,E}, observation::O) where {OS,AS,T,O,A,E}
    agent.observation = observation
    agent.done = false

    agent.action = _select_action(agent, observation)

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
function observe!(agent::ContinousRegularizedAC{OS,AS,T,O,A,E}, action::A, reward::Float64, observation::O, done::Bool) where {OS,AS,T,O,A,E}
    # if reward < 0
    #     println(reward)
    # end
    
    add!(agent.buffer, agent.observation, agent.action, reward)
    agent.unevaluated_experiences += 1

    if !done
        agent.action = _select_action(agent, observation)
    end

    agent.observation = observation
    agent.done = done

  # Finding the edges of the domain
    agent.min_obs = min.(agent.min_obs)
    agent.max_obs = max.(agent.max_obs)

    nothing
end

function _select_action(agent::ContinousRegularizedAC{OS,AS,T,O,A,E}, observation::O) where {OS,AS,T,O,A,E}
    select_action(agent.actor, observation)
end

function select_action!(agent::ContinousRegularizedAC{OS,AS,T,O,A,E}, observation::O) where {OS,AS,T,O,A,E}
    if !isnothing(agent.action)
        return agent.action
    end
  
    return _select_action(agent, observation)
end

function update!(agent::ContinousRegularizedAC)
    if agent.done
        G = 0.0
        # for transition in Iterators.reverse(agent.buffer)
        #     G = transition.reward + agent.γ * G
  
        #     δ = G - (predict(agent.critic, transition.observation) |> first)
        #     update!(agent.actor, transition.observation, transition.action, δ)
        #     train!(agent.critic, transition.observation, G)
        # end

        # reset!(agent.buffer)

        for ex in take_last(agent.buffer, agent.unevaluated_experiences)
            G = ex.reward + agent.γ * G
            ex.value = G

            # Não estou fazendo uso de vantagem aqui! (diferença entre valor previso e observado)
            update!(agent.actor, sample!(agent.buffer, agent.batchsize), agent.critic)
            # train!(agent.critic, ex.observation, G)
            update!(agent.critic, ex.observation, G)
        end

        agent.unevaluated_experiences = 0

    # Preste atenção nesse teste! Como você não remove mais experiências do buffer (popfirst), essa
    # condição só será satisfeita uma única vez! Não é o comportamento desejado!
    # talvez length(agent.buffer) > agent.steps? Não exatamente. Preciso que já tenham ocorrido
    # agent.steps no episódio corrente. Posso usar agent.unevaluated_experiences!
    elseif agent.unevaluated_experiences == agent.steps
        # G = predict(agent.critic, agent.observation) |> first
        G = agent.critic(agent.observation)
  
        # for transition in Iterators.reverse(agent.buffer)
        #     G = transition.reward + agent.γ * G
        # end
  
        # t = popfirst!(agent.buffer)
  
        # δ = G - (predict(agent.critic, t.observation) |> first)
        # update!(agent.actor, t.observation, t.action, δ)
        # train!(agent.critic, t.observation, G)

        for ex in take_last(agent.buffer, agent.steps)
            G = ex.reward + agent.γ * G
        end

        ex = get(agent.buffer, agent.buffer.tail - agent.steps)
        ex.value = G

        # Não estou fazendo uso de vantagem aqui! (diferença entre valor previso e observado)
        update!(agent.actor, sample!(agent.buffer, agent.batchsize), agent.critic)
        # train!(agent.critic, ex.observation, G)
        update!(agent.critic, ex.observation, G)

        agent.unevaluated_experiences -= 1
    end
  
    nothing
end

function reset!(agent::ContinousRegularizedAC; seed::Union{Nothing,Int}=nothing)
    if !isnothing(seed)
        if seed ≥ 0
        seed!(agent.rng, seed)
        else
            throw(DomainError(seed, "Seed must be non-negative"))
        end
    end
    
    reset!(agent.actor)
    # Ramnet.reset!(agent.critic)
    reset!(agent.critic)
    reset!(agent.buffer)

    nothing
end
