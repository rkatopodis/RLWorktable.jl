struct DynamicExperienceBuffer{O,A} <: AbstractExperienceBuffer
    observations::Vector{O}
    actions::Vector{A}
    rewards::Vector{Float64}
    terminals::Vector{Bool}

    DynamicExperienceBuffer{O,A}() where {O,A} = new(Vector{O}(), A[], Float64[], Bool[])
end

DynamicBinaryBuffer{A} = DynamicExperienceBuffer{Vector{Bool},A}

function add!(buffer::DynamicExperienceBuffer{O,A}, observation::O) where {O,A}
    length(buffer.observations) != length(buffer.actions) && error(
      "Incomplete transition"
    )

    push!(buffer.observations, observation)
    push!(buffer.terminals, false)
    nothing
end

function add!(buffer::DynamicExperienceBuffer{O,A}, action::A, reward::Float64, next_obs::O, terminal::Bool) where {O,A}
    length(buffer.observations) == length(buffer.actions) && error(
      "Incomplete transition"
    )

    push!(buffer.actions, action)
    push!(buffer.rewards, reward)
    push!(buffer.observations, next_obs)
    push!(buffer.terminals, terminal)

    nothing
end

function reset!(buffer::DynamicExperienceBuffer)
    empty!(buffer.observations)
    empty!(buffer.actions)
    empty!(buffer.rewards)
    empty!(buffer.terminals)

    nothing
end

function Base.iterate(buffer::DynamicExperienceBuffer, state=1)
    state > length(buffer.actions) && return nothing

    return ((buffer.observations[state],
        buffer.actions[state],
        buffer.rewards[state],
        buffer.observations[state + 1],
        buffer.terminals[state + 1]),
      state + 1)
end

function Base.iterate(rbuffer::Base.Iterators.Reverse{DynamicExperienceBuffer{O,A}}, state=length(rbuffer.itr.actions)) where {O,A}
    state < 1 && return nothing

    return ((rbuffer.itr.observations[state],
        rbuffer.itr.actions[state],
        rbuffer.itr.rewards[state],
        rbuffer.itr.observations[state + 1],
        rbuffer.itr.terminals[state + 1]),
      state - 1)
end

Base.length(buffer::DynamicExperienceBuffer) = length(buffer.actions)
Base.eltype(buffer::DynamicExperienceBuffer{O,A}) where {O,A} = Tuple{O,A,Float64,O,Bool}