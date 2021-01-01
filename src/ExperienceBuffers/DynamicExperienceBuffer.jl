struct DynamicExperienceBuffer{O,A,R} <: AbstractExperienceBuffer
    observations::Vector{O}
    actions::Vector{A}
    rewards::Vector{R}
    terminals::Vector{Bool}

    DynamicExperienceBuffer{O,A,R}() where {O,A,R} = new(Vector{O}(), A[], R[], Bool[])
end

function add!(buffer::DynamicExperienceBuffer{O,A,R}, observation::O) where {O,A,R}
    length(buffer.observations) != length(buffer.actions) && error(
      "Incomplete transition"
    )

    push!(buffer.observations, observation)
    push!(buffer.terminals, false)
    nothing
end

function add!(buffer::DynamicExperienceBuffer{O,A,R}, action::A, reward::R, next_obs::O, terminal::Bool) where {O,A,R}
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

function Base.iterate(rbuffer::Base.Iterators.Reverse{DynamicExperienceBuffer{O,A,R}}, state=length(rbuffer.itr.actions)) where {O,A,R}
    state < 1 && return nothing

    return ((rbuffer.itr.observations[state],
        rbuffer.itr.actions[state],
        rbuffer.itr.rewards[state],
        rbuffer.itr.observations[state + 1],
        rbuffer.itr.terminals[state + 1]),
      state - 1)
end

Base.length(buffer::DynamicExperienceBuffer) = length(buffer.actions)
Base.eltype(buffer::DynamicExperienceBuffer{O,A,R}) where {O,A,R} = Tuple{O,A,R,O,Bool}