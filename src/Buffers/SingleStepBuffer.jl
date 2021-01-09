mutable struct SingleStepBuffer{O,A} <: AbstractBuffer
    transition::Union{Nothing,Transition{O,A}}

    SingleStepBuffer{O,A}() where {O,A} = new(nothing)
end

function add!(buffer::SingleStepBuffer{O,A}, obs::O, action::A, reward::Float64) where {O,A}
    buffer.transition = Transition(obs, action, reward)
end

ready(buffer::SingleStepBuffer) = !isnothing(buffer.transition)

reset!(buffer::SingleStepBuffer) = buffer.transition = nothing

function Base.iterate(buffer::SingleStepBuffer, state=1)
    state != 1 && return nothing

    return buffer.transition, 0
end

Base.length(buffer::SingleStepBuffer) = isnothing(buffer.transition) ? 0 : 1

Base.eltype(buffer::SingleStepBuffer{O,A}) where {O,A} = Transition{O,A}