mutable struct MultiStepDynamicBuffer{O,A} <: AbstractBuffer
    transitions::Vector{Transition}

    MultiStepDynamicBuffer{O,A}() where {O,A} = new(Transition[])
end

function add!(buffer::MultiStepDynamicBuffer{O,A}, obs::O, action::A, reward::Float64) where {O,A}
    push!(buffer.transitions, Transition(obs, action, reward))
end

reset!(buffer::MultiStepDynamicBuffer) = empty!(buffer.transitions)

function Base.iterate(buffer::MultiStepDynamicBuffer, state=1)
    state > length(buffer.transitions) && return nothing

    return buffer.transitions[state], state + 1
end

function Base.iterate(rbuffer::Base.Iterators.Reverse{MultiStepDynamicBuffer{O,A}}, state=length(rbuffer.itr.transitions)) where {O,A}
    state < 1 && return nothing

    return rbuffer.itr.transitions[state], state - 1
end

Base.length(buffer::MultiStepDynamicBuffer) = length(buffer.transitions)

Base.eltype(::MultiStepDynamicBuffer{O,A}) where {O,A} = Transition{O,A}

Base.popfirst!(buffer::MultiStepDynamicBuffer) = popfirst!(buffer.transitions)