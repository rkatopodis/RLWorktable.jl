module Buffers

# TODO: Static buffers!!!

import ..reset!

export add!

abstract type AbstractBuffer end

function add! end
# function reset! end

struct Transition{O,A}
    observation::O
    action::A
    reward::Float64
end

function MonteCarloExperiences end
function TDExperiences end

include("DynamicBuffer.jl")
export DynamicBuffer, DynamicBinaryBuffer

include("SingleStepBuffer.jl")
export SingleStepBuffer, ready

include("MultiStepDynamicBuffer.jl")
export MultiStepDynamicBuffer

struct StaticBuffer{O,A,R} <: AbstractBuffer
    size::Int
    observations::Vector{O}
    actions::Vector{A}
    rewards::Vector{R}
    next_obs::Int
    next_action::Int

    function StaticBuffer{O,A,R}(size::Int) where {O,A,R}
        maxsize ≤ 0 && throw(DomainError(
            size, "Buffer size must be greater then zero."
        ))

        new(
            size,
            Vector{O}(undef, size),
            Vector{A}(undef, size),
            Vector{R}(undef, size),
            1,
            1
        )
    end
end

function full(buffer::StaticBuffer)
    (buffer.next_obs == buffer.next_action) && buffer.next_action > buffer.size
end

function add!(buffer::StaticBuffer{O,A,R}, obs::O) where {O,A,R}
    full(buffer) && error("Buffer is full") # TODO: Better exception
    buffer.next_obs != buffer.next_action && error("Incomplete transition")

    buffer.observations[buffer.next_obs] = obs
    buffer.next_obs += 1

    nothing
end

function add!(buffer::StaticBuffer{O,A,R}, action::A, reward::R, obs::O) where {O,A,R}
    full(buffer) && error("Buffer is full") # TODO: Better exception
    buffer.next_obs == buffer.next_action && error("Incomplete transition")

    i = buffer.next_action
    buffer.actions[i] = action
    buffer.rewards[i] = reward
    buffer.next_action += 1

    if buffer.next_obs ≤ buffer.size
        buffer.observations[buffer.next_obs] = obs
        buffer.next_obs += 1
    end

    nothing
end

# Returns (obs, action, reward, next_obs, next_action, done) pairs. If a transition
# leads to a terminal state, next_obs will be zero(O) and next_action, zero(A)
function Base.iterate(buffer::StaticBuffer{O,A,R}, state=1) where {O,A,R}
    state == buffer.next_obs && return nothing

    
end

struct EpisodicBuffer{O,A,R} <: AbstractBuffer
    maxsize::Int
    episodes::Int
    observations::AbstractVector{O} # In the context of weightless neural networks,
                                    # individual observations will be binary vectors.
                                    # Shouldn't I just make this a binary matrix?
    actions::AbstractVector{A}
    rewards::AbstractVector{R}
end

end