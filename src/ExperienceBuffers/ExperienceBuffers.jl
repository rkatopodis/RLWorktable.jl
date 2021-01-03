module ExperienceBuffers

export add!, reset!

abstract type AbstractExperienceBuffer end

function add! end
function reset! end

function MonteCarloExperiences end
function TDExperiences end

include("DynamicExperienceBuffer.jl")
export DynamicExperienceBuffer, DynamicBinaryBuffer

struct StaticExperienceBuffer{O,A,R} <: AbstractExperienceBuffer
    size::Int
    observations::Vector{O}
    actions::Vector{A}
    rewards::Vector{R}
    next_obs::Int
    next_action::Int

    function StaticExperienceBuffer{O,A,R}(size::Int) where {O,A,R}
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

function full(buffer::StaticExperienceBuffer)
    (buffer.next_obs == buffer.next_action) && buffer.next_action > buffer.size
end

function add!(buffer::StaticExperienceBuffer{O,A,R}, obs::O) where {O,A,R}
    full(buffer) && error("Buffer is full") # TODO: Better exception
    buffer.next_obs != buffer.next_action && error("Incomplete transition")

    buffer.observations[buffer.next_obs] = obs
    buffer.next_obs += 1

    nothing
end

function add!(buffer::StaticExperienceBuffer{O,A,R}, action::A, reward::R, obs::O) where {O,A,R}
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
function Base.iterate(buffer::StaticExperienceBuffer{O,A,R}, state=1) where {O,A,R}
    state == buffer.next_obs && return nothing

    
end

struct EpisodicBuffer{O,A,R} <: AbstractExperienceBuffer
    maxsize::Int
    episodes::Int
    observations::AbstractVector{O} # In the context of weightless neural networks,
                                    # individual observations will be binary vectors.
                                    # Shouldn't I just make this a binary matrix?
    actions::AbstractVector{A}
    rewards::AbstractVector{R}
end

end