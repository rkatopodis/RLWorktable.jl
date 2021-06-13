using Random

mutable struct Experience{O,A}
    observation::O
    action::A
    reward::Float64
    value::Float64
end

mutable struct FixedSizeBuffer{O,A} <: AbstractBuffer
    experiences::Vector{Experience{O,A}}
    buffersize::Int
    tail::Int
    len::Int
    sample_order::Vector{Int}
    last_sample::Int

    function FixedSizeBuffer{O,A}(buffersize::Int, rng::MersenneTwister) where {O,A}
        v = Vector{Experience{O,A}}(undef, buffersize)

        new(v, buffersize, 1, 0, randperm(rng, buffersize), 1)
    end
end

function add!(buffer::FixedSizeBuffer{O,A}, obs::O, action::A, reward::Float64) where {O,A}
    buffer.experiences[buffer.tail] = Experience{O,A}(obs, action, reward, 0.0)
    buffer.tail = mod1(buffer.tail + 1, buffer.buffersize)
    buffer.len = buffer.len == buffer.buffersize ? buffer.buffersize : buffer.len + 1 
    
    nothing
end

Base.length(buffer::FixedSizeBuffer) = buffer.len

# Should i conform to the AbstractArray interface?
get(buffer::FixedSizeBuffer, i::Int) = buffer.experiences[mod1(i, buffer.len)]

# Preciso de um mecanismo para iterar pelas últimas n experiências (para atualizar seus valores)
struct BufferIterator{O,A}
    buffer::FixedSizeBuffer{O,A}
    len::Int
end

function Base.iterate(iter::BufferIterator, state=1)
    state > iter.len && return nothing

    return (get(iter.buffer, iter.buffer.tail - state), state + 1)
end

Base.eltype(::Type{BufferIterator{O,A}}) where {O,A} = Experience{O,A}

Base.length(iter::BufferIterator) = iter.len

take_last(buffer::FixedSizeBuffer, n::Int) = BufferIterator(buffer, n)

# Preciso de um mecanismo para amostrar um minibatch do buffer

function sample!(buffer::FixedSizeBuffer{O,A}, batchsize::Int) where {O,A}
    batch = Vector{Experience{O,A}}(undef, batchsize)

        for i in 1:batchsize
        batch[i] = get(buffer, buffer.last_sample + i)
    end

    buffer.last_sample = mod1(buffer.last_sample + batchsize, buffer.buffersize)

    return batch
end