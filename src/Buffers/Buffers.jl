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

include("FixedSizeBuffer.jl")
export FixedSizeBuffer, take_last, sample!

end