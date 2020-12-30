module Agents

export AbstractAgent, observe_first!, select_action, observe!, update!, RandomAgent

abstract type AbstractAgent end

function observe_first! end
function select_action end
function observe! end
function update! end

include("RandomAgent.jl")

end