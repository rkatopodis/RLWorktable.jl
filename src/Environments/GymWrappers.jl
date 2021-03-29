module GymWrappers

import ..AbstractEnvironment,
    ..reset!,
    ..set_render_mode!,
    ..render,
    ..close,
    ..step!,
    ..observation_type,
    ..observation_extrema,
    ..observation_length,
    ..action_type,
    ..action_set,
    ..action_length

using PyCall
using StaticArrays

export GymCartPoleV1,
    GymMountainCarDiscrete, GymAcrobot, GymBreakoutRAM, GymInvertedPendulum, GymHopperBulletV0

const gym = PyNULL()
const pybullet_envs = PyNULL()

abstract type AbstractGymEnvironment{O <: AbstractVector,A} <:
              AbstractEnvironment{O,A} end

reset!(env::AbstractGymEnvironment) = 0.0, env.env.reset(), false
set_render_mode!(env::AbstractGymEnvironment, mode::Symbol) = nothing
function render(env::AbstractGymEnvironment, mode::Union{Nothing,Symbol}=nothing)
    isnothing(mode) && return env.env.render()

    env.env.render(mode="human")
end
close(env::AbstractGymEnvironment) = env.env.close()

# --------------------------------- CartPole --------------------------------- #
struct GymCartPoleV1 <: AbstractGymEnvironment{SizedVector{4,Float64},Int}
    env::PyObject

    GymCartPoleV1() = new(gym.make("CartPole-v1"))
end

observation_type(::GymCartPoleV1) = SizedVector{4,Float64}
observation_extrema(::GymCartPoleV1) =
    (SVector(-4.8, -4.0, -0.418, -3.0), SVector(4.8, 4.0, 0.418, 3.0))
observation_length(::GymCartPoleV1) = 4
action_type(::GymCartPoleV1) = Int
action_set(::GymCartPoleV1) = SA[-1, 1]

function step!(env::GymCartPoleV1, action::Int)
    (action != -1 && action != 1) &&
        throw(DomainError(action, "Invalid action"))

    obs, reward, done, _ = env.env.step(action == -1 ? 0 : 1)
    
    return reward, SizedVector{4}(obs), done
end

reset!(env::GymCartPoleV1) = 0.0, SizedVector{4}(env.env.reset()), false

# -------------------------- MontainCar (Discrete) --------------------------- #
struct GymMountainCarDiscrete <: AbstractGymEnvironment{Vector{Float64},Int}
    env::PyObject
    
    function GymMountainCarDiscrete(steps::Int)
        env = gym.make("MountainCar-v0")
        env._max_episode_steps = steps

        new(env)
    end
end

# TODO: Validate
GymMountainCarDiscrete(; steps200) = GymMountainCarDiscrete(steps)

observation_type(::GymMountainCarDiscrete) = Vector{Float64}
observation_extrema(::GymMountainCarDiscrete) = ([-1.2, -0.07], [0.6, 0.07])
observation_length(::GymMountainCarDiscrete) = 2
action_type(::GymMountainCarDiscrete) = Int
action_set(::GymMountainCarDiscrete) = SA[0,1,2]

function step!(env::GymMountainCarDiscrete, action::Int)
    (action != 0 && action != 1 && action != 2) &&
        throw(DomainError(action, "Invalid action"))
    
    obs, reward, done, _ = env.env.step(action)
    
    return reward, obs, done
end

# --------------------------------- Acrobot ---------------------------------- #
struct GymAcrobot <: AbstractGymEnvironment{Vector{Float64},Int}
    env::PyObject
    
    function GymAcrobot(steps::Int)
        env = gym.make("Acrobot-v1")
        env._max_episode_steps = steps

        new(env)
    end
end

# TODO: Validate
GymAcrobot(; steps=500) = GymAcrobot(steps)

observation_type(::GymAcrobot) = Vector{Float64}
observation_extrema(::GymAcrobot) = (fill(-1, 6), fill(1, 6))
observation_length(::GymAcrobot) = 6
action_type(::GymAcrobot) = Int
action_set(::GymAcrobot) = SA[-1, 0, 1]

function step!(env::GymAcrobot, action::Int)
    (action != 0 && action != 1 && action != -1) &&
        throw(DomainError(action, "Invalid action"))
    
    obs, reward, done, _ = env.env.step(action)
    
    return reward, obs, done
end
# =============================== Atari ====================================== #
# ----------------------------- Breakout (RAM) ------------------------------- #
struct GymBreakoutRAM <: AbstractGymEnvironment{Vector{UInt8},Int}
    env::PyObject
    
    GymBreakoutRAM() = new(gym.make("Breakout-ram-v0"))
end

observation_type(::GymBreakoutRAM) = Vector{UInt8}
observation_extrema(::GymBreakoutRAM) = (typemin(UInt8), typemax(UInt8))
observation_length(::GymBreakoutRAM) = 128
action_type(::GymBreakoutRAM) = Int
action_set(::GymBreakoutRAM) = 0:3

function step!(env::GymBreakoutRAM, action::Int)
    !(action in action_set(env)) && throw(DomainError(action, "Invalid action"))
    
    obs, reward, done, _ = env.env.step(action)
    
    return reward, obs, done
end

# ================================== Mujoco ================================== #
# --------------------------- InvertedPendulum-v2 ---------------------------- #
struct GymInvertedPendulum <: AbstractGymEnvironment{Vector{Float64},Float64}
    env::PyObject
    
    GymInvertedPendulum() = new(gym.make("InvertedPendulum-v2"))
end

observation_type(::GymInvertedPendulum) = Vector{Float64}
observation_extrema(::GymInvertedPendulum) =
    ([-4.8, -4.0, -0.418, -3.0], [4.8, 4.0, 0.418, 3.0])
observation_length(::GymInvertedPendulum) = 4
action_type(::GymInvertedPendulum) = Float64
action_set(::GymInvertedPendulum) = -3.0:1:3.0

# ================================= pybullet ================================= #
# ---------------------------- HopperBulletEnv-v0 ---------------------------- #
struct GymHopperBulletV0 <:
       AbstractGymEnvironment{Vector{Float32},SVector{3,Float32}}
    env::PyObject
    
    GymHopperBulletV0() = new(gym.make("HopperBulletEnv-v0"))
end

observation_type(::GymHopperBulletV0) = Vector{Float32}
observation_extrema(::GymHopperBulletV0) = error("Not implemented")
observation_length(::GymHopperBulletV0) = 15
action_type(::GymHopperBulletV0) = SVector{3,Float32} # Vector{Float32}
action_length(::GymHopperBulletV0) = 3
action_set(::GymHopperBulletV0) = error("Not implemented")

set_render_mode!(env::GymHopperBulletV0, mode::Symbol=:human) = render(env, mode)

function step!(env::GymHopperBulletV0, action::StaticArray{Tuple{3},Float32,1})
    # !env.env.action_space.contains(action) && throw(DomainError(action, "Invalid action"))
    
    obs, reward, done, _ = env.env.step(action)
    
    return reward, obs, done
end

close(env::GymHopperBulletV0) = nothing

function __init__()
    copy!(gym, pyimport_conda("gym", "gym", "conda-forge"))
    copy!(pybullet_envs, pyimport_conda("pybullet_envs", "pybullet"))
end

end
