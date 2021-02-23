module GymWrappers

import ..AbstractEnvironment,
..reset!,
..render,
..close,
..step!,
..observation_type,
..observation_extrema,
..observation_length,
..action_type,
..action_set

using PyCall

export GymCartPoleV1, GymMountainCarDiscrete, GymAcrobot, GymBreakoutRAM, GymInvertedPendulum

const gym = PyNULL()

abstract type AbstractGymEnvironment{O <: AbstractVector,A <: Real} <: AbstractEnvironment{O,A} end

reset!(env::AbstractGymEnvironment) = 0.0, env.env.reset(), false
render(env::AbstractGymEnvironment) = env.env.render()
close(env::AbstractGymEnvironment)  = env.env.close()

# --------------------------------- CartPole --------------------------------- #
struct GymCartPoleV1 <: AbstractGymEnvironment{Vector{Float64},Int}
    env::PyObject

    GymCartPoleV1() = new(gym.make("CartPole-v1"))
end

observation_type(::GymCartPoleV1) = Vector{Float64}
observation_extrema(::GymCartPoleV1) = ([-4.8, -4.0, -0.418, -3.0], [4.8, 4.0, 0.418, 3.0])
observation_length(::GymCartPoleV1) = 4
action_type(::GymCartPoleV1) = Int
action_set(::GymCartPoleV1) = -1:2:1

function step!(env::GymCartPoleV1, action::Int)
    (action != -1 && action != 1) && throw(DomainError(action, "Invalid action"))

    obs, reward, done, _ = env.env.step(action == -1 ? 0 : 1)

    return reward, obs, done
end

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
action_type(::GymMountainCarDiscrete) = Int

function step!(env::GymMountainCarDiscrete, action::Int)
    (action != 0 && action != 1 && action != 2) && throw(DomainError(action, "Invalid action"))

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
action_type(::GymAcrobot) = Int

function step!(env::GymAcrobot, action::Int)
    (action != 0 && action != 1 && action != -1) && throw(DomainError(action, "Invalid action"))

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
observation_extrema(::GymInvertedPendulum) = ([-4.8, -4.0, -0.418, -3.0], [4.8, 4.0, 0.418, 3.0])
observation_length(::GymInvertedPendulum) = 4
action_type(::GymInvertedPendulum) = Float64
action_set(::GymInvertedPendulum) = -3.0:1:3.0

function __init__()
    copy!(gym, pyimport_conda("gym", "gym", "conda-forge"))
end

end
