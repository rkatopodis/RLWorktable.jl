module GymWrappers

import ..AbstractEnvironment, ..reset!, ..render, ..close, ..step!

using PyCall

export GymCartPoleV1, GymMountainCarDiscrete

const gym = PyNULL()

abstract type AbstractGymEnvironment <: AbstractEnvironment end

reset!(env::AbstractGymEnvironment) = 0.0, env.env.reset(), false
render(env::AbstractGymEnvironment) = env.env.render()
close(env::AbstractGymEnvironment)  = env.env.close()

# --------------------------------- CartPole --------------------------------- #
struct GymCartPoleV1 <: AbstractGymEnvironment
    env::PyObject

    GymCartPoleV1() = new(gym.make("CartPole-v1"))
end

# reset!(env::GymCartPoleV1) = 0.0, env.env.reset(), false
# render(env::GymCartPoleV1) = env.env.render()
# close(env::GymCartPoleV1) = env.env.close()

function step!(env::GymCartPoleV1, action::Int)
    (action != 0 && action != 1) && throw(DomainError(action, "Invalid action"))

    obs, reward, done, _ = env.env.step(action)

    return reward, obs, done
end

# -------------------------- MontainCar (Discrete) --------------------------- #
struct GymMountainCarDiscrete <: AbstractGymEnvironment
    env::PyObject

    GymMountainCarDiscrete() = new(gym.make("MountainCar-v0"))
end

function step!(env::GymMountainCarDiscrete, action::Int)
    (action != 0 && action != 1 && action != 2) && throw(DomainError(action, "Invalid action"))

    obs, reward, done, _ = env.env.step(action)

    return reward, obs, done
end

function __init__()
    copy!(gym, pyimport_conda("gym", "gym", "conda-forge"))
end

end