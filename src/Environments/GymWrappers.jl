module GymWrappers

import ..AbstractEnvironment, ..reset!, ..render, ..close, ..step!

using PyCall

export GymCartPoleV1

const gym = PyNULL()

struct GymCartPoleV1 <: AbstractEnvironment
    env::PyObject

    GymCartPoleV1() = new(gym.make("CartPole-v1"))
end

reset!(env::GymCartPoleV1) = 0.0, env.env.reset(), false
render(env::GymCartPoleV1) = env.env.render()
close(env::GymCartPoleV1) = env.env.close()

function step!(env::GymCartPoleV1, action::Int)
    (action != 0 && action != 1) && throw(DomainError(action, "Invalid action"))

    obs, reward, done, _ = env.env.step(action)

    return reward, obs, done
end

function __init__()
    copy!(gym, pyimport_conda("gym", "gym", "conda-forge"))
end

end