using Random
using Distributions

mutable struct CartPole <: AbstractEnvironment{Vector{Float64},Int}
    max_steps::Int
    steps::Int
    state::Vector{Float64}
    done::Bool
    rng::MersenneTwister

    function CartPole(max_steps::Int; seed=Union{Nothing,Int} = nothing)
        max_steps < 1 && throw(DomainError(max_steps, "max_steps may not be lesser then 1"))
        !isnothing(seed) && seed < 0 && throw(DomainError(seed, "Seed must be non-negative"))
        rng = isnothing(seed) ? MersenneTwister() : MersenneTwister(seed)

        new(max_steps, 0, Vector{Float64}(undef, 4), false, rng)
    end
end

CartPole(; steps, seed) = CartPole(steps; seed)
CartPoleV1(;seed=nothing) = CartPole(500; seed)

function reset!(env::CartPole)
    rand!(env.rng, Uniform(-0.05, 0.05), env.state)
    env.done = false
    env.steps = 0

    return 0.0, env.state, env.done
end

observation_type(::CartPole) = Vector{Float64}
observation_extrema(::CartPole) = ([-4.8, -4.0, -0.418, -3.0], [4.8, 4.0, 0.418, 3.0])
observation_length(::CartPole) = 4
action_type(::CartPole) = Int
action_set(::CartPole) = -1:2:1

# TODO: Check if env has been reseted
function step!(env::CartPole, action::Int)
    env.done && error("Cannot step after episode's end. Reset environment")
    (action != -1 && action != 1) && throw(DomainError(action, "Invalid action"))

    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = masspole + masscart
    length = 0.5  # actually half the pole's length
    polemass_length = masspole * length
    force_mag = 10.0
    tau = 0.02  # seconds between state updates

  # Angle at which to fail the episode
    theta_threshold_radians = 12 * 2 * pi / 360
    x_threshold = 2.4

    x, x_dot, theta, theta_dot = env.state
    force = (action == 1) ? force_mag : -force_mag
    costheta = cos(theta)
    sintheta = sin(theta)

    temp = (force + polemass_length * theta_dot^2 * sintheta) / total_mass
    thetaacc = (gravity * sintheta - costheta * temp) / (length * (4.0 / 3.0 - masspole * costheta^2 / total_mass))
    xacc = temp - polemass_length * thetaacc * costheta / total_mass

    env.state[1] += tau * x_dot
    env.state[2] += tau * xacc
    env.state[3] += tau * theta_dot
    env.state[4] += tau * thetaacc
    env.steps += 1
    env.done = x < -x_threshold ||
               x > x_threshold ||
               theta < -theta_threshold_radians ||
               theta > theta_threshold_radians ||
               env.steps ≥ env.max_steps

    # If no copy is made, all transitions in a buffer are going to end up
    # pointing to the same state.
    # The copying might be unecessary if I were to use StructArrays in the
    # buffer (???)
    # TODO: Think of a more elegant and/or efficient solution for the problem
    #       described above.
    return 1.0, env.state[:], env.done
end
