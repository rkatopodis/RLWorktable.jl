using Random
using Distributions

mutable struct CartPole <: AbstractEnvironment
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

CartPoleV1(;seed=nothing) = CartPole(500; seed)

function reset!(env::CartPole)
    rand!(env.rng, Uniform(-0.05, 0.05), env.state)
    env.done = false
    env.steps = 0

    return 0.0, env.state, env.done
end

# TODO: Check if env has been reseted
function step!(env::CartPole, action::Int)
    env.done && error("Cannot step after episode's end. Reset environment")
    (action != 0 && action != 1) && throw(DomainError(action, "Invalid action"))

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

  # state_max = Float64[
  #   2* x_threshold
  #   typemax(Float64),
  #   2 * theta_threshold_radians,
  #   typemax(Float64)
  # ]

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

    return 1.0, env.state, env.done 
end