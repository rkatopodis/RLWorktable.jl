using Ramnet
using Ramnet.Encoders
using RLWorktable

using BenchmarkTools
using ProfileView

function cartpole_test(episodes=1)
    my_env = CartPoleV1()

    cartpole_state_min = [-4.8, -4.0, -0.418, -3.0]
    cartpole_state_max = [4.8, 4.0, 0.418, 3.0]
    
    res = 270
    n = 30

    obs_size = 4
    input_size = 4 * res # Each of the four components of the state of the carpole will have res bits
    ϵ = 0.01 # Exploration probability
    α = 0.8  # Step size
    γ = 1.0

    thermo = CircularThermometer(cartpole_state_min, cartpole_state_max, res)

    new_agent = MCDiscriminatorAgent(
    my_env, thermo; n, style=:discounted, α, gamma=γ, epsilon=ϵ, seed=1)

    loop = EnvironmentLoop(my_env, new_agent)

    rewards = simulate(loop; episodes, show_progress=false);
end