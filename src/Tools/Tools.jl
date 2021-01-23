module Tools

using Plots
using StatsBase

export learning_curve

# Plot a learning curve from the results of a experiment
function learning_curve(experiment_result)
    plot(mean(experiment_result; dims=2), legend=false)
end

end