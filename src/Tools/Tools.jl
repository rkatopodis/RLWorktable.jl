module Tools

using Plots
pyplot()

using Statistics
using StatsBase

export learning_curve

function moving_average(a::AbstractVecOrMat, n::Int)
    @assert 1 <= n <= size(a, 1)
    out = similar(a, size(a, 1) - n + 1, size(a, 2))
    out[1, :] = mean(a[1:n, :], dims=1)
    for i in axes(out, 1)[2:end]
        out[i, :] = (n * out[i - 1, :] .- a[i - 1, :] .+ a[i + n - 1, :]) / n
    end
    return out
end

# Plot a learning curve from the results of a experiment
function learning_curve(experiment_result, ma::Int=1)
    smoothed = moving_average(experiment_result, ma)

    mean_series = mean(smoothed; dims=2)
    sd = std(smoothed; dims=2, mean=mean_series)

    # smoothed = (ma == 1) ? series : moving_average(series, ma)

    plot(mean_series, ribbon=sd, legend=false)
end

end