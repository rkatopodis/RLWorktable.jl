module Tools

using Plots
pyplot()

using Statistics
using StatsBase
using Bootstrap

using ..Sessions

export learning_curve, Uniform, curve_conf, curves, average_series

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

    plot(mean_series, ribbon=sd, legend=false)
end

function average_series(returns::AbstractMatrix; bs=1000, ci=0.95, window=5)
    bs_mean = Matrix{Float64}(undef, size(returns, 1), 3)
    for (i, samples) in Iterators.enumerate(eachrow(returns))
        bs_mean[i, :] .= confint(
            bootstrap(
                mean,
                samples,
                BasicSampling(bs)
            ),
            BasicConfInt(ci)
        )[1]
    end

    bs_mean = moving_average(bs_mean, window)

    m = bs_mean[:, 1]
    lo = abs.(m - bs_mean[:, 2])
    hi = abs.(m - bs_mean[:, 2])

    return m, lo, hi
end

average_series(trial::Trial; bs=1000, ci=0.95, window=5) = average_series(trial.cummulative_rewards; bs, ci, window)

# function average_series(trial; bs=1000, ci=0.95, window=5)
#     bs_mean = Matrix{Float64}(undef, size(trial.cummulative_rewards, 1), 3)
#     for (i, samples) in Iterators.enumerate(eachrow(trial.cummulative_rewards))
#         bs_mean[i, :] .= confint(
#             bootstrap(
#                 mean,
#                 samples,
#                 BasicSampling(bs)
#             ),
#             BasicConfInt(ci)
#         )[1]
#     end

#     bs_mean = moving_average(bs_mean, window)

#     m = bs_mean[:, 1]
#     lo = abs.(m - bs_mean[:, 2])
#     hi = abs.(m - bs_mean[:, 2])

#     return m, lo, hi
# end

function curve_conf(trial; bs=1000, ci=0.95, window=5)
    m, lo, hi = average_series(trial; bs, ci, window)

    plot(m, ribbon=(lo, hi), legend=false)
end

function curves(trials...; window=5)
    bs = 1000
    ci = 0.95
    p = plot(ylims=(0, 500))

    for trial in trials
        bs_mean = Matrix{Float64}(undef, size(trial.cummulative_rewards, 1), 3)
        for (i, samples) in Iterators.enumerate(eachrow(trial.cummulative_rewards))
            bs_mean[i, :] .= confint(
                bootstrap(
                    mean,
                    samples,
                    BasicSampling(bs)
                ),
                BasicConfInt(ci)
            )[1]
        end

        bs_mean = moving_average(bs_mean, window)

        m = bs_mean[:, 1]
        lo = abs.(m - bs_mean[:, 2])
        hi = abs.(m - bs_mean[:, 2])

        plot!(m, ribbon=(lo, hi), label=trial.name)
    end

    return p
end

learning_curve(s::Session, ma::Int=1) = learning_curve(s.cummulative_rewards, ma)

learning_curve(t::Trial, ma::Int=1) = learning_curve(t.cummulative_rewards, ma)

end
