using CSV
using DataFrames
# using StatsPlots
using Plots
using Plots.PlotMeasures
# using Makie
# using PlotlyJS
using Statistics
using OrderedCollections
using HypothesisTests

df = CSV.read("C:/Users/godicb/OneDrive - The University of Melbourne/Documents/Julia/AV_Transport_ABM/sedentary_comparison_plot_19042024.csv", DataFrame)

pre_AVs = (df[!, "sedentary_no_AVs"])
post_AVs = (df[!, "sedentary_post_AVs"])
steps = (df[!, "time"])

fig_1 = Plots.plot(steps, [pre_AVs, post_AVs], label = ["Pre AV sedentary average" "Post AV sedentary average"], linewidth = 2)

Plots.savefig(fig_1, "C:/Users/godicb/OneDrive - The University of Melbourne/Documents/Julia/AV_Transport_ABM/sedentary_graph_1_19042024")

