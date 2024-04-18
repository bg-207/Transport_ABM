using CSV
using DataFrames
# using StatsPlots
using Plots
using Plots.PlotMeasures
# using Makie
# using PlotlyJS
using Statistics
using OrderedCollections

df = CSV.read("C:/Users/godicb/OneDrive - The University of Melbourne/Documents/Julia/AV_Transport_ABM/sedentary_comparison_plot_18042024.csv", DataFrame)

pre_AVs = (df[!, "sedentary_no_avs"])
post_AVs = (df[!, "sedentary_post_avs"])
steps = (df[!, "time"])

display(Plots.plot(steps, [pre_AVs, post_AVs], label = ["Pre AV sedentary average" "Post AV sedentary average"], linewidth = 2))
