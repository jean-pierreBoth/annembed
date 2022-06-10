# Julia exploration of Csv dumps

using CSV
using Colors, ColorSchemes
using DataFrames


using CairoMakie
using GLMakie


"""
    This function reads an embedding result as a csv file of name fname and dumps corresponding 
    scatter plot in fname.png
"""
function plotcsv(fname)
    data = DataFrame(CSV.File(fname, header=false))

    # get xmin, xmax, ymin, ymax
    xmin = minimum(data.Column2)
    xmax = maximum(data.Column2)

    ymin = minimum(data.Column3)
    ymax = maximum(data.Column3)

    # colors for cvd people!
    mycolors = ColorScheme(distinguishable_colors(10, transform=protanopic))

    # avec makie
    pngname = fname*".png"
    fig = CairoMakie.scatter(data.Column2[1:end], data.Column3[1:end], color = mycolors[data.Column1[1:end] .+ 1], markersize=1)
    CairoMakie.save(pngname, fig)
end

