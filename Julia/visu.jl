# Julia exploration of embeddings dumped in Csv files.

# Note that Makie can need entering Julia after : export LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6
#                to force julia to use system libstdc++.so.6 if it is more recent than the one embedded in Julia
#                see https://github.com/JuliaGL/GLFW.jl/issues/198
#                Otherwise julia does not find OpenGl access.

using CSV
using Colors, ColorSchemes
using DataFrames


using CairoMakie
using GLMakie


"""
    This function reads an embedding result as a csv file of name fname and dumps corresponding 
    scatter plot in fname.png
"""
function plotCsvLabels(fname)
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


"""
    This function reads reloads the "first distance" file. See Rust crate function Embedder::get_quality_estimate_from_edge_length
    It draws a heatmap of exp(distance-mindist) and dumps it in fname.png
"""
function plotCsvDist(fname) 
    data = DataFrame(CSV.File(fname, header=false))
    # get xmin, xmax, ymin, ymax
    xmin = minimum(data.Column2)
    xmax = maximum(data.Column2)

    ymin = minimum(data.Column3)
    ymax = maximum(data.Column3)
    #
    # we bin the images in size*size to avoid too many points
    #
    nbdelta = 2000
    nbpoints = nbdelta + 1
    dist = zeros(nbpoints, nbpoints)
    count = zeros(Int64, nbpoints, nbpoints)
    nbrow = size(data)[1]
    deltax = (xmax-xmin)
    deltay = (ymax-ymin)
    xs = range(xmin, xmax, step = deltax/nbdelta)
    ys = range(ymin, ymax, step = deltay/nbdelta)
    for i=1:nbrow
        x = (data[i, 2] - xmin)/ deltax
        y = (data[i, 3] - ymin)/ deltay
        ix = floor(Int64, 1 + nbdelta * x)
        iy = floor(Int64, 1 + nbdelta * y)
        dist[ix, iy] += data[i, 1]
        count[ix, iy] += 1
    end
    dist = dist ./ max.(count, 1)
    dmin = minimum(dist)
    # enhance constrast by dmin correction
    distremap = map(x -> if x > 0 exp(-(x-dmin)) else 0 end , dist)
    @info "file reloaded"
    #
    pngname = fname*".png"
    fig, ax, hm = CairoMakie.heatmap(xs, ys, distremap)
    Colorbar(fig[:, end+1], hm)
    CairoMakie.save(pngname, fig)
    @info "image dumped in : ", pngname
    #
    fig, ax, hm = GLMakie.heatmap(xs, ys, distremap)
    Colorbar(fig[:, end+1], hm)
    #
end