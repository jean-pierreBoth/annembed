# Julia exploration of embeddings dumped in Csv files.

# Note that Makie can need entering Julia after : export LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6
#                to force julia to use system libstdc++.so.6 if it is more recent than the one embedded in Julia
#                see https://github.com/JuliaGL/GLFW.jl/issues/198
#                Otherwise julia does not find OpenGl access.

using CSV
using Colors, ColorSchemes
using DataFrames
using ImageCore


using CairoMakie
using GLMakie
using Statistics

"""
    This function reads an embedding result as a csv file of name fname and dumps corresponding
    scatter plot in fname.png.
    Default columns are the 2 first after labels

    The function returns as result a Dict associating data lables to colors.
    
    The colors used can be seen vizualized in IJulia or in VsCode  :
        # color extraction
        colors = collect(values(result))
        # color display
        colors
    
        Labels in the same order as colors are exracted by
        # 
        labels = collect(keys(result))
"""
function plotCsvLabels(fname, col1=2, col2=3, clip=false, cvd=false)
    data = DataFrame(CSV.File(fname, header=false))
    # We reindex labels to the set [1, nb_labels] for color mapping
    labeldict = Dict{Int64,Int64}()
    nb_labels = 0
    cidx = 1
    colorsIdx = zeros(Int64, length(data.Column1))
    for l in data.Column1
        if !haskey(labeldict, l)
            nb_labels += 1
            labeldict[l] = nb_labels
            colorsIdx[cidx] = nb_labels
        else
            val = get(labeldict, l, 0)
            @assert val > 0
            colorsIdx[cidx] = val
        end
        cidx += 1
    end
    # get xmin, xmax, ymin, ymax
    xmin = minimum(data[!, col1])
    xmax = maximum(data[!, col1])
    sigmax = sqrt(var(data[!, col1]))
    clipxplus = min(xmax / sigmax, 10.0) * sigmax
    clipxminus = max(xmin / sigmax, -10.0) * sigmax

    ymin = minimum(data[!, col2])
    ymax = maximum(data[!, col2])
    sigmay = sqrt(var(data[!, col2]))
    clipyplus = min(ymax / sigmay, 10.0) * sigmay
    clipyminus = max(ymin / sigmay, -10.0) * sigmay

    # colors for cvd people!
    mycolors = ColorScheme(distinguishable_colors(nb_labels, transform=protanopic))
    labelscolor = map(l -> mycolors[labeldict[l]], data.Column1)
    # avec makie. adapt marker size to number of values
    nbdata = length(data.Column1)
    if nbdata > 1000000
        mvalue = 1
    else
        mvalue = 2
    end
    pngname = fname * ".png"
    if clip
        fig = CairoMakie.scatter(clamp.(data[!, col1], clipxminus, clipxplus), clamp.(data[!, col2], clipyminus, clipyplus), color=labelscolor, markersize=mvalue)
    else
        fig = CairoMakie.scatter(data[!, col1], data[!, col2], color=labelscolor, markersize=mvalue)
    end
    CairoMakie.save(pngname, fig)
    # return Dictionary associating original label to colors
    originalLabels = collect(keys(labeldict))
    originalLabelColor = map(l -> mycolors[labeldict[l]], originalLabels)
    originLabelColorDict = Dict{Int64,RGB{FixedPointNumbers.N0f8}}()
    for l in keys(labeldict)
        originLabelColorDict[l] = mycolors[labeldict[l]]
    end
    return originLabelColorDict
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
    nbdelta = 1000
    nbpoints = nbdelta + 1
    dist = zeros(nbpoints, nbpoints)
    count = zeros(Int64, nbpoints, nbpoints)
    nbrow = size(data)[1]
    deltax = (xmax - xmin)
    deltay = (ymax - ymin)
    xs = range(xmin, xmax, step=deltax / nbdelta)
    ys = range(ymin, ymax, step=deltay / nbdelta)
    for i = 1:nbrow
        x = (data[i, 2] - xmin) / deltax
        y = (data[i, 3] - ymin) / deltay
        ix = floor(Int64, 1 + nbdelta * x)
        iy = floor(Int64, 1 + nbdelta * y)
        dist[ix, iy] += data[i, 1]
        count[ix, iy] += 1
    end
    dist = dist ./ max.(count, 1)
    dmin = minimum(dist)
    # enhance constrast by dmin correction
    distremap = map(x -> if x > 0
            exp(-(x - dmin))
        else
            0
        end, dist)
    @info "file reloaded"
    #
    pngname = fname * ".png"
    fig, ax, hm = CairoMakie.heatmap(xs, ys, distremap)
    Colorbar(fig[:, end+1], hm)
    CairoMakie.save(pngname, fig)
    @info "image dumped in : ", pngname
    #
    fig, ax, hm = GLMakie.heatmap(xs, ys, distremap)
    Colorbar(fig[:, end+1], hm)
    #
end

"""
    This function reads reloads the "continuity_ratio.csv" file. See Rust crate function Embedder::get_quality_estimate_from_edge_length
    It draws a heatmap of min(log(max(1., continuity_ratio)),2.) to filter out extreme values 
    Pixels not corresponding to a data point (giving background) are set -1.
    The name of the png file (in current working directory) is fname.png
 
"""
function plotCsvContinuity(fname)
    data = DataFrame(CSV.File(fname, header=false))
    # get xmin, xmax, ymin, ymax
    xmin = minimum(data.Column2)
    xmax = maximum(data.Column2)

    ymin = minimum(data.Column3)
    ymax = maximum(data.Column3)
    #
    m = median(data[!, 1])
    s = sqrt(var(data[!, 1]))
    #
    # we bin the images in size*size to avoid too many points
    #
    nbdelta = 1000
    nbpoints = nbdelta + 1
    continuity = -ones(nbpoints, nbpoints)
    nbrow = size(data)[1]
    deltax = (xmax - xmin)
    deltay = (ymax - ymin)
    xs = range(xmin, xmax, step=deltax / nbdelta)
    ys = range(ymin, ymax, step=deltay / nbdelta)
    for i = 1:nbrow
        x = (data[i, 2] - xmin) / deltax
        y = (data[i, 3] - ymin) / deltay
        ix = floor(Int64, 1 + nbdelta * x)
        iy = floor(Int64, 1 + nbdelta * y)
        continuity[ix, iy] = data[i, 1]
    end
    #   display points with continuity ratio greater 1
    #   and with logarirthmic intensity capped by exp(2) = 7.4
    f1(x::Float64) =
        if x > 0.
            return min(log(max(1., x)), 2.)
        else
            return -2.
        end
    #
    graph_continuity = map(f1, continuity)
    #
    @info "file reloaded"
    #
    pngname = fname * ".png"
    fig, ax, hm = CairoMakie.heatmap(xs, ys, graph_continuity)
    Colorbar(fig[:, end+1], hm)
    CairoMakie.save(pngname, fig)
    @info "image dumped in : ", pngname
    #
    fig, ax, hm = GLMakie.heatmap(xs, ys, graph_continuity)
    Colorbar(fig[:, end+1], hm)
    #
end