# Some Julia utilities to link exports to Julia Ripserer package
# Ripserer seems oriented towards Plots...

using Logging
using Base.CoreLogging
using Printf

using SparseArrays
using Ripserer
using PersistenceDiagrams
using Plots
using BSON

logger = ConsoleLogger(stdout, Base.CoreLogging.Debug)
global_logger(logger)


"""
This function reads a matrix in the triplet form i j val and returns
a CSC SparseArrays.
Correspond to data coming from a GraphProjection
"""
function cscmatLoad(fname)
    io = open(fname)
    I = Vector{Int64}()
    J = Vector{Int64}()
    V = Vector{Float64}()
    numline = 0
    for line in eachline(io)
        numline += 1
        sp = split(line, " ")
        if length(sp) != 3 
            @debug line, numline
        end
        # check we have 3 str
        # we add 1 to indexes to got 1 based indexation (we come from Rust)
        push!(I, 1+parse(Int64, sp[1]))
        push!(J, 1+parse(Int64, sp[2]))
        push!(V, parse(Float64, sp[3]))
    end
    cscmat = sparse(I,J,V)
    close(io)
    return cscmat
end


"""
    Reload a lower inferior matrix distance matrix from rust, in a Bson format
    Corresponds to local graph data
"""
function lowimatLoadBson(fname) 
    bsonv = BSON.load(fname)
    v = bsonv[:limat]
    v = Vector{Float64}(v)
    # get back to matrix form
    size = sqrt(2 * length(v))
    size = Int64(floor(size))
    @debug "matrix size" size
    # now we get back to a Matrix for Ripserer 
    distmat = zeros(Float64, size,size)
    rank = 1
    for i=1:size
        for j=1:i
            distmat[i,j] = v[rank]
            distmat[j,i] = v[rank]
            rank += 1
        end
    end
    return distmat
end



"""
    This function reloads the Bson matrix of distances dumped from Rust annembed::fromhnsw::kgproj
    It produces png files of persistence diagrams and barcodes corresponding to point cloud analyzed.
    dim_max is the maximum dimension for which homology is computed. 
"""
function localPersistency(fname; dim_max = 1)
    # implicit use of GR!
    mat = lowimatLoadBson(fname)
    pers = ripserer(mat, dim_max = dim_max)
    persplot = Plots.plot(pers, markersize = 2)
    Plots.png(persplot, fname*"-pers")
    barplot = barcode(pers)
    Plots.png(barplot, fname*"-bar")
end



"""
    This function reloads the a csc matrix of distances dumped from Rust annembed::fromhnsw::kgproj
    It produces png files of persistence diagrams and barcodes corresponding to point cloud analyzed.
    dim_max is the maximum dimension for which homology is computed. 
"""
function projectedPersistency(fname; dim_max = 1)
    mat = cscmatLoad(fname)
    pers = ripserer(mat, dim_max = dim_max)
    persplot = Plots.plot(pers, markersize = 2)
    Plots.png(persplot, fname*"-pers")
    barplot = barcode(pers)
    Plots.png(barplot, fname*"-bar")
end

