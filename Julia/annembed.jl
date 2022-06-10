# Julia utilities to complement this Rust crate

module AnnEmbed


using CSV
using Colors, ColorSchemes
using DataFrames


using CairoMakie

using SparseArrays
using Ripserer

debug_log = stdout
logger = ConsoleLogger(stdout, CoreLogging.Info)
global_logger(logger)

include("visu.jl")
include("toripserer.jl")

end