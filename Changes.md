- version 0.1.7
    -- added intrinsic dimension estimation by Facco's 2nn algorithm
    -- negative sampling can be done using hubness data (slightly better quality)
    -- update deps of lax, blas-src and ndarray


- version 0.1.6  
    -- The embedding computes a continuity ratio for each point 
    which can be examined with the Julia function plotCsvContinuity
    (see visu.jl/plotCsvContinuity)
    -- The Julia function plotCsvLabels returns a Dict mapping labels into colors making it possible to interpret proximities

- version 0.1.5
  
    -- switched to edition 2024  
    -- annembed binary changed name to embed (to avoid doc generation name clash).  
    -- python interface  
    -- added binary dmapembed to embed via diffusion maps