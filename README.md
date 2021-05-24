# Some data clustering and embedding tools

The crate provides:

1. Some variations on data embedding tools from t-Sne (2008) to Umap 2018.
   Our implementation is in fact a mix of the various embedding algorithms
    published recently and mentioned in References.

   - The graph is initialized by the Hnsw nearest neighbour algorithm. This provides for free sub-sampling in the data to embed by considering less densely occupied layers. It it is also possible to embed only the neighbourhood of a point by extracting recursively neighbours of a the initial point down to the most dense layers.
  
   - The preliminary graph used in the embedding is initialized with an exponential kernel (as in Umap) but taking into account a local density of points. The symetrisation is done as in t-sne or LargeVis. We use the diffusion maps algorithm (Lafon-Keller-Coifman).

   - We also use a cross entropy optimization of this initial layout but try to take into account the initial local density estimate.

2. An implementation of the Mapper algorithm using the C++ **Ripser** module from U. Bauer

3. Some by-products :
    - an implementation of range approximation and approximated SVD for denses and row compressed matrices as described in Halko-Tropp (Cf. [Tsvd](https://arxiv.org/abs/0909.4061)).

    - a Diffusion Maps implementation (Cf [Dmap](https://www.pnas.org/content/102/21/7426))

    - A single-linkage hierarchical clustering function

## Results

### Humap

### Mapper

### Randomized SVD

## Docs

To build the doc with latex use :
cargo rustdoc -- --html-in-header katex.html

## References

- Visualizing data using t_sne.
  Van der Maaten and Hinton 2008.

- Visualizing Large Scale High Dimensional Data
  Tang Liu WWW2016 2016 [LargeVis](https://arxiv.org/pdf/1602.00370.pdf)
  
- Phate Visualizing Structure and Transitions for Biological Data Exploration
  K.R Moon 2017.

- Umap: Uniform Manifold Approximation and Projection for Dimension Reduction.
  L.MacInnes, J.Healy and J.Melville 2018

## License

Licensed under either of

- Apache License, Version 2.0, [LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>
  
- MIT license [LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>

at your option.

This software was written on my own while working at [CEA](http://www.cea.fr/)
