# Some data clustering and embedding tools

It implements:

1. Some variations about the **Umap** embedding tool (L. Mac Innes) using Renyi entropy and hierarchical representation of neighbourhood data as provided by the Hnsw algorithm.

2. An implementation of the Mapper algorithm using the **Ripser** module from U. Bauer

3. As a by product the crate provides an implementation of some range approximation and approximated SVD for denses and row compressed matrices as described in Halko-Tropp

## Results

### Humap

### Mapper

### Randomized SVD

## References

- Visualizing Large Scale High Dimensional Data
  Tang Liu WWW2016 [LargeVis](https://arxiv.org/pdf/1602.00370.pdf)

- Visualizing data using t_sne.
  Van der Maaten 

- Umap: Uniform Manifold Approximation and Projection for Dimension Reduction.
  L.MacInnes,J. Healy and J.Melville 2018
  
## License

Licensed under either of

- Apache License, Version 2.0, [LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>
  
- MIT license [LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>

at your option.

This software was written on my own while working at [CEA](http://www.cea.fr/)
