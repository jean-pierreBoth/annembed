//! This module defines parameters for ann embedding.
//!
#[cfg_attr(doc, katexit::katexit)]
/// It is necessary to describe briefly the model used in the embedding:
/// 
/// ## Definition of the weight of an edge of the graph to embed
///
/// First we define the local scale $\rho$ around a point.  
/// It is defined as the mean of distances of points to their nearest neighbour.
/// The points taken into account to define $\rho$ are the node we consider and
/// all its knbn neighbours. So we compute the mean of distances to nearest neighbours 
/// on knbn + 1 points around current point.
///   
/// let ($d_{i}$) be the sorted distances in increasing order of neighbours  for i=0..k of a node n,
///     $$w_{i} = \exp\left(- \left(\frac{d_{i} - d_{0}}{S * \rho}\right)^{\beta} \right)$$
/// 
/// S is a scale factor modulating $\rho$. 
/// After that weights are normalized to a probability distribution.
///
/// So before normalization $w_{0}$ is always equal to 1. Augmenting β to 2. makes the weight $w_{i}$ decrease faster. 
/// *The least weight of an edge must not go under $10^{-5}$ to limit the range of weight and avoid Svd numeric difficulties*. 
/// The code stops with an error in this case.
/// So after normalization the range of weights from $w_{0}$ to $w_{k}$ is larger. 
/// Reducing S as similar effect but playing with both $\beta$ and the scale adjustment must not violate the range constraint on weights.
/// 
/// It must be noted that setting the scale as described before and renormalizing to get a probability distribution
/// gives a perplexity nearly equal to the number of neighbours.  
/// This can be verified by using the logging (implemented using the crates **env_logger** and **log**) and setting 
/// RUST_LOG=annembed=INFO in your environment. 
/// Then quantile summaries are given for the distributions of edge distances, edge weights, and perplexity
/// of nodes. This helps adjusting parameters β, Scale and show their impact on these quantiles.
/// 
///  
/// Default value :  
/// 
///  $\beta = 1$ so that we have exponential weights similar to Umap.  
/// 
///  $S = 0.5$
/// 
/// But it is possible to set β to 2. to get more gaussian weight or reduce to 0.5 and adjust S to respect the constraints on edge weights.
///    
/// ## Definition of the weight of an edge of the embedded graph
/// 
/// The embedded edge has the usual expression :
/// $$ w(x,y) = \frac{1}{1+ || \left((x - y)/a_{x} \right)||^{2*b}  } $$
/// 
/// by default b = 1.
/// The coefficient $a_{x}$ is deduced from the scale coefficient in the original space with some
/// restriction to avoid too large fluctuations.
/// 
/// 
/// 
/// - Initial step of the gradient and number of batches 
/// 
/// A number of batch for the Mnist digits data around 10-20 seems sufficient. 
/// The initial gradient step $\gamma_{0}$ can be chosen around 1. (in the range 1/5 ... 5.).    
/// Reasonably it should satisfy nb_batch $ * \gamma_{0} < 1 $
/// 
/// - asked_dimension : default is set to 2.
/// 
/// ## The optimization of the embedding
/// 
/// The embedding is optimized by minimizing the (Shannon at present time) cross entropy 
/// between distribution  of original and embedded weight of edges. This minimization is done
/// by a standard (multithreaded) stochastic gradient with negative sampling for the unobserved edges
/// (see [Mnih-Teh](https://arxiv.org/abs/1206.6426) or 
/// [Mikolov](https://proceedings.neurips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf))
/// 
/// The number of negative edge sampling is set to a fixed value 5.
/// 
/// - expression of the gradient
/// 


/// main parameters driving Embeding
#[derive(Clone, Copy)]
pub struct EmbedderParams {
    /// embedding dimension : default to 2
    pub asked_dim : usize,
    /// defines if embedder is initialized by a diffusion map step. default to true
    pub dmap_init : bool,
    /// exponent used in defining edge weight in original graph. 0.5 or 1.
    pub beta : f64,
    /// exponenent used in embedded space, default 1.
    pub b : f64,
    /// embedded scale factor. default to 1.
    pub scale_rho : f64,
    /// initial gradient step , default to 2.
    pub grad_step : f64,
    /// nb sampling by edge in gradient step. default = 10
    pub nb_sampling_by_edge : usize,
    /// number of gradient batch. default to 15
    pub nb_grad_batch : usize,
    /// the number of gradient batch in hierarchical case is nb_grad_batch multiplied by grad_factor.
    /// As the first iterations run on few points we can do more iterations. Default is 4.
    pub grad_factor : usize, 
    /// if layer > 0 means we have hierarchical initialization
    pub hierarchy_layer : usize
} // end of EmbedderParams


impl EmbedderParams {
    pub fn default()  -> Self {
        let asked_dim = 2;
        let dmap_init = true;
        let beta = 1.;
        let b = 1.;
        let grad_step = 2.;
        let nb_sampling_by_edge = 10;
        let scale_rho = 1.;
        let nb_grad_batch = 15;
        let grad_factor : usize = 4;
        let hierarchy_layer = 0;
        EmbedderParams{asked_dim, dmap_init, beta, b, scale_rho, grad_step, nb_sampling_by_edge , nb_grad_batch, grad_factor, hierarchy_layer}
    }


    pub fn log(&self) {
        log::info!("EmbedderParams");
        log::info!("\t asked dim : {}", self.asked_dim);
        log::info!("\t gradient step : {}", self.grad_step);
        log::info!("\t edge exponent in original graph : {} ", self.beta);
        log::info!("\t nb sampling by edge : {}", self.nb_sampling_by_edge);
        log::info!("\t beta : {}", self.beta);
        log::info!("\t scale factor : {}", self.scale_rho);
        log::info!("\t number of gradient batch : {}", self.nb_grad_batch);
        log::info!("\t factor for nbgradient batch in first hierarchical pass is  : {}", self.grad_factor);
        log::info!("\t hierarchy layer  : {}", self.hierarchy_layer);
    }

    /// set to false if random initialization is preferred
    pub fn set_dmap_init(&mut self, val:bool) {
        self.dmap_init = val;
    }

    /// set the number of gradient batch. At each batch each edge is sampled nb_sampling_by_edge times.
    /// default to 20
    pub fn set_nb_gradient_batch(&mut self, nb_batch : usize) {
        self.nb_grad_batch = nb_batch;
    }

    /// sets the dimension for data embedding. Default to 2
    pub fn set_dim(&mut self, dim : usize) {
        self.asked_dim = dim;
    }

    /// sets the number of time each edge should be sampled in a gradient batch. Default to 10
    pub fn set_nb_edge_sampling(&mut self, nb_sample_by_edge: usize) {
        self.nb_sampling_by_edge = nb_sample_by_edge;
    }

    /// get asked embedding dimension
    pub fn get_dimension(&self) -> usize {
        self.asked_dim
    }

    pub fn set_hierarchy_layer(&mut self, layer : usize) {
        self.hierarchy_layer = layer;
    }

    pub fn get_hierarchy_layer(&self) -> usize {
        self.hierarchy_layer
    }    
} // end of impl EmbedderParams
