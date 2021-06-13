//! defines parameters for ann embedding
//! 
//! 
//! 
/// main parameters driving Embeding
#[derive(Clone, Copy)]
pub struct EmbedderParams {
    /// embedding dimension : default to 2
    pub asked_dim : usize,
    /// defines if embedder is initialized by a diffusion map step. default to true
    pub dmap_init : bool,
    ///
    pub b : f64,
    /// scale factor. default to 1.
    pub scale_rho : f64,
    /// initial gradient step , default to 1.
    pub grad_step : f64,
    /// nb sampling by edge in gradient step. default = 5
    pub nb_sampling_by_edge : usize,
    /// number of gradient batch. default to 20
    pub nb_grad_batch : usize,
} // end of EmbedderParams


impl EmbedderParams {
    pub fn new()  -> Self {
        let asked_dim = 2;
        let dmap_init = true;
        let b = 1.;
        let grad_step = 0.05;
        let nb_sampling_by_edge = 5;
        let scale_rho = 1.;
        let nb_grad_batch = 20;
        EmbedderParams{asked_dim, dmap_init, b, scale_rho, grad_step, nb_sampling_by_edge , nb_grad_batch}
    }

    pub fn log(&self) {
        log::info!("EmbedderParams");
        log::info!("\t gradient step : {}", self.grad_step);
        log::info!("\t nb sampling by edge : {}", self.nb_sampling_by_edge);
        log::info!("\t scale factor : {}", self.scale_rho);
        log::info!("\t number of gradient batch : {}", self.nb_grad_batch);
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

    /// sets the number of time each edge should be sampled in a gradient batch. Default to 20
    pub fn set_nb_edge_sampling(&mut self, nb_sample_by_edge: usize) {
        self.nb_sampling_by_edge = nb_sample_by_edge;
    }

    ///
    pub fn get_dimension(&self) -> usize {
        self.asked_dim
    }
} // end of impl EmbedderParams
