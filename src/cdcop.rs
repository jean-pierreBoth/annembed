//! This module implements Carré Du champ operator computations.
//! It builds upon DiffusionMaps tools.
//!
//! Bibliography
//! - *Diffusion Maps*. Coifman Lafon Appl. Comput. Harmon. Anal. 21 (2006) 5–30
//! - *Diffusion Geometry*. Ioo Jones 2024 https://arxiv.org/abs/2405.10858
//! - * Bamberger.J Jones.I Carre du Champ 2025. https://arxiv.org/abs/2510.05930

use crate::diffmaps::*;
use crate::graphlaplace::*;

/// The structure computes the transition kernel to neighbours using DiffusionMaps with adhoc parameters.
/// Then it computes the Covariance of the transition kernel at each asked for point.
/// This provides the best local normal approximation of the data and gives information on the geometry of the data
///   
pub struct CarreDuChamp {
    dparams: DiffusionParams,
}
