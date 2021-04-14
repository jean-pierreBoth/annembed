//! single linkage hdbscan
//! 
//! 
//! 
//! implements single linkage clustering on top of Kruskal algorithm.
//! 
//! 

// 1.  We get from the hnsw a list of edges for kruskal algorithm
// 2.  Run kruskal algorithm ,  we get nodes of edge, weigth of edge and parent of nodes 
//        - so we get at each step the id of cluster representative that unionized.
//       
// 3. Fill a dendogram structure

/// This structure represent a merge step
/// 
pub struct UnionStep <NodeIdx : PrimInt, F : Float> {
    /// node a of edge removed
    nodea : NodeIdx,
    /// node b of edge removed
    nodeb : NodeIdx,
    /// weight of edge
    weight : F,
    /// step at which the merge occurs
    step : usize,
    /// representative of nodea in the union-find
    clusta : NodeIdx,
    /// representative of nodeb in the union-find
    clustb : NodeIdx
}


pub struct DendoGram<NodeIdx : PrimInt, F : Float> {
    steps : Vec<UnionStep<NodeIdx, F> >
}
