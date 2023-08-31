//! Find minimum-spanning-tree in an undirected graph using
//! [Kruskal's algorithm](https://en.wikipedia.org/wiki/Kruskal's_algorithm).

//! This file was adapted from crate pathfinding
//! 
//! 


#![allow(dead_code)]


use indexmap::IndexSet;
use std::hash::Hash;
use std::mem;

use num_traits::int::PrimInt;



// Our UnionFind is amix between those in petgraph and in pathfinding crates
/// union find structure <http://en.wikipedia.org/wiki/Disjoint-set_data_structure>
/// Ix must be an unsigned integer 
pub struct UnionFind<Ix> {
    parent: Vec<Ix>,
    rank: Vec<u32>,
}  // end of UnionFind<Ix>


impl <Ix> UnionFind<Ix>
    where Ix : PrimInt  {

    fn new(parent : Vec<Ix>) -> Self {
        let rank = vec![1; parent.len()];
        //
        UnionFind{parent, rank}
    } // end of new


    fn find(&mut self, mut node : Ix) -> Ix {
        while self.parent[Ix::to_usize(&node).unwrap()] != node {
            self.parent[Ix::to_usize(&node).unwrap()] = self.parent[Ix::to_usize(&self.parent[Ix::to_usize(&node).unwrap()]).unwrap()];
            node = self.parent[Ix::to_usize(&node).unwrap()];
        }
        node        
    }

    fn union(&mut self, mut a: usize, mut b: usize) {
        if self.rank[a] < self.rank[b] {
            mem::swap(&mut a, &mut b);
        }
        self.parent[b] = Ix::from(a).unwrap();
        if self.rank[a] == self.rank[b] {
            self.rank[a] += 1;
        }
    }  // end of union

    
    fn get_parent(&self) -> &Vec<Ix> {
        &self.parent
    }
}   // end of impl UnionFind

//=======================================================================================

// Find parent and compress path by path halving.
fn find(parents: &mut [usize], mut node: usize) -> usize {
    while parents[node] != node {
        parents[node] = parents[parents[node]];
        node = parents[node];
    }
    node
}

#[test]
fn test_path_halving() {
    let mut parents = vec![0, 0, 1, 2, 3, 4, 5, 6];
    assert_eq!(find(&mut parents, 7), 0);
    assert_eq!(parents, vec![0, 0, 1, 1, 3, 3, 5, 5]);
    assert_eq!(find(&mut parents, 7), 0);
    assert_eq!(parents, vec![0, 0, 1, 0, 3, 3, 5, 3]);
    assert_eq!(find(&mut parents, 7), 0);
    assert_eq!(parents, vec![0, 0, 1, 0, 3, 3, 5, 0]);
    assert_eq!(find(&mut parents, 6), 0);
    assert_eq!(parents, vec![0, 0, 1, 0, 3, 3, 3, 0]);
    assert_eq!(find(&mut parents, 6), 0);
    assert_eq!(parents, vec![0, 0, 1, 0, 3, 3, 0, 0]);
}

fn union(parents: &mut [usize], ranks: &mut [usize], mut a: usize, mut b: usize) {
    if ranks[a] < ranks[b] {
        mem::swap(&mut a, &mut b);
    }
    parents[b] = a;
    if ranks[a] == ranks[b] {
        ranks[a] += 1;
    }
}

/// Minimal-spanning-tree for nodes with integer indices. The nodes must have
/// consecutives indices between 0 and `number_of_nodes`-1.
///
/// # Panics
///
/// This function panics if a node is outside the range [0, `number_of_nodes`-1].
pub fn kruskal_indices<C>(
    number_of_nodes: usize,
    edges: &[(usize, usize, C)],
) -> impl Iterator<Item = (usize, usize, C)>
where
    C: Clone + PartialOrd,
{
    let mut parents = (0..number_of_nodes).collect::<Vec<_>>();
    let mut ranks = Vec::with_capacity(number_of_nodes);
    ranks.resize(number_of_nodes, 1);
    let mut edges = edges.to_vec();
    edges.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
    edges.into_iter().filter_map(move |(a, b, w)| {
        let ra = find(&mut parents, a);
        let rb = find(&mut parents, b);
        if ra != rb {
            union(&mut parents, &mut ranks, ra, rb);
            Some((a, b, w))
        } else {
            None
        }
    })
}

/// Find a minimum-spanning-tree. From a collection of
/// weighted edges, return a vector of edges forming
/// a minimum-spanning-tree.
pub fn kruskal<N, C>(edges: &[(N, N, C)]) -> impl Iterator<Item = (N, N, C)>
where
    N: Clone + Hash + Eq,
    C: Clone + PartialOrd,
{
    let mut nodes = IndexSet::new();
    let edges = edges
        .iter()
        .map(|&(ref a, ref b, ref w)| {
            let ia = nodes.insert_full(a.clone()).0;
            let ib = nodes.insert_full(b.clone()).0;
            (ia, ib, w.clone())
        })
        .collect::<Vec<_>>();
    kruskal_indices(nodes.len(), &edges).map(move |(ia, ib, w)| {
        (
            nodes.get_index(ia).unwrap().clone(),
            nodes.get_index(ib).unwrap().clone(),
            w,
        )
    })
}


//===============================================================================================================


#[cfg(test)]
mod tests {

use super::*;
    
    
    
    
fn log_init_test() {
    let _ = env_logger::builder().is_test(true).try_init();
}  
    
// taken from path-finding
#[test]
    fn test_union_find() {
        let parents = vec![0, 0, 1, 2, 3, 4, 5, 6];
        let mut unionf = UnionFind::<usize>::new(parents);

        assert_eq!(unionf.find(7), 0);
        assert_eq!(unionf.get_parent(), &vec![0, 0, 1, 1, 3, 3, 5, 5]);

        assert_eq!(unionf.find(7), 0);
        assert_eq!(unionf.get_parent(), &vec![0, 0, 1, 0, 3, 3, 5, 3]);

        assert_eq!(unionf.find(7), 0);
        assert_eq!(unionf.get_parent(), &vec![0, 0, 1, 0, 3, 3, 5, 0]);

        assert_eq!(unionf.find(6), 0);
        assert_eq!(unionf.get_parent(), &vec![0, 0, 1, 0, 3, 3, 3, 0]);

        assert_eq!(unionf.find(6), 0);
        assert_eq!(unionf.get_parent(), &vec![0, 0, 1, 0, 3, 3, 0, 0]);
    }  // end test_union_find

} // end of mod tests
