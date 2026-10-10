//! V11: bounded tactical proof, not a pattern-shape reward.
//! Only a reachable *full* T-spin double can earn a tactical bonus.
//! This uses the engine's 90-degree SRS move generator; 180-kick TSDs remain unsupported.
use super::{Board, Node, Spin, locks, board_value, v9_patterns};
use std::collections::{HashMap, HashSet};
use std::time::Instant;

#[derive(Default)]
pub(super) struct ProofCache {
    direct: HashMap<Board, bool>,
    with_setup: HashMap<(Board, char), bool>,
    pub(super) checked: usize,
    pub(super) confirmed: usize,
}

fn direct_tsd(board: &Board, cache: &mut ProofCache) -> bool {
    if let Some(&hit) = cache.direct.get(board) { return hit; }
    // Bounded path search from T spawn. A positive result is executable by
    // the same 90-degree movement model as the rest of the native engine.
    let hit = locks(board, 'T', 1800).iter()
        .any(|lock| lock.lines == 2 && lock.spin == Spin::Full);
    cache.checked += 1;
    if hit { cache.confirmed += 1; }
    cache.direct.insert(*board, hit);
    hit
}

fn one_setup_tsd(board: &Board, setup: char, cache: &mut ProofCache, deadline: Instant) -> bool {
    if let Some(&hit) = cache.with_setup.get(&(*board, setup)) { return hit; }
    // This is intentionally *selective*. Positive proofs are valid; a miss
    // only means the limited search did not establish a continuation.
    let mut preparations: Vec<(f64, Board)> = locks(board, setup, 900).into_iter()
        .map(|m| {
            let h = v9_patterns::analyze(&m.board);
            let score = board_value(&m.board) + 2.0 * h.tsd + h.kaidan + h.cave;
            (score, m.board)
        }).collect();
    preparations.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut checked_boards = HashSet::new();
    let mut result = false;
    for (_, candidate) in preparations.iter() {
        if Instant::now() >= deadline { break; }
        if !checked_boards.insert(*candidate) { continue; }
        if checked_boards.len() > 12 { break; }
        if direct_tsd(candidate, cache) { result = true; break; }
    }
    cache.with_setup.insert((*board, setup), result);
    result
}

fn opportunities(node: &Node, queue: &[char]) -> (bool, Option<char>) {
    if node.index >= queue.len() { return (false, None); }
    let natural = queue[node.index];
    let immediate = natural == 'T' || node.hold == 'T';
    // Only one piece before T (no guessed hidden sequence). After a normal
    // placement, the next preview is T. No hold-dependent conjecture here.
    let setup = if !immediate && queue.get(node.index + 1) == Some(&'T') {
        Some(natural)
    } else { None };
    (immediate, setup)
}

/// Add a *temporary* priority to candidate estimates, not accumulated attack.
/// Only applied to bounded shortlisted candidates to avoid stalling the UI.
/// It disappears after T is played; a real TSD still earns reward() normally.
pub(super) fn prioritize(nodes: &mut [Node], queue: &[char], cache: &mut ProofCache,
                         max_direct: usize, max_setup: usize, deadline: Instant,
                         hints_cache: &HashMap<Board,(f64,v9_patterns::Hints)>) {
    if nodes.is_empty() || cache.checked >= 75 || Instant::now() >= deadline { return; }
    if !nodes.iter().any(|n| { let (ready, setup) = opportunities(n,queue); ready || setup.is_some() }) { return; }
    let mut ranked: Vec<usize> = (0..nodes.len()).collect();
    ranked.sort_by(|&a, &b| nodes[b].estimate.total_cmp(&nodes[a].estimate));
    let mut picks: Vec<usize> = ranked.iter().take(max_direct.min(12)).copied().collect();

    // Include T-slot candidates missed by the height-first top list.
    let mut shapes: Vec<(f64, usize)> = ranked.iter().copied()
        .filter_map(|i| {
            let h = hints_cache.get(&nodes[i].board).map(|(_,h)| *h)
                .unwrap_or_else(|| v9_patterns::analyze(&nodes[i].board));
            let value = h.tsd * 2.0 + h.kaidan + h.cave + h.stsd;
            if value > 1.0 { Some((value, i)) } else { None }
        }).collect();
    shapes.sort_by(|a, b| b.0.total_cmp(&a.0));
    for (_, i) in shapes.into_iter().take(max_direct) {
        if !picks.contains(&i) { picks.push(i); }
    }
    picks.truncate(max_direct.saturating_mul(2));

    let mut checked_direct = 0;
    let mut checked_setup = 0;
    for i in picks {
        if Instant::now() >= deadline { break; }
        let (immediate, setup) = opportunities(&nodes[i], queue);
        if immediate && checked_direct < max_direct {
            checked_direct += 1;
            if direct_tsd(&nodes[i].board, cache) { nodes[i].estimate += 27.0; }
        } else if let Some(piece) = setup {
            if checked_setup >= max_setup { continue; }
            // One-setup searches are only attempted for credible silhouettes.
            if hints_cache.get(&nodes[i].board).map(|(_,h)| h.tsd).unwrap_or(0.0) < 2.0 { continue; }
            checked_setup += 1;
            if one_setup_tsd(&nodes[i].board, piece, cache, deadline) {
                nodes[i].estimate += 12.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::{H, FULL};
    #[test]
    fn no_spins_on_empty_board() {
        assert!(!direct_tsd(&[0; H], &mut ProofCache::default()));
    }
    #[test]
    fn confirms_reachable_full_tspin_double() {
        // Deterministic movement-search fixture with a reachable final-rotation
        // T lock at [(5,16),(6,16),(7,16),(6,17)] clearing two lines.
        let mut b = [0u16; H];
        b[14] = 0x203;
        b[15] = 0x22b;
        b[16] = 0x31f;
        b[17] = 0x3bf;
        b[18] = 0x2a7;
        b[19] = 0x3f3;
        assert!(direct_tsd(&b, &mut ProofCache::default()));
    }
    #[test]
    fn v9_false_positive_is_not_a_tsd_proof() {
        // V9 reports this pattern as a ready TSD silhouette, but a T from
        // spawn cannot complete a full T-spin double using our path rules.
        let mut b = [0u16; H];
        b[17] = FULL ^ (1 << 4);
        b[18] = FULL ^ ((1 << 3) | (1 << 4) | (1 << 5));
        b[19] = 1 << 3;
        assert!(v9_patterns::analyze(&b).tsd >= 8.0);
        assert!(!direct_tsd(&b, &mut ProofCache::default()));
    }
}
