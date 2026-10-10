//! V9: structural search hints. These are NOT assertions that a named setup is
//! executable. Only locks()/spin_type() can award an actual T-spin clear.
use super::{Board, H, W, Pose, cells, legal, occupied};
#[cfg(test)]
use super::FULL;

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Hints {
    // Two-line T silhouette with favorable pivot corners (V8 carried forward).
    pub tsd: f64,
    // Approximate construction opportunities, not canonical pattern proofs.
    pub kaidan: f64,
    pub cave: f64,
    pub stsd: f64,
}

fn clean_column(b: &Board, x: usize, height: usize) -> bool {
    // Disallow hidden holes under the surface in the relevant columns.
    (H - height..H).all(|y| b[y] & (1u16 << x) != 0)
}
fn heights(b: &Board) -> [usize; W] {
    let mut result = [0; W];
    for x in 0..W {
        if let Some(y) = (0..H).find(|&y| b[y] & (1u16 << x) != 0) {
            result[x] = H-y;
        }
    }
    result
}

/// Recognizes *structural hints*, not guaranteed T-spin paths.
/// Does not create/reward generic buried holes. Ranges are all bounded.
pub(super) fn analyze(board: &Board) -> Hints {
    let mut out = Hints::default();
    let hs = heights(board);
    let maxh = *hs.iter().max().unwrap_or(&0);
    // Stage-1 pattern hints are intentionally weak and only for clean, low-risk stacks.
    if maxh >= 3 && maxh <= 15 {
        // Kaidan-like: staircase directly next to a narrow open column.
        // Require three consecutive columns, two stepped supporting heights.
        for x in 0..W-2 {
            for (well,first,second) in [(x,x+1,x+2),(x+2,x+1,x)] {
                if hs[first] >= 3 && hs[first] >= hs[well]+2 &&
                   hs[second] >= 2 && hs[first].abs_diff(hs[second]) == 1 &&
                   clean_column(board, well, hs[well]) &&
                   clean_column(board, first, hs[first]) &&
                   clean_column(board, second, hs[second]) {
                    out.kaidan = out.kaidan.max(2.9);
                }
            }
        }
        // STMB-like: approximately level shoulders around a clean 3-wide dip.
        // This is the 3-wide *opportunity*, not the final S/Z floating overhang.
        for x in 1..W-3 {
            let l=hs[x-1]; let r=hs[x+3];
            let mid=&hs[x..x+3];
            let low=*mid.iter().min().unwrap();
            let high=*mid.iter().max().unwrap();
            if l>=4 && r>=4 && l.abs_diff(r)<=1 && high.abs_diff(low)<=1 &&
               l>=high+2 && r>=high+2 && high<=13 &&
               (x-1..=x+3).all(|i| clean_column(board,i,hs[i])) {
                out.cave = out.cave.max(3.4);
            }
        }
    }

    // V8-style supported T-slot examination, strengthened with row completion.
    // Fast reject: no potentially fillable line -> no ready spin slot.
    if !board.iter().any(|r| r.count_ones() >= 6) { return out; }
    let mut slots: Vec<(i8,i8)> = Vec::new();
    for y in 0i8..H as i8 {
        for x in -1i8..W as i8 {
            for r in 0u8..4 {
                let p=Pose{x,y,r};
                if !legal(board,'T',p) || legal(board,'T',Pose{y:y+1,..p}) { continue; }
                let blocks=cells('T',p);
                let cx=x+1; let cy=y+1;
                let corners=[
                    occupied(board,cx-1,cy-1),occupied(board,cx+1,cy-1),
                    occupied(board,cx-1,cy+1),occupied(board,cx+1,cy+1),
                ];
                let corner_count=corners.iter().filter(|&&v|v).count();
                if corner_count<2 { continue; }
                let front=match r { 0 => (0,1), 1 => (1,3), 2 => (2,3), _ => (0,2) };
                let front_count=(corners[front.0] as usize)+(corners[front.1] as usize);
                let mut rows_completed=0;
                let mut rows_one_away=0;
                for yy in y..=(y+2).min((H-1) as i8) {
                    let additions=blocks.iter()
                        .filter(|&&(_,by)|by==yy)
                        .fold(0u16,|bits,&(bx,_)|bits | (1u16 << (bx as u32)));
                    if additions==0 { continue; }
                    let cnt=(board[yy as usize] | additions).count_ones() as usize;
                    if cnt==W {rows_completed+=1;} else if cnt+1==W {rows_one_away+=1;}
                }
                if rows_completed>=2 && corner_count>=3 && front_count==2 {
                    out.tsd=out.tsd.max(8.0);
                    // Deduplicate rotations with same pivot.
                    if !slots.contains(&(cx,cy)) { slots.push((cx,cy)); }
                } else if rows_completed>=2 && corner_count>=3 {
                    out.tsd=out.tsd.max(4.0);
                } else if rows_completed>=1 && rows_one_away>=1 && corner_count>=2 {
                    out.tsd=out.tsd.max(2.5);
                } else if rows_completed>=1 && corner_count>=3 && front_count==2 {
                    out.tsd=out.tsd.max(3.0);
                }
            }
        }
    }
    // STSD-like hint only for multiple, distinct, vertically stacked TSD
    // silhouettes in approximately the same lane. May be inaccessible.
    for (i,&(ax,ay)) in slots.iter().enumerate() {
        for &(bx,by) in slots.iter().skip(i+1) {
            if (ax-bx).abs()<=1 && (ay-by).abs()>=3 {
                out.stsd = 4.0;
            }
        }
    }
    out
}

/// Queue-aware. No T in the known preview -> almost no speculative reward;
/// two T pieces are needed before giving an STSD bonus.
pub(super) fn bonus(h: Hints, next: &[char], held: char, index: usize) -> f64 {
    let preview=&next[index.min(next.len())..];
    let mut t_positions=preview.iter().enumerate().filter(|(_,p)|**p=='T')
        .map(|(idx,_)|idx).collect::<Vec<_>>();
    if held=='T' { t_positions.push(0); }
    if t_positions.is_empty() { return 0.0; }
    let nearest=*t_positions.iter().min().unwrap() as f64;
    let urgency=1.0/(1.0+0.24*nearest);
    let has_sz=held=='S' || held=='Z' || preview.iter().take(4).any(|p|*p=='S'||*p=='Z');
    let mut score=h.tsd + h.kaidan*if has_sz {1.0}else{0.50}
        + h.cave*if has_sz {1.0}else{0.45};
    if t_positions.len()>=2 { score+=h.stsd; }
    // Keep this a modest temporary incentive compared with actual attack rewards.
    (score*urgency).min(12.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn empty_board_has_no_patterns() {
        let hints=analyze(&[0;H]);
        assert_eq!(hints.tsd,0.0);
        assert_eq!(hints.kaidan,0.0);
        assert_eq!(hints.cave,0.0);
        assert_eq!(hints.stsd,0.0);
    }
    #[test]
    fn finds_structural_two_line_t_slot() {
        let mut b=[0;H];
        b[17]=FULL ^ (1<<4);
        b[18]=FULL ^ ((1<<3)|(1<<4)|(1<<5));
        b[19]=1 << 3;
        assert!(analyze(&b).tsd>=8.0);
    }
    #[test]
    fn kaidan_like_step_recognized_but_not_a_buried_hole() {
        let mut b=[0;H];
        for y in 16..H {b[y]|=1<<5;}
        for y in 15..H {b[y]|=1<<6;}
        for y in 19..H {b[y]|=1<<4;}
        assert!(analyze(&b).kaidan>0.0);
        b[18]&=!(1<<6);
        assert_eq!(analyze(&b).kaidan,0.0);
    }
    #[test]
    fn stmb_like_three_wide_gap_detected() {
        let mut b=[0;H];
        for y in 14..H { b[y] |= (1<<2)|(1<<6); }
        for y in 17..H { b[y] |= (1<<3)|(1<<4)|(1<<5); }
        assert!(analyze(&b).cave>0.0);
        b[18] &= !(1<<4);
        assert_eq!(analyze(&b).cave,0.0);
    }
    #[test]
    fn pattern_bonus_requires_t_in_queue_or_hold() {
        // Stay below the global score cap so the two-T STSD increment remains
        // observable instead of saturating both sides of the comparison.
        let h=Hints{tsd:4.0,kaidan:1.0,cave:1.0,stsd:4.0};
        assert_eq!(bonus(h,&['I','O','S'], '-',0),0.0);
        assert!(bonus(h,&['T','S'], '-',0)>bonus(h,&['I','O','T','S'],'-',0));
        assert!(bonus(h,&['I','S'], 'T',0)>0.0);
        assert!(bonus(h,&['T','S','T'], '-',0)>bonus(h,&['T','S','I'], '-',0));
    }
}
