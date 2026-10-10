//! V7 offline strategy engine. No screen capture, network, or game input.
//! One request per line. See protocol described in README_V7.md.
use std::collections::{HashMap, HashSet, VecDeque};
use std::io::{self, BufRead, Write};
use std::time::{Duration, Instant};

const H: usize = 20;
const W: usize = 10;
const FULL: u16 = (1 << W) - 1;
type Board = [u16; H];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Spin { None, Mini, Full }
impl Spin {
    fn parse(s: &str) -> Result<Self, String> {
        match s { "n" => Ok(Self::None), "m" => Ok(Self::Mini), "f" => Ok(Self::Full),
            _ => Err(format!("invalid spin {s}")) }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Pose { x: i8, y: i8, r: u8 }
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct State { pose: Pose, last_rotation: bool }
#[derive(Clone, Copy, Debug)]
struct Lock { board: Board, lines: u8, spin: Spin, cells: [(i8,i8);4] }

#[derive(Clone)]
struct Node {
    board: Board,
    hold: char,
    index: usize,
    b2b: u16,
    combo: i16,
    reward: f64,
    estimate: f64,
    root: usize,
    second_name: char,
    second_cells: [(i8,i8);4],
    third_name: char,
    third_cells: [(i8,i8);4],
}
#[derive(Clone, PartialEq, Eq, Hash)]
struct NodeKey { board: Board, hold: char, index: usize, b2b: u16, combo: i16 }

struct Request {
    next: Vec<char>, depth: usize, beam: usize, budget_ms: u64,
    initial_b2b: u16, initial_combo: i16, allow_hold: bool,
    roots: Vec<Root>,
}
struct Root { id: usize, board: Board, lines: u8, spin: Spin, hold: char, index: usize }
struct SearchResult {
    root: usize, score: f64, depth: usize, nodes: usize, elapsed_ms: u128,
    second_name: char, second_cells: [(i8,i8);4],
    third_name: char, third_cells: [(i8,i8);4],
}

fn valid_piece(p: char) -> bool { matches!(p, 'I'|'O'|'T'|'S'|'Z'|'J'|'L') }
fn parse_board(s: &str) -> Result<Board, String> {
    let parts: Vec<_> = s.split(',').collect();
    if parts.len() != H { return Err(format!("expected {H} rows, got {}", parts.len())); }
    let mut b = [0u16; H];
    for (i, part) in parts.iter().enumerate() {
        b[i] = u16::from_str_radix(part, 16).map_err(|_| "invalid hex row")?;
        if b[i] & !FULL != 0 { return Err(format!("row {i} outside board")); }
    }
    Ok(b)
}
fn parse_request(line: &str) -> Result<Request, String> {
    let mut it = line.split_whitespace();
    if it.next() != Some("V7") { return Err("bad protocol marker".into()); }
    let next_str = it.next().ok_or("missing queue")?;
    let next = if next_str == "-" { vec![] } else { next_str.chars().collect::<Vec<_>>() };
    if !next.iter().all(|p| valid_piece(*p)) { return Err("invalid queue".into()); }
    macro_rules! number { ($ty:ty) => { it.next().ok_or("missing field")?.parse::<$ty>().map_err(|_| "bad integer")? } }
    let depth: usize = number!(usize);
    let beam: usize = number!(usize);
    let budget_ms: u64 = number!(u64);
    let initial_b2b: u16 = number!(u16);
    let initial_combo: i16 = number!(i16);
    let allow_hold = match it.next() { Some("1") => true, Some("0") => false, _ => return Err("bad allow_hold".into()) };
    let count: usize = number!(usize);
    if !(1..=10).contains(&depth) || !(1..=512).contains(&beam) || count > 1000 || budget_ms > 60000 || count == 0 {
        return Err("invalid search limits".into());
    }
    let mut roots = Vec::with_capacity(count);
    for _ in 0..count {
        let id: usize = number!(usize);
        let board = parse_board(it.next().ok_or("missing board")?)?;
        let lines: u8 = number!(u8);
        if lines > 4 { return Err("invalid lines".into()); }
        let spin = Spin::parse(it.next().ok_or("missing spin")?)?;
        let hold = it.next().ok_or("missing hold")?.chars().next().ok_or("empty hold")?;
        if hold != '-' && !valid_piece(hold) { return Err("invalid hold".into()); }
        let index: usize = number!(usize);
        if index > next.len() { return Err("invalid queue index".into()); }
        roots.push(Root { id, board, lines, spin, hold, index });
    }
    if it.next().is_some() { return Err("trailing input".into()); }
    Ok(Request { next, depth, beam, budget_ms, initial_b2b, initial_combo, allow_hold, roots })
}

fn piece_base(name: char) -> ([(i8,i8);4], i8) {
    match name {
        'I' => ([(0,1),(1,1),(2,1),(3,1)],4),
        'O' => ([(0,0),(1,0),(0,1),(1,1)],2),
        'T' => ([(1,0),(0,1),(1,1),(2,1)],3),
        'S' => ([(1,0),(2,0),(0,1),(1,1)],3),
        'Z' => ([(0,0),(1,0),(1,1),(2,1)],3),
        'J' => ([(0,0),(0,1),(1,1),(2,1)],3),
        'L' => ([(2,0),(0,1),(1,1),(2,1)],3),
        _ => unreachable!("validated piece name"),
    }
}
fn cells(name: char, pose: Pose) -> [(i8,i8);4] {
    let (mut shape, size) = piece_base(name);
    for _ in 0..pose.r {
        for p in &mut shape { *p = (size - 1 - p.1, p.0); }
    }
    for p in &mut shape { p.0 += pose.x; p.1 += pose.y; }
    shape
}
fn occupied(board: &Board, x: i8, y: i8) -> bool {
    x < 0 || x >= W as i8 || y >= H as i8 || (y >= 0 && board[y as usize] & (1 << x) != 0)
}
fn legal(board: &Board, name: char, pose: Pose) -> bool {
    cells(name, pose).iter().all(|&(x,y)| y >= -8 && !occupied(board,x,y))
}
// SRS 90 degree kick tables in screen coordinates (positive Y down).
// Symmetric I kicks approximate SRS+; 180 kicks are not yet enabled.
fn kicks(name: char, from: u8, to: u8) -> [(i8,i8);5] {
    let up: [(i8,i8);5] = if name == 'I' {
        match (from,to) {
            (0,1) => [(0,0),(-2,0),(1,0),(1,2),(-2,-1)],
            (1,0) => [(0,0),(2,0),(-1,0),(2,1),(-1,-2)],
            (1,2) => [(0,0),(-1,0),(2,0),(-1,2),(2,-1)],
            (2,1) => [(0,0),(-2,0),(1,0),(-2,1),(1,-1)],
            (2,3) => [(0,0),(2,0),(-1,0),(2,1),(-1,-1)],
            (3,2) => [(0,0),(1,0),(-2,0),(1,2),(-2,-1)],
            (3,0) => [(0,0),(-2,0),(1,0),(1,-2),(-2,1)],
            (0,3) => [(0,0),(2,0),(-1,0),(-1,2),(2,-1)],
            _ => [(0,0);5],
        }
    } else {
        match (from,to) {
            (0,1) => [(0,0),(-1,0),(-1,1),(0,-2),(-1,-2)],
            (1,0) => [(0,0),(1,0),(1,-1),(0,2),(1,2)],
            (1,2) => [(0,0),(1,0),(1,-1),(0,2),(1,2)],
            (2,1) => [(0,0),(-1,0),(-1,1),(0,-2),(-1,-2)],
            (2,3) => [(0,0),(1,0),(1,1),(0,-2),(1,-2)],
            (3,2) => [(0,0),(-1,0),(-1,-1),(0,2),(-1,2)],
            (3,0) => [(0,0),(-1,0),(-1,-1),(0,2),(-1,2)],
            (0,3) => [(0,0),(1,0),(1,1),(0,-2),(1,-2)],
            _ => [(0,0);5],
        }
    };
    up.map(|(dx,dy)| (dx,-dy))
}
fn rotate(board: &Board, name: char, pose: Pose, dir: i8) -> Option<Pose> {
    if name == 'O' { return None; }
    let r = ((pose.r as i8 + dir).rem_euclid(4)) as u8;
    for (dx,dy) in kicks(name,pose.r,r) {
        let next = Pose { x: pose.x+dx, y: pose.y+dy, r };
        if legal(board,name,next) { return Some(next); }
    }
    None
}
fn fall(board: &Board, name: char, mut p: Pose) -> Pose {
    while legal(board,name,Pose { y:p.y+1, ..p }) { p.y += 1; }
    p
}
fn spin_type(board: &Board, pose: Pose, last_rotation: bool) -> Spin {
    if !last_rotation { return Spin::None; }
    let cx = pose.x+1; let cy = pose.y+1;
    let coords = [(cx-1,cy-1),(cx+1,cy-1),(cx-1,cy+1),(cx+1,cy+1)];
    let corners = coords.map(|(x,y)| occupied(board,x,y));
    if corners.iter().filter(|&&occupied| occupied).count() < 3 { return Spin::None; }
    let front = match pose.r { 0 => (0,1), 1 => (1,3), 2 => (2,3), _ => (0,2) };
    if corners[front.0] && corners[front.1] { Spin::Full } else { Spin::Mini }
}
fn lock(board: &Board, shape: [(i8,i8);4]) -> Option<(Board,u8)> {
    if shape.iter().any(|&(x,y)| x < 0 || x >= W as i8 || y < 0 || y >= H as i8 || occupied(board,x,y)) {
        return None;
    }
    let mut result = *board;
    for (x,y) in shape { result[y as usize] |= 1 << x; }
    let mut post = [0u16;H];
    let mut dest = H;
    for src in (0..H).rev() {
        if result[src] != FULL { dest -= 1; post[dest] = result[src]; }
    }
    Some((post, dest as u8)) // full rows occupy the now-empty leading rows
}

fn locks(board: &Board, name: char, max_states: usize) -> Vec<Lock> {
    // Exact collision checks on future pieces; ignores gravity/lock-timing.
    let start = Pose { x:3, y:-2, r:0 };
    if !legal(board,name,start) { return Vec::new(); }
    let initial = State { pose:start, last_rotation:false };
    let mut q = VecDeque::from([initial]);
    let mut seen = HashSet::from([initial]);
    let mut output = Vec::new();
    let mut dedup = HashSet::new();
    while let Some(s) = q.pop_front() {
        let landing = fall(board,name,s.pose);
        let spin = if name == 'T' && landing == s.pose { spin_type(board,landing,s.last_rotation) } else { Spin::None };
        let shape = cells(name,landing);
        if let Some((post,lines)) = lock(board,shape) {
            // Dedup equivalent states, preserving distinct spin classifications.
            let mut canonical = shape;
            canonical.sort_unstable();
            if dedup.insert((canonical,spin)) {
                output.push(Lock { board:post, lines, spin, cells:shape });
            }
        }
        if seen.len() >= max_states { continue; }
        let p = s.pose;
        let mut next = Vec::with_capacity(5);
        for dx in [-1,1] {
            let t = Pose {x:p.x+dx, ..p};
            if legal(board,name,t) { next.push(State {pose:t,last_rotation:false}); }
        }
        let down = Pose {y:p.y+1, ..p};
        if legal(board,name,down) { next.push(State {pose:down,last_rotation:false}); }
        for dir in [-1,1] {
            if let Some(t) = rotate(board,name,p,dir) { next.push(State {pose:t,last_rotation:true}); }
        }
        for new in next {
            if seen.len() >= max_states { break; }
            if seen.insert(new) { q.push_back(new); }
        }
    }
    output
}

fn board_value(rows: &Board) -> f64 {
    // Moderate clean stacks should survive; cavities and top-out are dangerous.
    let mut heights = [0i32;W];
    let mut holes=0f64;
    let mut covers=0f64;
    let mut row_trans=0f64;
    for x in 0..W {
        let mut blocks_above=0i32;
        for (y,&row) in rows.iter().enumerate() {
            let full = row & (1<<x) != 0;
            if full {
                if heights[x]==0 { heights[x]=(H-y) as i32; }
                blocks_above += 1;
            } else if blocks_above>0 {
                holes+=1.0;
                covers+=(blocks_above as f64).sqrt();
            }
        }
    }
    for &r in rows {
        let mut prev=true;
        for x in 0..W {
            let cur = r & (1<<x) != 0;
            if prev!=cur { row_trans += 1.0; }
            prev=cur;
        }
        if !prev { row_trans+=1.0; }
    }
    let maxh = *heights.iter().max().unwrap() as f64;
    let aggregate: i32 = heights.iter().sum();
    let mut rough = 0.0;
    for x in 0..(W-1) { rough += (heights[x] - heights[x+1]).abs() as f64; }
    let mut well_reward: f64=0.0;
    if holes == 0.0 && maxh < 17.0 {
        // Side wells only. Reward *readiness*, not indefinite height growth.
        for (well,neighbor) in [(0usize,1usize),(9usize,8usize)] {
            let depth=(heights[neighbor]-heights[well]).max(0) as f64;
            if depth >= 2.0 {
                let other_well_depth = heights[if well==0 {9} else {0}];
                // Don't pay for multiple simultaneous edge wells.
                if other_well_depth <= heights[if well==0 {8} else {1}] {
                    well_reward=well_reward.max(1.65*depth.min(4.0) + 1.2*(depth-3.0).max(0.0).min(1.0));
                }
            }
        }
    }
    let danger=(maxh-12.0).max(0.0);
    let emergency=(maxh-16.0).max(0.0);
    - 6.8*holes - 0.9*covers - 0.21*row_trans
        - 0.19*rough - 0.026*(aggregate as f64)
        - 0.45*danger*danger - 3.0*emergency*emergency + well_reward
}
fn reward(lines: u8, spin: Spin, b2b: u16, combo: i16, perfect_clear: bool) -> (f64,u16,i16) {
    let base = match spin {
        Spin::None => match lines { 2=>1.0,3=>2.0,4=>4.0,_=>0.0 },
        Spin::Mini => match lines {1=>1.0,2=>2.0,_=>0.0},
        Spin::Full => match lines {1=>2.0,2=>4.0,3=>6.0,_=>0.0},
    };
    let difficult=lines==4 || (lines>0 && spin!=Spin::None);
    let new_combo=if lines>0 { combo.saturating_add(1) } else { -1 };
    let new_b2b=if difficult { b2b.saturating_add(1) } else if lines>0 { 0 } else { b2b };
    let tier=if new_b2b<2 {0.0} else if new_b2b<4 {1.0} else if new_b2b<9 {2.0} else if new_b2b<25 {3.0} else {4.0};
    let attack=(base*(1.0+0.25*(new_combo.max(0) as f64))).floor() + if difficult {tier} else {0.0};
    let b2b_value = if difficult { 2.0+0.3*(new_b2b as f64).min(15.0) } else {0.0};
    let pc_value = if perfect_clear && lines>0 { 45.0 } else {0.0};
    let break_cost = if lines>0 && !difficult && b2b>0 { 2.0+(b2b as f64).min(8.0)*0.5 } else {0.0};
    (7.0*attack+0.35*lines as f64+b2b_value+pc_value-break_cost,new_b2b,new_combo)
}

fn score_next(parent: &Node, mv: Lock, name: char, hold: char, idx: usize, depth: usize,
              cache: &mut HashMap<Board,f64>) -> Node {
    let (gain,b2b,combo)=reward(mv.lines,mv.spin,parent.b2b,parent.combo,mv.board.iter().all(|&r|r==0));
    let discount=0.95f64.powi(depth as i32);
    let total=parent.reward+discount*gain;
    let value=*cache.entry(mv.board).or_insert_with(|| board_value(&mv.board));
    let (second_name,second_cells,third_name,third_cells)=if parent.second_name=='-' {
        (name,mv.cells,'-',[(0,0);4])
    } else if parent.third_name=='-' {
        (parent.second_name,parent.second_cells,name,mv.cells)
    } else {
        (parent.second_name,parent.second_cells,parent.third_name,parent.third_cells)
    };
    Node { board:mv.board, hold, index:idx, b2b, combo, reward:total,
        estimate:total+value, root:parent.root, second_name, second_cells,
        third_name, third_cells }
}
fn prune(nodes: Vec<Node>, width: usize, diversity: usize) -> Vec<Node> {
    let mut seen: HashMap<NodeKey,Node> = HashMap::new();
    for n in nodes {
        let key=NodeKey {board:n.board,hold:n.hold,index:n.index,b2b:n.b2b,combo:n.combo};
        match seen.get_mut(&key) {
            Some(old) => { if n.estimate > old.estimate { *old=n; } },
            None => { seen.insert(key,n); },
        }
    }
    let mut sorted: Vec<Node> = seen.into_values().collect();
    sorted.sort_by(|a,b| b.estimate.total_cmp(&a.estimate).then_with(|| a.root.cmp(&b.root)));
    let mut result = Vec::with_capacity(width);
    let mut used_roots = HashSet::new();
    for node in &sorted {
        if used_roots.insert(node.root) {
            result.push(node.clone());
            if result.len()>=diversity.min(width) {break;}
        }
    }
    for node in sorted {
        if result.len()>=width {break;}
        if !result.iter().any(|r| r.root==node.root && r.index==node.index && r.board==node.board && r.hold==node.hold) {
            result.push(node);
        }
    }
    result
}

fn search(request: Request) -> SearchResult {
    let start=Instant::now();
    let deadline=start+Duration::from_millis(request.budget_ms.max(20));
    let mut cache: HashMap<Board,f64>=HashMap::new();
    let mut all = Vec::new();
    let mut expanded=0;
    for r in request.roots.iter() {
        let (gain,b2b,combo)=reward(r.lines,r.spin,request.initial_b2b,request.initial_combo,r.board.iter().all(|&v|v==0));
        let value=*cache.entry(r.board).or_insert_with(||board_value(&r.board));
        all.push(Node {board:r.board,hold:r.hold,index:r.index,b2b,combo,
            reward:gain,estimate:gain+value,root:r.id,
            second_name:'-',second_cells:[(0,0);4],
            third_name:'-',third_cells:[(0,0);4]});
        expanded+=1;
    }
    let mut beam=prune(all,request.beam,request.beam.min(12));
    let mut best=beam[0].clone();
    let mut depth_reached=1;
    // Do not accept a partially expanded beam; keep previous completed depth.
    for ply in 1..request.depth {
        if Instant::now()>=deadline { break; }
        let mut next=Vec::new();
        let mut interrupted=false;
        for parent in beam.iter() {
            if Instant::now() >= deadline { interrupted=true; break; }
            if parent.index>=request.next.len() { continue; }
            let natural=request.next[parent.index];
            let mut options=vec![(natural,parent.hold,parent.index+1)];
            if request.allow_hold {
                if parent.hold!='-' {
                    options.push((parent.hold,natural,parent.index+1));
                } else if parent.index+1 < request.next.len() {
                    options.push((request.next[parent.index+1],natural,parent.index+2));
                }
            }
            let mut local=Vec::new();
            for (piece,held,idx) in options {
                for mv in locks(&parent.board,piece,1100) {
                    local.push(score_next(parent,mv,piece,held,idx,ply,&mut cache));
                    expanded+=1;
                }
            }
            local.sort_by(|a,b| b.estimate.total_cmp(&a.estimate));
            next.extend(local.into_iter().take(9));
        }
        if interrupted || next.is_empty() { break; }
        beam=prune(next,request.beam,request.beam.min(12));
        best=beam[0].clone();
        depth_reached+=1;
    }
    SearchResult {root:best.root,score:best.estimate,depth:depth_reached,
        nodes:expanded,elapsed_ms:start.elapsed().as_millis(),
        second_name:best.second_name,second_cells:best.second_cells,
        third_name:best.third_name,third_cells:best.third_cells}
}
fn handle_request(line: &str) -> String {
    match parse_request(line) {
        Ok(req) => {
            let result=search(req);
            let future=if result.second_name=='-' {
                "-".to_string()
            } else {
                let encoded=result.second_cells.iter()
                    .map(|(x,y)|format!("{x},{y}"))
                    .collect::<Vec<_>>().join(";");
                format!("{}:{encoded}",result.second_name)
            };
            let third=if result.third_name=='-' {
                "-".to_string()
            } else {
                let encoded=result.third_cells.iter()
                    .map(|(x,y)|format!("{x},{y}"))
                    .collect::<Vec<_>>().join(";");
                format!("{}:{encoded}",result.third_name)
            };
            format!("OK {} {:.8} {} {} {} {} {}",result.root,result.score,
                result.depth,result.nodes,result.elapsed_ms,future,third)
        }
        Err(err) => format!("ERR {}",err.replace(' ',"_")),
    }
}
fn main() {
    if std::env::args().any(|arg| arg == "--version") {
        println!("trassist-v7 0.1.0 (S1-inspired; SRS 90; no 180 kicks)");
        return;
    }
    let stdin = io::stdin();
    let mut stdout=io::stdout().lock();
    for line in stdin.lock().lines() {
        let message=match line {Ok(s)=>s,Err(_)=>break};
        let output=handle_request(&message);
        if writeln!(stdout,"{output}").is_err() {break;}
        if stdout.flush().is_err() {break;}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn quad_clears_four_rows() {
        let mut b=[0u16;H];
        for y in 16..20 {b[y]=FULL ^ 1;}
        let i=[(0,16),(0,17),(0,18),(0,19)];
        let (after,cleared)=lock(&b,i).unwrap();
        assert_eq!(cleared,4);
        assert_eq!(after,[0;H]);
    }
    #[test]
    fn t_corners_require_last_rotation() {
        let mut b=[0u16;H];
        b[17]=(1<<3)|(1<<5);
        b[19]=1<<3;
        let pose=Pose{x:3,y:17,r:0};
        assert_eq!(spin_type(&b,pose,false),Spin::None);
        assert_eq!(spin_type(&b,pose,true),Spin::Full);
    }
    #[test]
    fn clean_quad_well_beats_corrupt_same_height() {
        let mut clean=[0u16;H];
        for row in clean.iter_mut().skip(16) {*row=FULL^1;}
        let mut bad=clean;
        bad[18] &= !(1<<4);
        assert!(board_value(&clean)>board_value(&bad));
    }
    #[test]
    fn line_protocol_roundtrip() {
        let board=vec!["000";20].join(",");
        let req=format!("V7 IOS 5 24 200 0 -1 1 1 0 {board} 0 n - 0");
        let reply=handle_request(&req);
        assert!(reply.starts_with("OK "),"{reply}");
    }
    #[test]
    fn holds_and_spins_have_greater_reward() {
        let (quad,..)=reward(4,Spin::None,1,-1,false);
        let (single,..)=reward(1,Spin::None,1,-1,false);
        let (tsd,..)=reward(2,Spin::Full,1,-1,false);
        assert!(quad>single);
        assert!(tsd>single);
    }
}
