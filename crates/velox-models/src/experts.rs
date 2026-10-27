//! Routed experts in system RAM and which of them live on the device
//! (docs/qwen4_exp.md#expert-offload); shared by the MoE models.
//!
//! Every layer's experts stay in system RAM; a pool of device slots shared by all layers holds
//! copies of the most selected ones. `moe_route` counts selections per expert on the device;
//! [`ExpertCache::update`] folds the counts into decayed scores and moves the best non-resident
//! experts into the slots of the worst resident ones. A move is stream-ordered (unmap, copy, map),
//! so queued steps read a consistent placement, and a slot is an exact copy: residency changes
//! speed, never results.

use velox_core::{ActPolicy, Backend, ExpertBank, ExpertMatrix, Result};

/// One layer's routed experts as `[gate | up | down]` records of `record_bytes` (a multiple of
/// 16) in system RAM the device reads directly.
pub struct ExpertRecords<B: Backend> {
    pub records: B::Buffer,
    pub record_bytes: usize,
    pub gate: ExpertMatrix,
    pub up: ExpertMatrix,
    pub down: ExpertMatrix,
}

const UNMAPPED: i32 = -1;

/// Outcome of one [`ExpertCache::update`].
#[derive(Clone, Copy, Debug, Default)]
pub struct CacheReport {
    /// Selections since the previous update served from device slots, and from system RAM.
    pub hits: u64,
    pub misses: u64,
    /// Experts copied into slots by this update.
    pub moves: usize,
    pub resident: usize,
}

impl CacheReport {
    pub fn hit_rate(&self) -> f64 {
        self.hits as f64 / (self.hits + self.misses).max(1) as f64
    }
}

pub struct ExpertCache<B: Backend> {
    slots: B::Buffer,
    pub slot_bytes: usize,
    pub capacity: usize,
    /// Per layer `[experts]`: the slot of each expert (i32 bits), or -1.
    maps: Vec<B::Buffer>,
    /// Per layer `[experts]`: selections since the last update (f32).
    counts: Vec<B::Buffer>,
    host_map: Vec<Vec<i32>>,
    /// (layer, expert) in each slot.
    owner: Vec<Option<(usize, usize)>>,
    score: Vec<Vec<f32>>,
    /// Selections of each expert since the cache was made.
    totals: Vec<Vec<u64>>,
    /// Score kept per update: an expert's score is its decayed selection count.
    pub decay: f32,
    /// When set, each update appends the selections it folded in as `(layer, expert, count)`
    /// (routing traces for studying cache policies offline).
    pub trace: Option<Vec<Vec<(u16, u16, u32)>>>,
}

impl<B: Backend> ExpertCache<B> {
    /// A pool of `capacity` slots of `slot_bytes` (0 slots: every expert read from system RAM).
    pub fn new(
        b: &B,
        layers: usize,
        experts: usize,
        slot_bytes: usize,
        capacity: usize,
    ) -> Result<Self> {
        let unmapped = vec![f32::from_bits(UNMAPPED as u32); experts];
        Ok(ExpertCache {
            slots: b.alloc((capacity.max(1) * slot_bytes).div_ceil(4))?,
            slot_bytes,
            capacity,
            maps: (0..layers)
                .map(|_| b.upload(&unmapped))
                .collect::<Result<_>>()?,
            counts: (0..layers)
                .map(|_| b.alloc(experts))
                .collect::<Result<_>>()?,
            host_map: vec![vec![UNMAPPED; experts]; layers],
            owner: vec![None; capacity],
            score: vec![vec![0.0; experts]; layers],
            totals: vec![vec![0; experts]; layers],
            decay: 0.7,
            trace: None,
        })
    }

    /// Layer `l`'s bank over its records.
    pub fn bank<'a>(&'a self, l: usize, rec: &'a ExpertRecords<B>) -> ExpertBank<'a, B::Buffer> {
        ExpertBank {
            records: &rec.records,
            slots: &self.slots,
            map: &self.maps[l],
            experts: self.host_map[l].len(),
            slot_count: self.capacity,
            slot_bytes: self.slot_bytes,
            record_bytes: rec.record_bytes,
            gate: rec.gate,
            up: rec.up,
            down: rec.down,
            act: ActPolicy::AllowA8,
        }
    }

    /// Selections of every expert (`[layer][expert]`) counted by the updates so far.
    pub fn totals(&self) -> &[Vec<u64>] {
        &self.totals
    }

    /// Layer `l`'s selection counters, for `moe_route`.
    pub fn counts_mut(&mut self, l: usize) -> &mut B::Buffer {
        &mut self.counts[l]
    }

    /// Folds the selection counts into the scores, then moves up to `max_moves` experts into
    /// slots where they score better than the residents they replace.
    /// `layers`: each cache layer's records (what [`Self::bank`] reads).
    pub fn update(
        &mut self,
        b: &B,
        layers: &[&ExpertRecords<B>],
        max_moves: usize,
    ) -> Result<CacheReport> {
        self.update_steps(b, layers, max_moves, u64::MAX, false)
    }

    /// [`Self::update`] after `steps` forward steps: hits and misses count weight reads (a step
    /// reads a selected expert once for all its rows, so at most `steps` per expert), and with
    /// `score_reads` the scores do too, so a long prefill chunk weighs what it costs to read,
    /// not its token count.
    pub fn update_steps(
        &mut self,
        b: &B,
        layers: &[&ExpertRecords<B>],
        max_moves: usize,
        steps: u64,
        score_reads: bool,
    ) -> Result<CacheReport> {
        let experts = self.host_map.first().map_or(0, Vec::len);
        let mut report = CacheReport::default();
        let zeros = vec![0f32; experts];
        let mut host = vec![0f32; experts];
        b.synchronize()?;
        let mut seen = Vec::new();
        for l in 0..self.counts.len() {
            b.download(&self.counts[l], &mut host)?;
            for (e, &c) in host.iter().enumerate() {
                let selected = c as u64;
                let c = selected.min(steps);
                if c > 0 && self.trace.is_some() {
                    seen.push((l as u16, e as u16, selected as u32));
                }
                if self.host_map[l][e] >= 0 {
                    report.hits += c;
                } else {
                    report.misses += c;
                }
                let s = if score_reads { c } else { selected };
                self.score[l][e] = self.score[l][e] * self.decay + s as f32;
                self.totals[l][e] += selected;
            }
            if host.iter().any(|&c| c != 0.0) {
                b.write_at(&mut self.counts[l], 0, &zeros)?;
            }
        }
        if let Some(t) = &mut self.trace {
            t.push(seen);
        }
        // Candidates best first; residents worst first; free slots before any eviction.
        let mut wanted: Vec<(f32, usize, usize)> = Vec::new();
        for (l, scores) in self.score.iter().enumerate() {
            for (e, &s) in scores.iter().enumerate() {
                if s > 0.0 && self.host_map[l][e] < 0 {
                    wanted.push((s, l, e));
                }
            }
        }
        wanted.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut free: Vec<usize> = (0..self.capacity)
            .rev()
            .filter(|&i| self.owner[i].is_none())
            .collect();
        let mut residents: Vec<(f32, usize)> = (0..self.capacity)
            .filter_map(|i| self.owner[i].map(|(l, e)| (self.score[l][e], i)))
            .collect();
        residents.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut moves: Vec<(usize, usize, usize)> = Vec::new();
        let mut dirty = vec![false; self.maps.len()];
        for (s, l, e) in wanted.into_iter().take(max_moves) {
            let slot = match free.pop() {
                Some(i) => i,
                None => match residents.last() {
                    // Hysteresis: an incumbent keeps its slot against near-equal scores.
                    Some(&(worst, i)) if s > worst * 1.25 + 1.0 => {
                        residents.pop();
                        let (ol, oe) = self.owner[i].take().expect("resident slot");
                        self.host_map[ol][oe] = UNMAPPED;
                        dirty[ol] = true;
                        i
                    }
                    _ => break,
                },
            };
            self.owner[slot] = Some((l, e));
            moves.push((slot, l, e));
        }
        // Unmap the evicted, copy, then map the promoted: steps already queued keep reading the
        // old placement until the copies land.
        for (l, d) in dirty.iter().enumerate() {
            if *d {
                self.write_map(b, l)?;
            }
        }
        let mut promoted = vec![false; self.maps.len()];
        for &(slot, l, e) in &moves {
            let rec = layers[l];
            b.copy(
                &rec.records,
                e * rec.record_bytes / 4,
                &mut self.slots,
                slot * self.slot_bytes / 4,
                rec.record_bytes / 4,
            )?;
            self.host_map[l][e] = slot as i32;
            promoted[l] = true;
        }
        for (l, p) in promoted.iter().enumerate() {
            if *p {
                self.write_map(b, l)?;
            }
        }
        report.moves = moves.len();
        report.resident = self.owner.iter().filter(|o| o.is_some()).count();
        Ok(report)
    }

    fn write_map(&mut self, b: &B, l: usize) -> Result<()> {
        let v: Vec<f32> = self.host_map[l]
            .iter()
            .map(|&s| f32::from_bits(s as u32))
            .collect();
        b.write_at(&mut self.maps[l], 0, &v)
    }
}
