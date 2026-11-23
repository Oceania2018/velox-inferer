//! Reserved slots: the worker's table of lanes held for one client token.
//!
//! A hold has no fixed length: it lasts while the token is in use and lapses `idle` after the
//! last use (a reserve call with the token, or the token's request running on the lane, see
//! [`Reservations::touch`]), or when released.
//!
//! A hold keeps a lane out of normal admission until it lapses or is released, so the holder's
//! requests find it empty (and its KV prefix warm). A reserve call picks a lane and marks it held
//! at once; it is granted when no request of an outsider still runs on that lane. The table is
//! plain data owned by the worker thread (no locks); the worker feeds it the lanes' activity and
//! answers the replies it hands back. `W` is the reply handle of a waiting reserve call (a channel
//! sender in the engine, anything in tests).

use std::time::{Duration, Instant};

use velox_serve::SlotError;

/// A reserve call waiting for its lane's outsider work to finish.
struct Waiter<W> {
    reply: W,
    since: Instant,
    deadline: Instant,
}

/// A lane held for `token`.
struct Hold<W> {
    token: String,
    expires: Instant,
    /// `expires` as unix milliseconds, moved with it (reported as is).
    expires_ms: u64,
    /// The idle time: `expires` is this long after the token's last use.
    duration: Duration,
    /// No outsider runs on the lane any more. Until then reserve calls wait in `waiters`.
    granted: bool,
    waiters: Vec<Waiter<W>>,
}

/// A granted (or renewed) hold, as a reserve call is answered.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Grant {
    pub token: String,
    pub lane: usize,
    pub expires_ms: u64,
    pub duration: Duration,
    pub renewed: bool,
    pub waited: Duration,
}

/// A hold as `GET /v1/slots` lists it.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct HoldView {
    pub token: String,
    pub lane: usize,
    pub expires_ms: u64,
    pub granted: bool,
}

pub(crate) type Answer<W> = (W, Result<Grant, SlotError>);

pub(crate) struct Reservations<W> {
    /// One entry per lane.
    holds: Vec<Option<Hold<W>>>,
    max_holds: usize,
    /// The lane each recent token last held, so a token reserving again after its hold lapsed
    /// gets the same lane back when it is idle (its conversation's KV rows are still there).
    /// Newest last; bounded by [`RECENT_TOKENS`].
    last_lane: Vec<(String, usize)>,
}

/// Tokens whose last lane is remembered.
const RECENT_TOKENS: usize = 64;

impl<W> Reservations<W> {
    /// A table for `lanes` lanes holding at most `max_holds` of them (never all: one lane always
    /// takes normal traffic).
    pub fn new(lanes: usize, max_holds: usize) -> Self {
        Reservations {
            holds: (0..lanes).map(|_| None).collect(),
            max_holds: max_holds.min(lanes.saturating_sub(1)),
            last_lane: Vec::new(),
        }
    }

    pub fn max_holds(&self) -> usize {
        self.max_holds
    }

    fn live(&self, now: Instant) -> impl Iterator<Item = (usize, &Hold<W>)> {
        self.holds
            .iter()
            .enumerate()
            .filter_map(move |(l, h)| h.as_ref().filter(|h| h.expires > now).map(|h| (l, h)))
    }

    /// Drops expired holds nobody waits on (granted ones). Expired holds with waiters are
    /// answered by [`Self::tick`].
    fn drop_expired(&mut self, now: Instant) {
        for h in &mut self.holds {
            if h.as_ref()
                .is_some_and(|h| h.expires <= now && h.waiters.is_empty())
            {
                *h = None;
            }
        }
    }

    /// Reserve (or renew) a lane for `token` until it is idle for `duration`, waiting at most `wait` for its
    /// in-flight work. `rank(lane)` orders the candidate lanes (lowest first: idle lanes, then
    /// the one that frees soonest); `outsider_busy(lane, token)` says whether a request not
    /// running on `token`'s hold still uses the lane. Returns the answer when there is one now;
    /// otherwise the call waits in the table and [`Self::tick`] answers it.
    #[allow(clippy::too_many_arguments)]
    pub fn reserve(
        &mut self,
        token: &str,
        duration: Duration,
        wait: Duration,
        now: Instant,
        now_ms: u64,
        reply: W,
        rank: impl Fn(usize) -> (u8, u64),
        outsider_busy: impl Fn(usize, &str) -> bool,
    ) -> Option<Answer<W>> {
        self.drop_expired(now);
        let expires = now + duration;
        let expires_ms = now_ms + duration.as_millis() as u64;
        if let Some(lane) = self.lane_of(token, now) {
            // Renewal: the hold now ends `duration` from this call.
            let h = self.holds[lane].as_mut().expect("lane_of found it");
            h.expires = expires;
            h.expires_ms = expires_ms;
            h.duration = duration;
            if h.granted {
                return Some((
                    reply,
                    Ok(Grant {
                        token: token.to_string(),
                        lane,
                        expires_ms,
                        duration,
                        renewed: true,
                        waited: Duration::ZERO,
                    }),
                ));
            }
            // Still waiting for the lane: this call waits with the first one.
            h.waiters.push(Waiter {
                reply,
                since: now,
                deadline: now + wait,
            });
            return None;
        }
        if self.live(now).count() >= self.max_holds {
            let msg = format!(
                "all {} reservable slots are held (max holds {})",
                self.max_holds, self.max_holds
            );
            return Some((reply, Err(SlotError::NoSlot(msg))));
        }
        let previous = self
            .last_lane
            .iter()
            .rev()
            .find(|(t, _)| t == token)
            .map(|&(_, l)| l)
            .filter(|&l| self.holds[l].is_none() && rank(l).0 == 0);
        let Some(lane) = previous.or_else(|| {
            (0..self.holds.len())
                .filter(|&l| self.holds[l].is_none())
                .min_by_key(|&l| (rank(l), l))
        }) else {
            return Some((reply, Err(SlotError::NoSlot("no lane to hold".into()))));
        };
        self.last_lane.retain(|(t, _)| t != token);
        self.last_lane.push((token.to_string(), lane));
        if self.last_lane.len() > RECENT_TOKENS {
            self.last_lane.remove(0);
        }
        let granted = !outsider_busy(lane, token);
        let mut hold = Hold {
            token: token.to_string(),
            expires,
            expires_ms,
            duration,
            granted,
            waiters: Vec::new(),
        };
        if granted {
            self.holds[lane] = Some(hold);
            return Some((
                reply,
                Ok(Grant {
                    token: token.to_string(),
                    lane,
                    expires_ms,
                    duration,
                    renewed: false,
                    waited: Duration::ZERO,
                }),
            ));
        }
        // Held from now on (no new outsider is admitted); granted once the lane's work is done.
        hold.waiters.push(Waiter {
            reply,
            since: now,
            deadline: now + wait,
        });
        self.holds[lane] = Some(hold);
        None
    }

    /// Settles the table at `now`: grants waiting holds whose lane no outsider uses any more,
    /// answers waiters past their deadline (a hold nobody waits for any more is dropped) and
    /// drops expired holds.
    pub fn tick(
        &mut self,
        now: Instant,
        outsider_busy: impl Fn(usize, &str) -> bool,
    ) -> Vec<Answer<W>> {
        let mut out = Vec::new();
        for (lane, slot) in self.holds.iter_mut().enumerate() {
            let Some(h) = slot else { continue };
            if h.granted {
                if h.expires <= now {
                    *slot = None;
                }
                continue;
            }
            let timeout = |since: Instant| {
                SlotError::WaitTimeout(format!(
                    "lane {lane} still busy after {} ms",
                    now.duration_since(since).as_millis()
                ))
            };
            if h.expires <= now {
                // Expired before the lane freed up (duration shorter than the wait).
                for w in h.waiters.drain(..) {
                    out.push((w.reply, Err(timeout(w.since))));
                }
                *slot = None;
                continue;
            }
            if !outsider_busy(lane, &h.token) {
                h.granted = true;
                for w in h.waiters.drain(..) {
                    out.push((
                        w.reply,
                        Ok(Grant {
                            token: h.token.clone(),
                            lane,
                            expires_ms: h.expires_ms,
                            duration: h.duration,
                            renewed: false,
                            waited: now.duration_since(w.since),
                        }),
                    ));
                }
                continue;
            }
            let (late, keep): (Vec<_>, Vec<_>) =
                h.waiters.drain(..).partition(|w| w.deadline <= now);
            h.waiters = keep;
            for w in late {
                out.push((w.reply, Err(timeout(w.since))));
            }
            if h.waiters.is_empty() {
                // The tentative hold goes with its last waiter.
                *slot = None;
            }
        }
        out
    }

    /// The holder of `lane` is using it at `now` (its request runs there): the hold now lapses
    /// its idle time from `now`. Moves the expiry only when it gains at least a second, so the
    /// published list does not change every round.
    pub fn touch(&mut self, lane: usize, now: Instant, now_ms: u64) {
        if let Some(h) = self.holds.get_mut(lane).and_then(Option::as_mut) {
            let expires = now + h.duration;
            if h.expires > now && expires >= h.expires + Duration::from_secs(1) {
                h.expires = expires;
                h.expires_ms = now_ms + h.duration.as_millis() as u64;
            }
        }
    }

    /// Releases `token`'s hold. `None` when it holds nothing (unknown or expired); otherwise the
    /// reserve calls still waiting on it, for the caller to answer.
    pub fn release(&mut self, token: &str, now: Instant) -> Option<Vec<W>> {
        let lane = self.lane_of(token, now)?;
        let h = self.holds[lane].take().expect("lane_of found it");
        Some(h.waiters.into_iter().map(|w| w.reply).collect())
    }

    /// The lane `token` holds (granted or still waiting), unless the hold has expired.
    pub fn lane_of(&self, token: &str, now: Instant) -> Option<usize> {
        self.live(now)
            .find(|(_, h)| h.token == token)
            .map(|(l, _)| l)
    }

    /// Lanes held at `now` (closed to outsiders).
    pub fn held_lanes(&self, now: Instant) -> Vec<usize> {
        self.live(now).map(|(l, _)| l).collect()
    }

    /// When the table next needs [`Self::tick`]: the earliest expiry or wait deadline.
    pub fn next_deadline(&self) -> Option<Instant> {
        self.holds
            .iter()
            .flatten()
            .flat_map(|h| std::iter::once(h.expires).chain(h.waiters.iter().map(|w| w.deadline)))
            .min()
    }

    pub fn view(&self, now: Instant) -> Vec<HoldView> {
        self.live(now)
            .map(|(lane, h)| HoldView {
                token: h.token.clone(),
                lane,
                expires_ms: h.expires_ms,
                granted: h.granted,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: Duration = Duration::from_secs(1);

    fn idle(_: usize) -> (u8, u64) {
        (0, 0)
    }

    fn free(_: usize, _: &str) -> bool {
        false
    }

    fn ok(a: Option<Answer<u32>>) -> (u32, Grant) {
        match a {
            Some((w, Ok(g))) => (w, g),
            Some((_, Err(e))) => panic!("expected a grant, got {e:?}"),
            None => panic!("expected an answer, the call waits"),
        }
    }

    #[test]
    fn use_keeps_a_hold_alive() {
        let mut r = Reservations::new(4, 2);
        let t0 = Instant::now();
        let (_, g) = ok(r.reserve("a", 10 * S, S, t0, 1_000, 0, idle, free));
        // Used at 8 s: lapses 10 s after that, not at 10 s.
        r.touch(g.lane, t0 + 8 * S, 9_000);
        assert_eq!(r.lane_of("a", t0 + 15 * S), Some(g.lane));
        assert_eq!(r.view(t0 + 15 * S)[0].expires_ms, 19_000);
        // Under a second of gain: unchanged.
        r.touch(g.lane, t0 + 8 * S + S / 2, 9_500);
        assert_eq!(r.view(t0)[0].expires_ms, 19_000);
        assert_eq!(r.lane_of("a", t0 + 18 * S + S), None);
        // A lapsed hold is not revived by a late touch.
        r.touch(g.lane, t0 + 20 * S, 21_000);
        assert_eq!(r.lane_of("a", t0 + 20 * S), None);
    }

    #[test]
    fn reserving_again_gets_the_same_lane_back() {
        let mut r = Reservations::new(4, 2);
        let t0 = Instant::now();
        // "a" is given lane 3 (least recently used), then lapses.
        let lru = |l: usize| (0, [5, 6, 7, 1][l]);
        let (_, g) = ok(r.reserve("a", 10 * S, S, t0, 0, 0, lru, free));
        assert_eq!(g.lane, 3);
        // Reserving again after the lapse: lane 3 again, though lane 0 is now older.
        let older = |l: usize| (0, [0, 6, 7, 9][l]);
        let (_, g) = ok(r.reserve("a", 10 * S, S, t0 + 11 * S, 0, 0, older, free));
        assert_eq!(g.lane, 3);
        // Unless that lane is busy: then the usual choice.
        r.release("a", t0 + 12 * S);
        let busy3 = |l: usize| if l == 3 { (1, 5) } else { (0, [0, 6, 7, 9][l]) };
        let (_, g) = ok(r.reserve("a", 10 * S, S, t0 + 12 * S, 0, 0, busy3, free));
        assert_eq!(g.lane, 0);
    }

    #[test]
    fn grants_an_idle_lane_at_once() {
        let mut r = Reservations::new(4, 2);
        let t0 = Instant::now();
        let (w, g) = ok(r.reserve("a", 10 * S, 30 * S, t0, 1_000, 7, idle, free));
        assert_eq!(w, 7);
        assert_eq!(g.lane, 0);
        assert!(!g.renewed);
        assert_eq!(g.expires_ms, 11_000);
        assert_eq!(r.lane_of("a", t0), Some(0));
        assert_eq!(r.held_lanes(t0), vec![0]);
    }

    #[test]
    fn prefers_the_best_ranked_lane() {
        let mut r = Reservations::new(4, 2);
        let t0 = Instant::now();
        // Lanes 0 and 1 busy (1 frees sooner), 2 idle but used recently, 3 idle and older.
        let rank = |l: usize| [(1, 50), (1, 10), (0, 9), (0, 3)][l];
        let (_, g) = ok(r.reserve("a", S, S, t0, 0, 0, rank, free));
        assert_eq!(g.lane, 3);
        let (_, g) = ok(r.reserve("b", S, S, t0, 0, 0, rank, free));
        assert_eq!(g.lane, 2);
    }

    #[test]
    fn renewal_extends_and_answers_at_once() {
        let mut r = Reservations::new(4, 2);
        let t0 = Instant::now();
        ok(r.reserve("a", 10 * S, S, t0, 0, 0, idle, free));
        let t1 = t0 + 8 * S;
        let (_, g) = ok(r.reserve("a", 10 * S, S, t1, 8_000, 0, idle, |_, _| true));
        assert!(g.renewed);
        assert_eq!(g.lane, 0);
        assert_eq!(g.expires_ms, 18_000);
        // Still held past the first expiry.
        assert_eq!(r.lane_of("a", t0 + 15 * S), Some(0));
        assert!(r.tick(t0 + 15 * S, free).is_empty());
        assert_eq!(r.lane_of("a", t0 + 18 * S), None);
    }

    #[test]
    fn conflict_when_max_holds_are_taken() {
        let mut r = Reservations::new(4, 2);
        let t0 = Instant::now();
        ok(r.reserve("a", 10 * S, S, t0, 0, 0, idle, free));
        ok(r.reserve("b", 10 * S, S, t0, 0, 0, idle, free));
        match r.reserve("c", 10 * S, S, t0, 0, 9, idle, free) {
            Some((9, Err(SlotError::NoSlot(_)))) => {}
            other => panic!(
                "expected no_slot, got {:?}",
                other.map(|(w, r)| (w, r.is_ok()))
            ),
        }
        // The holders themselves still renew.
        assert!(
            ok(r.reserve("b", 10 * S, S, t0, 0, 0, idle, free))
                .1
                .renewed
        );
        // A hold that expired frees its place.
        ok(r.reserve("c", 10 * S, S, t0 + 11 * S, 0, 0, idle, free));
    }

    #[test]
    fn never_holds_every_lane() {
        let r: Reservations<u32> = Reservations::new(4, 9);
        assert_eq!(r.max_holds(), 3);
        let r: Reservations<u32> = Reservations::new(1, 2);
        assert_eq!(r.max_holds(), 0);
    }

    #[test]
    fn expiry_releases_the_lane() {
        let mut r = Reservations::new(4, 2);
        let t0 = Instant::now();
        ok(r.reserve("a", 5 * S, S, t0, 0, 0, idle, free));
        assert_eq!(r.next_deadline(), Some(t0 + 5 * S));
        // Lazily: an expired hold no longer counts, even before a tick.
        assert_eq!(r.lane_of("a", t0 + 5 * S), None);
        assert!(r.held_lanes(t0 + 6 * S).is_empty());
        assert!(r.tick(t0 + 6 * S, free).is_empty());
        assert_eq!(r.next_deadline(), None);
        assert!(r.release("a", t0 + 6 * S).is_none());
    }

    #[test]
    fn busy_lane_is_held_then_granted_when_it_frees() {
        let mut r = Reservations::new(2, 1);
        let t0 = Instant::now();
        let busy = |l: usize, _: &str| l == 0;
        let rank = |l: usize| if l == 0 { (1, 5) } else { (1, 9) };
        assert!(r
            .reserve("a", 60 * S, 30 * S, t0, 0, 4, rank, busy)
            .is_none());
        // Held at once: closed to outsiders while the in-flight request finishes.
        assert_eq!(r.held_lanes(t0), vec![0]);
        assert!(r.tick(t0 + S, busy).is_empty());
        let mut a = r.tick(t0 + 3 * S, free);
        assert_eq!(a.len(), 1);
        let (w, g) = ok(a.pop());
        assert_eq!((w, g.lane, g.renewed), (4, 0, false));
        assert_eq!(g.waited, 3 * S);
        // Renewal of a granted hold now answers at once.
        assert!(
            ok(r.reserve("a", 60 * S, S, t0 + 4 * S, 0, 0, rank, busy))
                .1
                .renewed
        );
    }

    #[test]
    fn wait_timeout_drops_the_tentative_hold() {
        let mut r = Reservations::new(2, 1);
        let t0 = Instant::now();
        let busy = |_: usize, _: &str| true;
        assert!(r
            .reserve("a", 60 * S, 2 * S, t0, 0, 4, idle, busy)
            .is_none());
        assert_eq!(r.next_deadline(), Some(t0 + 2 * S));
        assert!(r.tick(t0 + S, busy).is_empty());
        let a = r.tick(t0 + 2 * S, busy);
        assert!(matches!(
            a.as_slice(),
            [(4, Err(SlotError::WaitTimeout(_)))]
        ));
        assert!(r.held_lanes(t0 + 2 * S).is_empty());
        assert_eq!(r.lane_of("a", t0 + 2 * S), None);
        // The place is free for the next caller.
        assert!(r
            .reserve("b", 60 * S, S, t0 + 3 * S, 0, 5, idle, busy)
            .is_none());
    }

    #[test]
    fn release_answers_waiters_and_frees_the_lane() {
        let mut r = Reservations::new(2, 1);
        let t0 = Instant::now();
        let busy = |_: usize, _: &str| true;
        assert!(r
            .reserve("a", 60 * S, 9 * S, t0, 0, 4, idle, busy)
            .is_none());
        assert_eq!(r.release("a", t0), Some(vec![4]));
        assert!(r.held_lanes(t0).is_empty());
        assert!(r.release("a", t0).is_none());
    }

    #[test]
    fn holder_work_does_not_block_a_grant() {
        let mut r = Reservations::new(2, 1);
        let t0 = Instant::now();
        assert!(r
            .reserve("a", 60 * S, S, t0, 0, 0, idle, |_, _| true)
            .is_none());
        // The outsider is done; lane 0 now runs a request of token "a" itself.
        let a = r.tick(t0, |_, t| t != "a");
        assert!(matches!(a.as_slice(), [(0, Ok(_))]));
    }
}
