//! GPU time per op from CUDA events (`VELOX_GPU_TIMING=1`). Unlike the synchronizing
//! `--profile-ops` wrapper, events are recorded on the stream around each op, so the GPU runs as
//! it normally would and the totals are kernel time including inter-kernel gaps on the device.
//! The report prints when the backend is dropped.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::Mutex;

use crate::ffi;

#[derive(Default)]
struct State {
    /// (op, start, end) not yet read back.
    pending: Vec<(String, *mut c_void, *mut c_void)>,
    totals: BTreeMap<String, (f64, u64)>,
}

// SAFETY: events are plain handles used only under the mutex.
unsafe impl Send for State {}

pub(crate) struct Timing(Mutex<State>);

impl Timing {
    pub(crate) fn from_env() -> Option<Timing> {
        std::env::var_os("VELOX_GPU_TIMING").map(|_| Timing(Mutex::new(State::default())))
    }

    fn event() -> *mut c_void {
        let mut e = std::ptr::null_mut();
        // SAFETY: out-pointer to a local. A failed create leaves null, which record ignores.
        unsafe { ffi::velox_cuda_event_create(&mut e) };
        e
    }

    pub(crate) fn time<T>(&self, op: &str, f: impl FnOnce() -> T) -> T {
        let (a, b) = (Self::event(), Self::event());
        // SAFETY: valid event handles.
        unsafe { ffi::velox_cuda_event_record(a) };
        let r = f();
        unsafe { ffi::velox_cuda_event_record(b) };
        let mut s = self.0.lock().unwrap();
        s.pending.push((op.to_string(), a, b));
        if s.pending.len() >= 4096 {
            Self::flush(&mut s);
        }
        r
    }

    fn flush(s: &mut State) {
        for (op, a, b) in s.pending.drain(..) {
            let mut ms = 0f32;
            // SAFETY: events recorded above; elapsed waits for the end event.
            unsafe {
                ffi::velox_cuda_event_elapsed(a, b, &mut ms);
                ffi::velox_cuda_event_destroy(a);
                ffi::velox_cuda_event_destroy(b);
            }
            let e = s.totals.entry(op).or_insert((0.0, 0));
            e.0 += f64::from(ms) / 1e3;
            e.1 += 1;
        }
    }
}

impl Drop for Timing {
    fn drop(&mut self) {
        let mut s = self.0.lock().unwrap();
        Self::flush(&mut s);
        let total: f64 = s.totals.values().map(|v| v.0).sum();
        let mut rows: Vec<_> = s.totals.iter().collect();
        rows.sort_by(|a, b| b.1 .0.total_cmp(&a.1 .0));
        eprintln!("[gpu timing] total {total:.3}s");
        for (op, (t, n)) in rows {
            eprintln!(
                "  {op:<32} {t:>8.3}s {:>5.1}%  {n:>7} calls  {:>8.1} us/call",
                100.0 * t / total.max(1e-12),
                1e6 * t / *n as f64
            );
        }
    }
}
