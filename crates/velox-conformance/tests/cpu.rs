//! The suite against the CPU backend itself: exercises the harness and every case's plumbing.
use velox_conformance::run;
use velox_cpu::CpuBackend;

#[test]
fn cpu_reference_is_self_consistent() {
    let results = run(&CpuBackend).unwrap();
    assert!(results.len() > 20);
    for r in &results {
        assert!(
            r.passed(),
            "{} err {:.3e} > {:.1e}",
            r.name,
            r.max_err,
            r.tol
        );
    }
}
