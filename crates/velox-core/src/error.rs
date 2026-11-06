use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The request is valid but this backend, variant or device cannot serve it.
    Unsupported(String),
    /// Buffer lengths or dimensions do not match the op's contract.
    Shape(String),
    /// A device runtime call failed.
    Backend(String),
    /// Invalid user configuration (unknown profile, bad flag, ...).
    Config(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Unsupported(m) => write!(f, "unsupported: {m}"),
            Error::Shape(m) => write!(f, "shape error: {m}"),
            Error::Backend(m) => write!(f, "backend error: {m}"),
            Error::Config(m) => write!(f, "config error: {m}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// Set once the device runtime reports an error it cannot recover from (a CUDA "sticky" error
/// such as 700 or 999): every later call on the context fails the same way, so the process
/// must restart to serve again. Process-wide because a context is.
static DEVICE_LOST: std::sync::OnceLock<String> = std::sync::OnceLock::new();

/// Records that the device context is unusable; the first reason is kept.
pub fn mark_device_lost(reason: &str) {
    if DEVICE_LOST.set(reason.to_owned()).is_ok() {
        eprintln!("[fatal] device lost: {reason}");
    }
}

/// The reason the device context became unusable, if it has.
pub fn device_lost() -> Option<&'static str> {
    DEVICE_LOST.get().map(String::as_str)
}
