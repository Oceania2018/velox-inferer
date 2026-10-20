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
