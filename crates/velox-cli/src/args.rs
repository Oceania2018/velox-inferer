//! Minimal `--key value` parser; the CLI has no third-party dependencies.

use std::collections::HashMap;
use std::str::FromStr;

use velox_core::{Error, Result};

pub struct Args {
    pub command: Option<String>,
    opts: HashMap<String, String>,
}

impl Args {
    pub fn from_args(mut it: impl Iterator<Item = String>) -> Result<Self> {
        let command = it.next();
        let mut opts = HashMap::new();
        while let Some(key) = it.next() {
            let name = key
                .strip_prefix("--")
                .ok_or_else(|| Error::Config(format!("unexpected argument '{key}'")))?;
            let (name, value) = match name.split_once('=') {
                Some((n, v)) => (n.to_string(), v.to_string()),
                None => {
                    let v = it
                        .next()
                        .ok_or_else(|| Error::Config(format!("--{name} needs a value")))?;
                    (name.to_string(), v)
                }
            };
            opts.insert(name, value);
        }
        Ok(Args { command, opts })
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.opts.get(key).map(String::as_str)
    }

    pub fn parse<T: FromStr>(&self, key: &str, default: T) -> Result<T> {
        match self.get(key) {
            None => Ok(default),
            Some(v) => v
                .parse()
                .map_err(|_| Error::Config(format!("invalid value '{v}' for --{key}"))),
        }
    }
}
