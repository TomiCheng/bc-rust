use std::sync::LazyLock;

pub struct Global {
    x509_name_default_reverse: bool,
}

impl Global {
    pub fn x509_name_default_reverse(&self) -> bool {
        self.x509_name_default_reverse
    }
}

pub static GLOBAL: LazyLock<Global> = LazyLock::new(|| Global {
    x509_name_default_reverse: option_env!("BC_RUST_X509_NAME_DEFAULT_REVERSE").map(|s| s == "true").unwrap_or(false),
});
