use std::sync::Arc;

pub type EcPointMapRc = Arc<dyn EcPointMap>;

pub trait EcPointMap {
    fn map(&self);
}

// TODO