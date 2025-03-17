use std::any;
use std::sync;
use core::slice;

use super::*;

pub struct Asn1Sequence {
    elements: Vec<sync::Arc<dyn Asn1Convertiable>>,
}

impl Asn1Sequence {
    pub fn iter(&self) -> &[sync::Arc<dyn Asn1Convertiable>] {
        self.elements.as_slice()
    }
}

