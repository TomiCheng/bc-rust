use std::any::Any;
use std::fmt::{Debug, Display, Formatter};
use std::io::Write;
use std::sync;
use std::sync::Arc;
use super::*;

/// A Der encoded set object
pub struct Asn1Set {
    elements: Vec<Arc<dyn Asn1Convertiable>>,
}

impl Asn1Set {
    pub fn new() -> Self {
        Asn1Set {
            elements: Vec::new(),
        }
    }
    pub fn with_element(element: Arc<dyn Asn1Convertiable>) -> Self {
        Asn1Set {
            elements: vec![element],
        }
    }
    pub fn with_elements(elements: Vec<sync::Arc<dyn Asn1Convertiable>>) -> Self {
        Asn1Set {
            elements,
        }
    }
}
