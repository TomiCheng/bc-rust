use std::fmt::{Display, Formatter};

pub struct Asn1Null;

impl Display for Asn1Null {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("NULL")
    }
}