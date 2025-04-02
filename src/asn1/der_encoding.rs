use crate::asn1::asn1_encoding::Asn1Encoding;
use std::any::Any;
use std::cmp::Ordering;

pub(crate) trait DerEncoding: Asn1Encoding  {}
