//use super::{Asn1Object, DerNull};
//use std::any::Any;
//use std::fmt::Display;

//pub trait Asn1Null: Asn1Object + Display {

//}

// pub struct Asn1NullImpl;

// impl Display for Asn1NullImpl {
//     fn fmt(&self, f: &mut Formatter) -> Result {
//         write!(f, "NULL")
//     }
// }
// use crate::Result;
// use std::fmt::Display;
// use std::fmt::Formatter;
// use super::asn1_object::Asn1ObjectImpl;
// use super::asn1_object::Asn1ObjectInternal;
// use super::Asn1Object;

// pub trait Asn1Null: Asn1Object {}

// pub(crate) fn fmt(f: &mut Formatter) -> Result {
//     write!(f, "NULL")
// }

// pub(crate) trait Asn1NullInternal : Asn1ObjectInternal {
//     fn as_asn1_object_internal(&self) -> &dyn Asn1ObjectInternal;
// }

// pub struct Asn1NullImpl<'a> {
//     instance: &'a dyn Asn1NullInternal,
// }

// impl<'a> Asn1NullImpl<'a> {
//     pub(crate) fn new(instance: &'a dyn Asn1NullInternal) -> Self {
//         Asn1NullImpl { instance }
//     }
//     pub(crate) fn get_encoded_alloc(
//         &self,
//         encoding: &str,
//         pre_alloc: usize,
//         post_alloc: usize,
//     ) -> Result<Vec<u8>> {
//         Asn1ObjectImpl::new(self.instance.as_asn1_object_internal()).get_encoded_alloc(encoding, pre_alloc, post_alloc)
//     }
//     // pub(crate) fn get_encoded(&self) -> Result<Vec<u8>> {
//     //     Asn1ObjectImpl::new(self.instance.as_asn1_object_internal()).get_encoded()
//     // }
// }

// impl<'a> Display for Asn1NullImpl<'a> {
//     fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
//         write!(f, "NULL")
//     }
// }