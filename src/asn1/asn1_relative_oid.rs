use std::cell::LazyCell;
use std::fmt::{Display, Formatter};
use std::iter::Skip;
use std::str::Chars;
use crate::math::BigInteger;
use crate::Result;

pub struct Asn1RelativeOid {
    contents: Vec<u8>,
    
}

impl Asn1RelativeOid {
    pub fn with_str(identifier: &str) -> Result<Self> {

        let identifier = identifier.to_string();
        Ok(Asn1RelativeOid {
            contents: Vec::new(),
            
        })
    }
    pub fn is_valid_identifier(chars: Skip<Chars>) -> bool {

        todo!()
    }

    const I64_LIMIT: u64 = ((i64::MAX >> 7)  - 0x7F) as u64;
    pub(crate) fn parse_contents(contents: &[u8]) -> String {
        let mut result = String::new();
        let mut value = 0u64;
        let mut first = true;
        let mut big_value = None;
        for &b in contents {
            if value <= Self::I64_LIMIT {
                value += b as u64 & 0x7F;
                if b & 0x80 == 0 {
                    if first {
                        first = false;
                    } else {
                        result.push('.');
                    }
                    result.push_str(&value.to_string());
                    value = 0;
                } else {
                    value <<= 7;
                }
            } else {
                if big_value == None {
                    big_value = Some(BigInteger::with_u64(value));
                }
                big_value = Some(big_value.unwrap().or(&BigInteger::with_u64(b as u64 & 0x7F)));
                if b & 0x80 == 0 {
                    if first {
                        first = false;
                    } else {
                        result.push('.');
                    }

                    result.push_str(&big_value.unwrap().to_string());
                    big_value = None;
                    value = 0;
                } else {
                    big_value = Some(big_value.unwrap().shift_left(7));
                }
            }
        }
        result
    }

}

// impl Display for Asn1RelativeOid {
//     fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
//         write!(f, "{}", self.get_id())
//     }
// }

#[cfg(test)]
mod tests {
    use crate::util::encoders::hex::to_decode_with_str;
    use super::*;
    #[test]
    fn test() {
        recode_check("180.3",  &to_decode_with_str("0D03813403").unwrap());
    }

    fn recode_check(oid: &str, enc: &[u8]) {
        let o = Asn1RelativeOid::with_str(oid).unwrap();
        // let relative_oid = Asn1RelativeOid::with_str(oid).unwrap();
        // assert_eq!(relative_oid.contents, enc);
        // let oid2 = Asn1RelativeOid::parse_contents(&relative_oid.contents);
        // assert_eq!(oid2, oid);
    }
}