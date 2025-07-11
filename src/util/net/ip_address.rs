// /// Validate the given IPv4 or IPv6 address.
// ///
// /// # Arguments
// /// * `address` - the IP address as a string.
// ///
// /// # Returns
// /// `true` if the address is valid, `false` otherwise.
// pub fn is_valid(address: &str) -> bool {
//     is_valid_ipv4(address) || is_valid_ipv6(address)
// }
// pub fn is_valid_ipv4(address: &str) -> bool {
//     let length = address.chars().count();
//     if length < 7 || length > 15 {
//         return false;
//     }
//
//     let splits = address.split('.');
//     if splits.clone().count() != 4 {
//         return false;
//     }
//     for part in splits {
//         if !is_parseable_ipv4_octet(part) {
//             return false;
//         }
//     }
//     true
// }
// pub fn is_valid_ipv6(address: &str) -> bool {
//     todo!();
//     // if address.is_empty() {
//     //     return false;
//     // }
//     // if let Some(c) = address.chars().nth(0) {
//     //     if c != ':' && get_digit_hexadecimal(c).is_none() {
//     //         return false;
//     //     }
//     // }
//     //
//     // let mut segment_count = 0;
//     // let temp = format!("{}:", address);
//     // let mut double_colon_found = false;
//     //
//     // let mut pos = 0;
//     //
//     // while pos < temp.chars().count() &&  let Some(end) = temp[pos..].find(':') {
//     //     if segment_count == 8 {
//     //         return false;
//     //     }
//     //
//     //     if pos != end {
//     //         let value = &temp[pos..end];
//     //
//     //         if end == temp.chars().count() - 1 && value.find('.').is_some() {
//     //             segment_count += 1;
//     //             if segment_count == 8 {
//     //                 return false;
//     //             }
//     //
//     //             if !is_valid_ipv4(value) {
//     //                 return false;
//     //             }
//     //         } else if !is_parseable_ipv6_segment(&temp[pos..pos + end]) {
//     //             return false;
//     //         }
//     //     } else {
//     //         if end != 1 && end != temp.chars().count() - 1 && double_colon_found {
//     //             return false;
//     //         }
//     //         double_colon_found = true;
//     //     }
//     //
//     //     pos = end + 1;
//     //     segment_count += 1;
//     // }
//     // segment_count == 8 || double_colon_found
// }
// pub fn is_valid_ipv6_with_net_mask(address: &str) -> bool {
//     let index = address.find('/');
//     if let Some(index) = index {
//         let before = &address[0..index];
//         let after = &address[index + 1..];
//
//         is_valid_ipv6(before) && (is_valid_ipv6(after) || is_parseable_ipv6_mask(after))
//     } else {
//         false
//     }
// }
// pub fn is_parseable_ipv4_mask(s: &str) -> bool {
//     is_parseable_decimal(s, 2, false, 0, 32)
// }
// fn is_parseable_ipv4_octet(s: &str) -> bool {
//     is_parseable_decimal(s, 3, true, 0, 255)
// }
// fn is_parseable_ipv6_mask(s: &str) -> bool {
//     is_parseable_decimal(s, 3, false, 1, 128)
// }
// fn is_parseable_ipv6_segment(s: &str) -> bool {
//     is_parseable_hexadecimal(s, 4, true, 0x0, 0xFFFF)
// }
// fn is_parseable_decimal(s: &str, max_length: usize, allow_leading_zero: bool,
//                         min_value: u64, max_value: u64) -> bool {
//     let length = s.chars().count();
//     if length == 0 || length > max_length {
//         return false;
//     }
//
//     let check_leading_zero = length > 1 && !allow_leading_zero;
//     if check_leading_zero && s.starts_with('0') {
//         return false;
//     }
//
//     let mut value = 0;
//     let mut chars = s.chars();
//     while let Some(c) = chars.next() {
//         if let Some(d) = get_digit_decimal(c) {
//             value *= 10;
//             value += d as u64;
//         } else {
//             return false;
//         }
//     }
//     value >= min_value && value <= max_value
// }
// fn is_parseable_hexadecimal(s: &str, max_length: usize, allow_leading_zero: bool,
//                             min_value: u64, max_value: u64) -> bool {
//     let length = s.chars().count();
//     if length == 0 || length > max_length {
//         return false;
//     }
//
//     let check_leading_zero = length > 1 && !allow_leading_zero;
//     if check_leading_zero && s.starts_with('0') {
//         return false;
//     }
//
//     let mut value = 0;
//     let mut chars = s.chars();
//     while let Some(c) = chars.next() {
//         if let Some(d) = get_digit_hexadecimal(c) {
//             value *= 16;
//             value += d as u64;
//         } else {
//             return false;
//         }
//     }
//     value >= min_value && value <= max_value
// }
// fn get_digit_decimal(c: char) -> Option<u8> {
//     if c.is_ascii_digit() {
//         Some(c as u8 - b'0')
//     } else {
//         None
//     }
// }
// fn get_digit_hexadecimal(c: char) -> Option<u8> {
//     if c.is_ascii_digit() {
//         Some(c as u8 - b'0')
//     } else if c.is_ascii_hexdigit() {
//         Some(c.to_ascii_lowercase() as u8 - b'a' + 10)
//     } else {
//         None
//     }
// }
//
// #[cfg(test)]
// mod tests {
//     use super::*;
//     #[test]
//     fn test_ipv4() {
//         assert!(is_valid("0.0.0.0"));
//         assert!(is_valid("255.255.255.255"));
//         assert!(is_valid("192.168.0.0"));
//         assert!(!is_valid("0.0.0.0.1"));
//         assert!(!is_valid("256.255.255.255"));
//         assert!(!is_valid("1"));
//         assert!(!is_valid("A.B.C"));
//         assert!(!is_valid("1:.4.6.5"));
//     }
//     #[test]
//     fn test_ipv6() {
//         assert!(is_valid("0:0:0:0:0:0:0:0"));
//         // assert!(is_valid("255.255.255.255"));
//         // assert!(is_valid("192.168.0.0"));
//         // assert!(!is_valid("0.0.0.0.1"));
//         // assert!(!is_valid("256.255.255.255"));
//         // assert!(!is_valid("1"));
//         // assert!(!is_valid("A.B.C"));
//         // assert!(!is_valid("1:.4.6.5"));
//     }
// }