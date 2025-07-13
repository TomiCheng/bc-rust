/// Validate the given IPv4 or IPv6 address.
///
/// # Arguments
/// * `address` - the IP address as a string.
///
/// # Returns
/// `true` if the address is valid, `false` otherwise.
pub fn is_valid(address: &str) -> bool {
    is_valid_ipv4(address) || is_valid_ipv6(address)
}
/// Validate the given IPv4 or IPv6 address and netmask.
///
/// # Arguments
/// * `address` - the IP address with netmask as a string.
///
/// # Returns
/// `true` if the address is valid with netmask, `false` otherwise.
pub fn is_valid_with_net_mask(address: &str) -> bool {
    is_valid_ipv4_with_net_mask(address) || is_valid_ipv6_with_net_mask(address)
}
pub fn is_valid_ipv4(address: &str) -> bool {
    let length = address.chars().count();
    if length < 7 || length > 15 {
        return false;
    }

    let mut index = 0;
    for _ in 0..3 {
        if let Some(end) = address[index..].find('.') {
            if !is_parseable_ipv4_octet(&address[index..index + end]) {
                return false;
            }
            index += end + 1;
        } else {
            return false;
        }
    }

    is_parseable_ipv4_octet(&address[index..])
}
pub fn is_valid_ipv6(address: &str) -> bool {
    if address.is_empty() {
        return false;
    }

    let c = address.chars().nth(0).unwrap();
    if c != ':' && !c.is_ascii_hexdigit() {
        return false;
    }

    let mut segment_count = 0;
    let temp = format!("{}:", address);
    let length = temp.chars().count();
    let mut double_colon_found = false;
    let mut pos = 0;
    while pos < length {
        if let Some(end) = temp[pos..].find(':') {
            if segment_count == 8 {
                return false; // Too many segments
            }

            if end == 0 {
                if pos != 1 && pos != (length - 1) && double_colon_found {
                    return false; // Multiple double colons not allowed
                }
                double_colon_found = true;
            } else {
                let value = &temp[pos..pos + end];

                if (pos + end) == length - 1 && value.find('.').is_some() {
                    // add an extra one as address covers 2 words.
                    if {
                        segment_count += 1;
                        segment_count
                    } == 8
                    {
                        return false; // Too many segments
                    }
                    if !is_valid_ipv4(value) {
                        return false; // Invalid IPv4 segment
                    }
                } else if !is_parseable_ipv6_segment(value) {
                    return false; // Invalid segment
                }
            }

            pos += end + 1; // Move past the colon
            segment_count += 1;
        } else {
            break;
        }
    }
    segment_count == 8 || double_colon_found
}
pub fn is_valid_ipv4_with_net_mask(address: &str) -> bool {
    let index = address.find('/');
    if let Some(index) = index {
        let before = &address[0..index];
        let after = &address[index + 1..];

        is_valid_ipv4(before) && (is_valid_ipv4(after) || is_parseable_ipv4_mask(after))
    } else {
        false
    }
}
pub fn is_valid_ipv6_with_net_mask(address: &str) -> bool {
    let index = address.find('/');
    if let Some(index) = index {
        let before = &address[0..index];
        let after = &address[index + 1..];

        is_valid_ipv6(before) && (is_valid_ipv6(after) || is_parseable_ipv6_mask(after))
    } else {
        false
    }
}
fn is_parseable_ipv4_octet(s: &str) -> bool {
    is_parseable_decimal(s, 3, true, 0, 255)
}
fn is_parseable_ipv6_mask(s: &str) -> bool {
    is_parseable_decimal(s, 3, false, 1, 128)
}
fn is_parseable_ipv4_mask(s: &str) -> bool {
    is_parseable_decimal(s, 2, false, 0, 32)
}
fn is_parseable_ipv6_segment(s: &str) -> bool {
    is_parseable_hexadecimal(s, 4, true, 0x0, 0xFFFF)
}
fn is_parseable_decimal(s: &str, max_length: usize, allow_leading_zero: bool, min_value: u64, max_value: u64) -> bool {
    if s.is_empty() {
        return false;
    }

    let length = s.chars().count();
    if length > max_length {
        return false;
    }

    let check_leading_zero = length > 1 && !allow_leading_zero;
    if check_leading_zero && s.starts_with('0') {
        return false;
    }

    let mut value = 0;
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if let Some(d) = get_digit_decimal(c) {
            value *= 10;
            value += d as u64;
        } else {
            return false;
        }
    }
    value >= min_value && value <= max_value
}
fn is_parseable_hexadecimal(s: &str, max_length: usize, allow_leading_zero: bool, min_value: u64, max_value: u64) -> bool {
    let length = s.chars().count();
    if length == 0 || length > max_length {
        return false;
    }

    let check_leading_zero = length > 1 && !allow_leading_zero;
    if check_leading_zero && s.starts_with('0') {
        return false;
    }

    let mut value = 0;
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if let Some(d) = get_digit_hexadecimal(c) {
            value *= 16;
            value += d as u64;
        } else {
            return false;
        }
    }
    value >= min_value && value <= max_value
}
fn get_digit_decimal(c: char) -> Option<u8> {
    if c.is_ascii_digit() { Some(c as u8 - b'0') } else { None }
}
fn get_digit_hexadecimal(c: char) -> Option<u8> {
    if c.is_ascii_digit() {
        Some(c as u8 - b'0')
    } else if c.is_ascii_hexdigit() {
        Some(c.to_ascii_lowercase() as u8 - b'a' + 10)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_ipv4() {
        assert!(is_valid("0.0.0.0"));
        assert!(is_valid("255.255.255.255"));
        assert!(is_valid("192.168.0.0"));
        assert!(!is_valid("0.0.0.0.1"));
        assert!(!is_valid("256.255.255.255"));
        assert!(!is_valid("1"));
        assert!(!is_valid("A.B.C"));
        assert!(!is_valid("1:.4.6.5"));
    }
    #[test]
    fn test_ipv6() {
        assert!(is_valid("0:0:0:0:0:0:0:0"));
        assert!(is_valid("FFFF:FFFF:FFFF:FFFF:FFFF:FFFF:FFFF:FFFF"));
        assert!(is_valid("0:1:2:3:FFFF:5:FFFF:1"));
        assert!(is_valid("fe80:0000:0000:0000:0202:b3ff:fe1e:8329"));
        assert!(is_valid("fe80:0:0:0:202:b3ff:fe1e:8329"));
        assert!(is_valid("2001:db8:85a3::8a2e:370:7334"));
        assert!(is_valid("::ffff:192.0.2.128"));
        assert!(!is_valid("0.0.0.0:1"));
        assert!(!is_valid("FFFF:FFFF:FFFF:FFFF:FFFF:FFFF:FFFF:FFFFF"));
    }
}
