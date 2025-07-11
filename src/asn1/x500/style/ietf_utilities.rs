use crate::Result;
use crate::asn1::EncodingType::Der;
use crate::asn1::{Asn1Encodable, Asn1Object};
use crate::util::encoders::hex::to_hex_string;

pub fn asn1_object_to_string(value: &Asn1Object) -> Result<String> {
    let mut str = String::new();
    if let Some(v) = value.as_string()
        && !matches!(value, Asn1Object::UniversalString(_))
    {
        let s = v.to_asn1_string()?;
        if s.chars().count() > 0 && s.chars().nth(0) == Some('#') {
            str.push('\\');
        }
        str.push_str(&s);
    } else {
        str.push('#');
        str.push_str(&to_hex_string(value.get_encoded(Der)?.as_slice()));
    }

    let buffer = escape_dn_string(&str);
    Ok(buffer)
}
pub(crate) fn escape_dn_string(str: &str) -> String {
    let count = str.chars().count();
    let f_space = str.chars().take_while(|c| *c == ' ').count();
    let r_space = count - str.chars().rev().take_while(|c| *c == ' ').count();
    let mut buffer = String::new();
    let mut chars = str.char_indices();
    let mut c1 = false;
    let mut c2 = false;
    while let Some((i, c)) = chars.next() {
        if i == 0 && c == '\\' {
            c1 = true;
            continue;
        }
        if i == 1 && c == '#' {
            c2 = true;
            continue;
        }

        if c1 && c2 {
            buffer.push_str("\\#");
            c1 = false;
            c2 = false;
        }

        // if c1 == Some('\\') && c2 == Some('#') {
        //     buffer.push(c1.unwrap());
        //     buffer.push(c2.unwrap());
        //     continue;
        // }

        if i < f_space && c == ' ' {
            buffer.push('\\');
            buffer.push(c);
            continue;
        }

        if i >= r_space && c == ' ' {
            buffer.push('\\');
            buffer.push(c);
            continue;
        }

        if c == ',' || c == '"' || c == '\\' || c == '+' || c == '=' || c == '<' || c == '>' || c == ';' {
            buffer.push('\\');
            buffer.push(c);
            continue;
        }

        buffer.push(c);
    }
    buffer
}
pub(crate) fn unescape(elt: &str) -> String {
    if elt.is_empty() {
        return elt.to_string();
    }

    if elt.find('\\').is_none() && elt.find('"').is_none() {
        return elt.trim().to_string();
    }

    let mut escaped = false;
    let mut quoted = false;
    let mut buffer = String::new();
    let mut start = 0;

    let mut chars = elt.chars();

    // if it's an escaped hash string and not an actual encoding in string form,
    // we need to leave it escaped.
    if chars.next() == Some('\\') {
        if chars.next() == Some('#') {
            start = 2;
            buffer.push_str("\\#");
        }
    }

    let mut non_white_space_encountered = false;
    let mut last_escaped = 0;
    let mut hex1 = '\0';

    let mut chars = elt[start..].chars();
    while let Some(c) = chars.next() {
        if c != ' ' {
            non_white_space_encountered = true;
        }

        if c == '"' {
            if !escaped {
                quoted = !quoted;
            } else {
                buffer.push(c);
                escaped = false;
            }
        } else if c == '\\' && !(escaped || quoted) {
            escaped = true;
            last_escaped = buffer.chars().count();
        } else {
            if c == ' ' && !escaped && !non_white_space_encountered {
                continue;
            }
            if escaped && is_hex_digit(c) {
                if hex1 != '\0' {
                    let v = (convert_hex(hex1) * 16 + convert_hex(c)) as char;
                    buffer.push(v);
                    escaped = false;
                    hex1 = '\0';
                    continue;
                }
                hex1 = c;
                continue;
            }
            buffer.push(c);
            escaped = false;
        }
    }

    if buffer.len() > 0 {
        let index = buffer.chars().count() - 1;
        if buffer.chars().nth(index) == Some(' ') && last_escaped != index {
            buffer.remove(index);
        }
    }
    buffer
}
fn is_hex_digit(c: char) -> bool {
    c.is_ascii_hexdigit()
}
fn convert_hex(c: char) -> u8 {
    if '0' <= c && c <= '9' {
        return (c as u8) - b'0';
    }

    if 'A' <= c && c <= 'F' {
        return (c as u8) - b'A' + 10;
    }

    if 'a' <= c && c <= 'f' {
        return (c as u8) - b'a' + 10;
    }

    0
}
