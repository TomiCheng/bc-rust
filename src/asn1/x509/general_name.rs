use crate::asn1::asn1_utilities::try_from_choice_tagged;
use crate::asn1::try_from_tagged::{TryFromTagged, TryIntoTagged};
use crate::asn1::x509::{EdiPartyName, X509Name};
use crate::asn1::{
    Asn1Ia5String, Asn1Object, Asn1ObjectIdentifier, Asn1OctetString, Asn1Sequence, Asn1String,
    Asn1TaggedObject,
};
use crate::util::net::ip_address::{
    is_valid_ipv4, is_valid_ipv4_with_net_mask, is_valid_ipv6, is_valid_ipv6_with_net_mask,
};
use crate::{BcError, Result};
use std::fmt;
use std::str::FromStr;

/// The GeneralName object.
/// ```text
/// GeneralName ::= CHOICE {
///     otherName [0] OtherName,
///     rfc822Name [1] IA5String,
///     dNSName [2] IA5String,
///     x400Address [3] ORAddress,
///     directoryName [4] Name,
///     ediPartyName [5] EDIPartyName,
///     uniformResourceIdentifier [6] IA5String,
///     ipAddress [7] OCTET STRING,
///     registeredID [8] OBJECT IDENTIFIER
/// }
///
/// OtherName ::= SEQUENCE {
///     type-id OBJECT IDENTIFIER,
///     value [0] EXPLICIT ANY DEFINED BY type-id
/// }
///
/// EDIPartyName ::= SEQUENCE {
///    nameAssigner [0] DirectoryString OPTIONAL,
///    partyName [1] DirectoryString
/// }
/// ```
pub enum GeneralName {
    OtherName(Asn1Sequence),
    Rfc822Name(Asn1Ia5String),
    DnsName(Asn1Ia5String),
    X400Address(Asn1Sequence),
    DirectoryName(X509Name),
    EdiPartyName(EdiPartyName),
    UniformResourceIdentifier(Asn1Ia5String),
    IpAddress(Asn1OctetString),
    RegisteredId(Asn1ObjectIdentifier),
}

impl GeneralName {
    pub const OTHER_NAME: u8 = 0;
    pub const RFC822_NAME: u8 = 1;
    pub const DNS_NAME: u8 = 2;
    pub const X400_ADDRESS: u8 = 3;
    pub const DIRECTORY_NAME: u8 = 4;
    pub const EDI_PARTY_NAME: u8 = 5;
    pub const UNIFORM_RESOURCE_IDENTIFIER: u8 = 6;
    pub const IP_ADDRESS: u8 = 7;
    pub const REGISTERED_ID: u8 = 8;

    pub fn with_directory_name(name: X509Name) -> Self {
        GeneralName::DirectoryName(name)
    }
    pub fn with_ip(ip: &str) -> Result<Self> {
        let encoding = parse_ip_address(ip)?;
        Ok(GeneralName::IpAddress(Asn1OctetString::new(encoding)))
    }
    pub fn with_uniform_resource_identifier(uri: &str) -> Result<Self> {
        Asn1Ia5String::with_str(uri).map(GeneralName::UniformResourceIdentifier)
    }
    pub fn with_registered_id(object_id: &str) -> Result<Self> {
        Asn1ObjectIdentifier::with_str(object_id).map(GeneralName::RegisteredId)
    }

    fn try_from_base_object(tagged_object: Asn1TaggedObject) -> Result<Self> {
        if tagged_object.has_context_tag() {
            match tagged_object.tag_no() {
                Self::OTHER_NAME => Ok(GeneralName::OtherName(
                    tagged_object.try_into_tagged(false)?,
                )),
                Self::RFC822_NAME => Ok(GeneralName::Rfc822Name(
                    tagged_object.try_into_tagged(false)?,
                )),
                Self::DNS_NAME => Ok(GeneralName::DnsName(tagged_object.try_into_tagged(false)?)),
                Self::X400_ADDRESS => Ok(GeneralName::X400Address(
                    tagged_object.try_into_tagged(false)?,
                )),
                Self::DIRECTORY_NAME => Ok(GeneralName::DirectoryName(
                    tagged_object.try_into_tagged(false)?,
                )),
                Self::EDI_PARTY_NAME => {
                    let sequence = tagged_object.try_into_tagged(false)?;
                    Ok(GeneralName::EdiPartyName(EdiPartyName::from_sequence(
                        sequence,
                    )?))
                }
                Self::UNIFORM_RESOURCE_IDENTIFIER => Ok(GeneralName::UniformResourceIdentifier(
                    tagged_object.try_into_tagged(false)?,
                )),
                Self::IP_ADDRESS => Ok(GeneralName::IpAddress(
                    tagged_object.try_into_tagged(false)?,
                )),
                Self::REGISTERED_ID => Ok(GeneralName::RegisteredId(
                    tagged_object.try_into_tagged(false)?,
                )),
                _ => Err(BcError::with_invalid_argument(format!(
                    "Expected a context tag for GeneralName tag_no: {}",
                    tagged_object.tag_no()
                ))),
            }
        } else {
            Err(BcError::with_invalid_argument(
                "Expected a context tag for GeneralName",
            ))
        }
    }
    fn to_tag(&self) -> u8 {
        match self {
            GeneralName::OtherName(_) => Self::OTHER_NAME,
            GeneralName::Rfc822Name(_) => Self::RFC822_NAME,
            GeneralName::DnsName(_) => Self::DNS_NAME,
            GeneralName::X400Address(_) => Self::X400_ADDRESS,
            GeneralName::DirectoryName(_) => Self::DIRECTORY_NAME,
            GeneralName::EdiPartyName(_) => Self::EDI_PARTY_NAME,
            GeneralName::UniformResourceIdentifier(_) => Self::UNIFORM_RESOURCE_IDENTIFIER,
            GeneralName::IpAddress(_) => Self::IP_ADDRESS,
            GeneralName::RegisteredId(_) => Self::REGISTERED_ID,
        }
    }
}
impl From<GeneralName> for Asn1Object {
    fn from(value: GeneralName) -> Self {
        let (is_explicit, tag_no, object) = match value {
            GeneralName::OtherName(seq) => (false, GeneralName::OTHER_NAME, seq.into()),
            GeneralName::Rfc822Name(ia5) => (false, GeneralName::RFC822_NAME, ia5.into()),
            GeneralName::DnsName(ia5) => (false, GeneralName::DNS_NAME, ia5.into()),
            GeneralName::X400Address(seq) => (false, GeneralName::X400_ADDRESS, seq.into()),
            GeneralName::DirectoryName(name) => (true, GeneralName::DIRECTORY_NAME, name.into()),
            GeneralName::EdiPartyName(edi_party_name) => {
                (true, GeneralName::EDI_PARTY_NAME, edi_party_name.into())
            }
            GeneralName::UniformResourceIdentifier(ia5) => {
                (false, GeneralName::UNIFORM_RESOURCE_IDENTIFIER, ia5.into())
            }
            GeneralName::IpAddress(octet_string) => {
                (false, GeneralName::IP_ADDRESS, octet_string.into())
            }
            GeneralName::RegisteredId(object_id) => {
                (false, GeneralName::REGISTERED_ID, object_id.into())
            }
        };
        Asn1TaggedObject::from_explicit_tag_object(is_explicit, tag_no, object).into()
    }
}
impl TryFrom<Asn1Object> for GeneralName {
    type Error = BcError;

    fn try_from(value: Asn1Object) -> Result<Self> {
        if let Asn1Object::Tagged(tagged) = value {
            Self::try_from_base_object(tagged)
        } else {
            Err(BcError::with_invalid_argument(
                "Expected a tagged object for GeneralName",
            ))
        }
    }
}
impl TryFromTagged for GeneralName {
    fn try_from_tagged(tagged: Asn1TaggedObject, declared_explicit: bool) -> Result<Self>
    where
        Self: Sized,
    {
        try_from_choice_tagged(tagged, declared_explicit, GeneralName::try_from)
    }
}
impl fmt::Display for GeneralName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: ", self.to_tag())?;
        match self {
            GeneralName::Rfc822Name(name) => write!(
                f,
                "{}",
                name.to_asn1_string()
                    .unwrap_or_else(|_| "Invalid Rfc822Name".to_string())
            ),
            GeneralName::DnsName(name) => write!(
                f,
                "{}",
                name.to_asn1_string()
                    .unwrap_or_else(|_| "Invalid DNSName".to_string())
            ),
            GeneralName::DirectoryName(name) => write!(f, "{}", name),
            GeneralName::UniformResourceIdentifier(name) => write!(
                f,
                "{}",
                name.to_asn1_string()
                    .unwrap_or_else(|_| { "Invalid UniformResourceIdentifier".to_string() })
            ),
            _ => write!(f, "Invalid GeneralName type"),
        }
    }
}
fn parse_ip_address(ip: &str) -> Result<Vec<u8>> {
    if is_valid_ipv6_with_net_mask(ip) || is_valid_ipv6(ip) {
        if let Some(lash_index) = ip.find('/') {
            let mut address = vec![0u8; 32];
            let mut parsed_ip = parse_ipv6(&ip[..lash_index]);
            copy_u16_to_u8(&parsed_ip, &mut address);

            let mask = &ip[lash_index + 1..];
            if mask.find(':').is_some() {
                parsed_ip = parse_ipv6(mask);
            } else {
                parsed_ip = parse_ipv6_mask(mask);
            }
            copy_u16_to_u8(&parsed_ip, &mut address[16..]);
            Ok(address)
        } else {
            let mut address = vec![0u8; 16];
            let parsed_ip = parse_ipv6(ip);
            copy_u16_to_u8(&parsed_ip, &mut address);
            Ok(address)
        }
    } else if is_valid_ipv4_with_net_mask(ip) || is_valid_ipv4(ip) {
        if let Some(lash_index) = ip.find('/') {
            let mut address = vec![0u8; 8];
            parse_ipv4(&ip[..lash_index], &mut address);

            let mask = &ip[lash_index + 1..];
            if mask.find('.').is_some() {
                parse_ipv4(mask, &mut address[4..]);
            } else {
                parse_ipv4_mask(mask, &mut address[4..]);
            }
            Ok(address)
        } else {
            let mut address = vec![0u8; 4];
            parse_ipv4(ip, &mut address);
            Ok(address)
        }
    } else {
        Err(BcError::with_invalid_argument("Invalid IP address format"))
    }
}
fn parse_ipv6(ip: &str) -> Vec<u16> {
    let mut ip = ip;
    if ip.starts_with("::") {
        ip = &ip[1..];
    }

    if ip.ends_with("::") {
        ip = &ip[..ip.len() - 1];
    }

    let mut val = Vec::with_capacity(8);
    let mut double_colon = None;

    ip.split(':').for_each(|e| {
        if e.is_empty() {
            double_colon = Some(val.len());
            val.push(0);
        } else {
            if e.find('.').is_none() {
                val.push(u16::from_str_radix(e, 16).unwrap());
            } else {
                let tokens: Vec<&str> = e.split('.').collect();
                val.push(
                    u16::from_str(tokens[0]).unwrap() << 8 | u16::from_str(tokens[1]).unwrap(),
                );
                val.push(
                    u16::from_str(tokens[2]).unwrap() << 8 | u16::from_str(tokens[3]).unwrap(),
                );
            }
        }
    });

    if let Some(index) = double_colon {
        let missing = 8 - val.len();
        for _ in 0..missing {
            val.insert(index, 0);
        }
    }
    val
}
fn parse_ipv6_mask(mask: &str) -> Vec<u16> {
    let mut val = vec![0u16; 8];
    let mut bits = u16::from_str(mask).unwrap() as i16;
    let mut res_pos = 0;
    while bits >= 16 {
        val[res_pos] = u16::MAX;
        res_pos += 1;
        bits -= 16;
    }

    if bits > 0 {
        val[res_pos] = u16::MAX >> (16 - bits);
    }
    val
}
fn copy_u16_to_u8(source: &[u16], destination: &mut [u8]) {
    for (i, &value) in source.iter().enumerate() {
        let start = i * 2;
        destination[start + 0] = (value >> 8) as u8;
        destination[start + 1] = value as u8;
    }
}
fn parse_ipv4(ip: &str, destination: &mut [u8]) {
    let tokens: Vec<&str> = ip.split(['.', '/']).collect();
    for (i, token) in tokens.iter().enumerate() {
        destination[i] = u8::from_str(token).unwrap();
    }
}
fn parse_ipv4_mask(mask: &str, destination: &mut [u8]) {
    let mut bits = u8::from_str(mask).unwrap();
    let mut index = 0;
    while bits >= 8 {
        destination[index] = 0xFF;
        index += 1;
        bits -= 8;
    }
    if bits > 0 {
        destination[index] = (0xFF00 >> bits) as u8;
    }
}
#[cfg(test)]
mod tests {
    use crate::asn1::EncodingType::Ber;
    use crate::asn1::x509::GeneralName;
    use crate::asn1::{Asn1Encodable, Asn1Object};
    use crate::util::encoders::hex;

    #[test]
    fn test_ipv4() {
        check_ip_address_encoding("10.9.8.0", "87040a090800");
        check_ip_address_encoding("10.9.8.0/255.255.255.0", "87080a090800ffffff00");
        check_ip_address_encoding("10.9.8.0/24", "87080a090800ffffff00");
        check_ip_address_encoding("10.9.8.0/255.252.0.0", "87080a090800fffc0000");
        check_ip_address_encoding("10.9.8.0/14", "87080a090800fffc0000");
    }
    #[test]
    fn test_ipv6() {
        check_ip_address_encoding(
            "2001:0db8:85a3:08d3:1319:8a2e:0370:7334",
            "871020010db885a308d313198a2e03707334",
        );
        check_ip_address_encoding(
            "2001:0db8:85a3::1319:8a2e:0370:7334",
            "871020010db885a3000013198a2e03707334",
        );
        check_ip_address_encoding("::1", "871000000000000000000000000000000001");
        check_ip_address_encoding(
            "2001:0db8:85a3::8a2e:0370:7334",
            "871020010db885a3000000008a2e03707334",
        );
        check_ip_address_encoding(
            "2001:0db8:85a3::8a2e:10.9.8.0",
            "871020010db885a3000000008a2e0a090800",
        );
        check_ip_address_encoding(
            "2001:0db8:85a3::8a2e:10.9.8.0/ffff:ffff:ffff::0000",
            "872020010db885a3000000008a2e0a090800ffffffffffff00000000000000000000",
        );
        check_ip_address_encoding(
            "2001:0db8:85a3::8a2e:10.9.8.0/128",
            "872020010db885a3000000008a2e0a090800ffffffffffffffffffffffffffffffff",
        );
        check_ip_address_encoding(
            "2001:0db8:85a3::/48",
            "872020010db885a300000000000000000000ffffffffffff00000000000000000000",
        );
    }
    fn check_ip_address_encoding(ip: &str, hex_encoded: &str) {
        let buffer = hex::to_decode_with_str(hex_encoded).unwrap();
        let name: Asn1Object = GeneralName::with_ip(ip).unwrap().into();
        let buffer2 = name.get_encoded(Ber).unwrap();
        assert_eq!(buffer, buffer2);
    }
}
