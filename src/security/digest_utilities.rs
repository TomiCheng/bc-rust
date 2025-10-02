use crate::BcError;
use crate::crypto::Digest;

pub fn do_final(digest: &mut dyn Digest, input: &[u8]) -> Result<Vec<u8>, BcError> {
    digest.block_update(input)?;
    let mut res_buf = vec![0u8; digest.digest_size()];
    digest.do_final(&mut res_buf)?;
    Ok(res_buf)
}
