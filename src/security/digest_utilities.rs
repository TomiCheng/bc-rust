use crate::Result;
use crate::crypto::Digest;

pub fn do_final(digest: &mut dyn Digest, input: &[u8]) -> Result<Vec<u8>> {
    digest.block_update(input)?;
    let mut res_buf = vec![0u8; digest.get_digest_size()];
    digest.do_final(&mut res_buf)?;
    Ok(res_buf)
}
