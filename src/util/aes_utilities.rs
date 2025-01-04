use crate::crypto::BlockCipher;
use crate::crypto::engines::AesEngine;

pub fn create_engine() -> Box<dyn BlockCipher> {
   Box::new(AesEngine::new())
}

#[test]
fn test() {
    create_engine();
}