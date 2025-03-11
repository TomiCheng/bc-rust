// use std::any::Any;

// use crate::math;
// use crate::{
//     crypto::{
//         parameters::{self, RsaKeyParameters},
//         CipherParameters,
//     },  Error
// };
// use crate::{ErrorKind, Result};
// /// This does your basic RSA algorithm.
pub struct RsaCoreEngine {
//     //key: Option<Box<dyn parameters::RsaKeyParameters>>,
//     for_encryption: bool,
//     bit_size: usize,
}

// impl RsaCoreEngine {
//     /// initialise the RSA engine.
//     /// # Arguments
//     /// * `for_encryption` - true if we are encrypting, false if we are decrypting.
//     /// * `parameters` - the necessary RSA key parameters.
//     pub fn init(&mut self, for_encryption: bool, parameters: Box<dyn parameters::RsaKeyParameters>) {
//         self.bit_size = *parameters.modulus().bit_length();
//         self.key = Some(parameters);
//         self.for_encryption = for_encryption;
//     }

//     fn check_initialized(&self) -> Result<()> {
//         if self.key.is_none() {
//             return Err(Error::with_message(
//                 ErrorKind::InvalidOperation,
//                 "RSA engine not initialised".to_owned(),
//             ));
//         }
//         Ok(())
//     }

//     pub fn get_output_block_size(&self) -> Result<usize> {
//         self.check_initialized()?;
//         return Ok(if self.for_encryption {
//             (self.bit_size + 7) / 8
//         } else {
//             (self.bit_size - 1) / 8
//         });
//     }

//     pub fn process_block(&self, input: math::BigInteger) -> Result<math::BigInteger> {
//         self.check_initialized()?;

//         //let key = self.key.as_ref().unwrap();
//         //let dd = Into::<dyn Any>::into(key);


//         todo!()
//     }
// }

impl Default for RsaCoreEngine {
    fn default() -> Self {
        RsaCoreEngine {
            // key: None,
            // for_encryption: false,
            // bit_size: 0,
        }
    }
}
