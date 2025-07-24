

#[cfg(test)]
mod tests {
    use crate::crypto::ec::custom_named_curves;

    #[test]
    fn test_sum_of_multiplies() {
        let x9 = custom_named_curves::get_by_name("secp256r1");
        //assert!(x9.is_some())
    }
}