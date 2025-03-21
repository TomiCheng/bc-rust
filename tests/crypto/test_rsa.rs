use std::sync::LazyLock;

use bc_rust::crypto::parameters;
use bc_rust::math::BigInteger;
use bc_rust::crypto::engines::RsaCoreEngine;

static MOD: LazyLock<BigInteger> = LazyLock::new(|| {
    BigInteger::with_string_radix("b259d2d6e627a768c94be36164c2d9fc79d97aab9253140e5bf17751197731d6f7540d2509e7b9ffee0a70a6e26d56e92d2edd7f85aba85600b69089f35f6bdbf3c298e05842535d9f064e6b0391cb7d306e0a2d20c4dfb4e7b49a9640bdea26c10ad69c3f05007ce2513cee44cfe01998e62b6c3637d3fc0391079b26ee36d5", 16).unwrap()
});
static PUB_EXP: LazyLock<BigInteger> =
    LazyLock::new(|| BigInteger::with_string_radix("11", 16).unwrap());
static PRIV_EXP: LazyLock<BigInteger> = LazyLock::new(|| {
    BigInteger::with_string_radix("92e08f83cc9920746989ca5034dcb384a094fb9c5a6288fcc4304424ab8f56388f72652d8fafc65a4b9020896f2cde297080f2a540e7b7ce5af0b3446e1258d1dd7f245cf54124b4c6e17da21b90a0ebd22605e6f45c9f136d7a13eaac1c0f7487de8bd6d924972408ebb58af71e76fd7b012a8d0e165f3ae2e5077a8648e619", 16).unwrap()
});
static P: LazyLock<BigInteger> = LazyLock::new(|| {
    BigInteger::with_string_radix("f75e80839b9b9379f1cf1128f321639757dba514642c206bbbd99f9a4846208b3e93fbbe5e0527cc59b1d4b929d9555853004c7c8b30ee6a213c3d1bb7415d03", 16).unwrap()
});
static Q: LazyLock<BigInteger> = LazyLock::new(|| {
    BigInteger::with_string_radix("b892d9ebdbfc37e397256dd8a5d3123534d1f03726284743ddc6be3a709edb696fc40c7d902ed804c6eee730eee3d5b20bf6bd8d87a296813c87d3b3cc9d7947", 16).unwrap()
});
static P_EXP: LazyLock<BigInteger> = LazyLock::new(|| {
    BigInteger::with_string_radix("1d1a2d3ca8e52068b3094d501c9a842fec37f54db16e9a67070a8b3f53cc03d4257ad252a1a640eadd603724d7bf3737914b544ae332eedf4f34436cac25ceb5", 16).unwrap()
});
static Q_EXP: LazyLock<BigInteger> = LazyLock::new(|| {
    BigInteger::with_string_radix("6c929e4e81672fef49d9c825163fec97c4b7ba7acb26c0824638ac22605d7201c94625770984f78a56e6e25904fe7db407099cad9b14588841b94f5ab498dded", 16).unwrap()
});
static CRT_COEF: LazyLock<BigInteger> = LazyLock::new(|| {
    BigInteger::with_string_radix("dae7651ee69ad1d081ec5e7188ae126f6004ff39556bde90e0b870962fa7b926d070686d8244fe5a9aa709a95686a104614834b0ada4b10f53197a5cb4c97339", 16).unwrap()
});
#[test]
fn test_paramters() {
    let public_key =
        parameters::RsaKeyParameters::public_key(MOD.clone(), PUB_EXP.clone()).unwrap();
    let private_key = parameters::RsaKeyParameters::private_key(
        MOD.clone(),
        PUB_EXP.clone(),
        PRIV_EXP.clone(),
        P.clone(),
        Q.clone(),
        P_EXP.clone(),
        Q_EXP.clone(),
        CRT_COEF.clone(),
    )
    .unwrap();

   let eng = RsaCoreEngine::default();
   //eng.init(true, public_key);
}
