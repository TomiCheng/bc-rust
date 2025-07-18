use crate::{BcError, Result};
use rand::prelude::*;
use crate::math::raw::nat;
use crate::util::pack;

const M32UL: u32 = 0xFFFFFFFF;
const M30: i32 = 0x3FFFFFFF;
pub fn checked_mod_odd_inverse(m: &[u32], x: &[u32], z: &mut [u32]) -> Result<()> {
    if mod_odd_inverse(m, x, z) == 0 {
        return Err(BcError::with_arithmetic_error("Inverse does not exist."));
    }
    Ok(())
}
pub fn checked_mod_odd_inverse_var(m: &[u32], x: &[u32], z: &mut [u32]) -> Result<()> {
    if !mod_odd_inverse_var(m, x, z) {
        return Err(BcError::with_arithmetic_error("Inverse does not exist."));
    }
    Ok(())
}
pub(crate) fn inverse_u32(d: u32) -> u32 {
    debug_assert!(d & 1 == 1);

    //int x = d + (((d + 1) & 4) << 1);   // d.x == 1 mod 2**4
    let mut x = d; // d.x == 1 mod 2**3
    x = x.wrapping_mul(2u32.wrapping_sub(d.wrapping_mul(x))); // d.x == 1 mod 2**6
    x = x.wrapping_mul(2u32.wrapping_sub(d.wrapping_mul(x))); // d.x == 1 mod 2**12
    x = x.wrapping_mul(2u32.wrapping_sub(d.wrapping_mul(x))); // d.x == 1 mod 2**24
    x = x.wrapping_mul(2u32.wrapping_sub(d.wrapping_mul(x))); // d.x == 1 mod 2**48

    debug_assert!(d.wrapping_mul(x) == 1);
    x
}
pub(crate) fn inverse_u64(d: u64) -> u64 {
    debug_assert!(d & 1 == 1);

    //int x = d + (((d + 1) & 4) << 1);   // d.x == 1 mod 2**4
    let mut x = d; // d.x == 1 mod 2**3
    x = x.wrapping_mul(2u64.wrapping_sub(d.wrapping_mul(x))); // d.x == 1 mod 2**6
    x = x.wrapping_mul(2u64.wrapping_sub(d.wrapping_mul(x))); // d.x == 1 mod 2**12
    x = x.wrapping_mul(2u64.wrapping_sub(d.wrapping_mul(x))); // d.x == 1 mod 2**24
    x = x.wrapping_mul(2u64.wrapping_sub(d.wrapping_mul(x))); // d.x == 1 mod 2**48
    x = x.wrapping_mul(2u64.wrapping_sub(d.wrapping_mul(x))); // d.x == 1 mod 2**96

    debug_assert!(d.wrapping_mul(x) == 1);
    x
}
pub fn mod_odd_inverse(m: &[u32], x: &[u32], z: &mut [u32]) -> u32 {
    let len32 = m.len();

    debug_assert!(len32 > 0);
    debug_assert!((m[0] & 1) != 0);
    debug_assert!(m[len32 - 1] != 0);

    let bits = (len32 << 5) - m[len32 - 1].leading_zeros() as usize;
    let len30 = (bits + 29) / 30;

    let mut t = [0i32; 4];
    let mut d0 = vec![0i32; len30];
    let mut e0 = vec![0i32; len30];
    let mut f0 = vec![0i32; len30];
    let mut g0 = vec![0i32; len30];
    let mut m0 = vec![0i32; len32];

    e0[0] = 1;
    encode_30(bits, x, &mut g0);
    encode_30(bits, m, &mut m0);

    f0.copy_from_slice(&m0);

    // We use the "half delta" variant here, with theta == delta - 1/2
    let mut theta = 0;
    let inv_32 = inverse_u32(m0[0] as u32) as i32;
    let max_div_steps = get_maximum_half_division_steps(bits);
    for _div_steps in (0..max_div_steps).step_by(30) {
        theta = half_division_steps_30(theta, f0[0], g0[0], &mut t);
        update_de_30(len30, &mut d0, &mut e0, &mut t, inv_32, &mut m0);
        update_fg_30(len30, &mut f0, &mut g0, &mut t);
    }

    let sign_f = f0[len30 - 1] >> 31;
    conditional_negating_30(len30, sign_f, &mut f0);
    normalize_by_conditional_negating_30(len30, sign_f, &mut d0, &m0);
    decode_30(bits, &d0, z);
    debug_assert!(nat::less_than(m.len(), z, m) != 0);
    (equal_to(len30, &f0, 1) & equal_to(len30, &g0, 0)) as u32
}
pub fn mod_odd_inverse_var(m: &[u32], x: &[u32], z: &mut [u32]) -> bool {
    let len32 = m.len();

    debug_assert!(len32 > 0);
    debug_assert!((m[0] & 1) != 0);
    debug_assert!(m[len32 - 1] != 0);

    let bits = (len32 << 5) - m[len32 - 1].leading_zeros() as usize;
    let len30 = (bits + 29) / 30;
    let clz = bits - nat::get_bit_length(len32, x);

    let mut t = [0i32; 4];
    let mut d0 = vec![0i32; len30];
    let mut e0 = vec![0i32; len30];
    let mut f0 = vec![0i32; len30];
    let mut g0 = vec![0i32; len30];
    let mut m0 = vec![0i32; len32];

    e0[0] = 1;
    encode_30(bits, x, &mut g0);
    encode_30(bits, m, &mut m0);

    f0.copy_from_slice(&m0);

    // We use the original safe gcd here, with eta == 1 - delta
    // For shorter x, configure as if low zeros of x had been shifted away by div steps
    let mut eta = !clz as i32;
    let len_de = len30;
    let mut len_fg = len30;
    let inv_32 = inverse_u32(m0[0] as u32) as i32;
    let max_div_steps = get_maximum_division_steps(bits);

    let mut div_steps = clz;
    while !equal_to_var(len_fg, &g0, 0) {
        if div_steps >= max_div_steps {
            return false;
        }

        div_steps += 30;

        eta = division_30_var(eta, f0[0], g0[0], &mut t);
        update_de_30(len_de, &mut d0, &mut e0, &mut t, inv_32, &mut m0);
        update_fg_30(len_fg, &mut f0, &mut g0, &mut t);
        len_fg = trim_fg_30_var(len_fg, &mut f0, &mut g0);
    }

    let mut sign_f = f0[len_fg - 1] >> 31;
    /*
     * D is in the range (-2.M, M). First, conditionally add M if D is negative, to bring it
     * into the range (-M, M). Then normalize by conditionally negating (according to signF)
     * and/or then adding M, to bring it into the range [0, M).
     */
    let mut sign_d = d0[len_de - 1] >> 31;
    if sign_d < 0 {
        sign_d = add_30(len_de, &mut d0, &m0);
    }
    if sign_f < 0 {
        sign_d = negate_30(len_de, &mut d0);
        sign_f = negate_30(len_fg, &mut f0);
    }

    debug_assert!(sign_f == 0);

    if equal_to_var(len_fg, &f0, 1) {
        return false;
    }

    if sign_d < 0 {
        sign_d = add_30(len_de, &mut d0, &m0);
    }

    debug_assert!(sign_d == 0);

    decode_30(bits, &d0, z);
    debug_assert!(!nat::gte(len32, z, m));
    true
}
pub fn mod_odd_is_coprime(m: &[u32], x: &[u32]) -> u32 {
    let len32 = m.len();

    debug_assert!(len32 > 0);
    debug_assert!((m[0] & 1) != 0);
    debug_assert!(m[len32 - 1] != 0);

    let bits = (len32 << 5) - m[len32 - 1].leading_zeros() as usize;
    let len30 = (bits + 29) / 30;

    let mut t = [0i32; 4];
    let mut f0 = vec![0i32; len30];
    let mut g0 = vec![0i32; len30];
    let mut m0 = vec![0i32; len30];

    encode_30(bits, x, &mut g0);
    encode_30(bits, m, &mut m0);

    f0.copy_from_slice(&m0);

    // We use the "half delta" variant here, with theta == delta - 1/2
    let mut theta = 0;
    let max_div_steps = get_maximum_half_division_steps(bits);

    for _div_steps in (0..max_div_steps).step_by(30) {
        theta = half_division_steps_30(theta, f0[0], g0[0], &mut t);
        update_fg_30(len30, &mut f0, &mut g0, &mut t);
    }

    let sign_f = f0[len30 - 1] >> 31;
    conditional_negating_30(len30, sign_f, &mut f0);

    (equal_to(len30, &f0, 1) & equal_to(len30, &g0, 0)) as u32
}
pub fn mod_odd_is_coprime_var(m: &[u32], x: &[u32]) -> bool {
    let len32 = m.len();

    debug_assert!(len32 > 0);
    debug_assert!((m[0] & 1) != 0);
    debug_assert!(m[len32 - 1] != 0);

    let bits = (len32 << 5) - m[len32 - 1].leading_zeros() as usize;
    let len30 = (bits + 29) / 30;
    let clz = bits - nat::get_bit_length(len32, x);

    let mut t = [0i32; 4];
    let mut f0 = vec![0i32; len30];
    let mut g0 = vec![0i32; len30];
    let mut m0 = vec![0i32; len32];

    encode_30(bits, x, &mut g0);
    encode_30(bits, m, &mut m0);

    f0.copy_from_slice(&m0);

    // We use the original safe gcd here, with eta == 1 - delta
    // For shorter x, configure as if low zeros of x had been shifted away by div steps
    let mut eta = !clz as i32;
    let mut len_fg = len30;
    let max_div_steps = get_maximum_division_steps(bits);

    let mut div_steps = clz;
    while !equal_to_var(len_fg, &g0, 0) {
        if div_steps >= max_div_steps {
            return false;
        }

        div_steps += 30;

        eta = division_30_var(eta, f0[0], g0[0], &mut t);
        update_fg_30(len_fg, &mut f0, &mut g0, &mut t);
        len_fg = trim_fg_30_var(len_fg, &mut f0, &mut g0);
    }

    let mut sign_f = f0[len_fg - 1] >> 31;
    if sign_f < 0 {
        sign_f = negate_30(len_fg, &mut f0);
    }

    debug_assert!(sign_f == 0);
    equal_to_var(len_fg, &f0, 1)
}
pub(crate) fn random<TRngCore: RngCore>(random: &mut TRngCore, p: &[u32], z: &mut [u32]) -> Result<()> {
    let len = p.len();

    if z.len() < len {
        return Err(BcError::with_invalid_argument("insufficient space"));
    }

    let s = &mut z[..len];
    let mut m = p[len - 1];
    m |= m >> 1;
    m |= m >> 2;
    m |= m >> 4;
    m |= m >> 8;
    m |= m >> 16;

    let alloc_size = len * size_of::<u32>();
    let mut bytes = vec![0u8; alloc_size];

    loop {
        random.fill_bytes(&mut bytes);
        pack::be_to_u32_buffer(&bytes, s);
        s[len - 1] &= m;
        if !nat::gte(len, s, p) {
            break;
        }
    }
    Ok(())
}
fn add_30(len30: usize, d: &mut [i32], e: &[i32]) -> i32 {
    debug_assert!(len30 > 0);
    debug_assert!(d.len() >= len30);
    debug_assert!(e.len() >= len30);

    let mut c = 0;
    let last = len30 - 1;
    for i in 0..last {
        c += d[i] + e[i];
        d[i] = c & M30;
        c >>= 30;
    }

    c += d[last] + e[last];
    d[last] = c;
    c >>= 30;
    c
}
fn conditional_negating_30(len30: usize, cond: i32,  d: &mut [i32]) {
    debug_assert!(len30 > 0);
    debug_assert!(d.len() >= len30);

    let mut c = 0;
    let last = len30 - 1;
    for i in 0..last {
        c += (d[i] ^ cond) - cond;
        d[i] = c & M30;
        c >>= 30;
    }

    c += (d[last] ^ cond) - cond;
    d[last] = c;
}
fn normalize_by_conditional_negating_30(len30: usize, cond_negate: i32, d: &mut [i32], m: &[i32]) {
    debug_assert!(len30 > 0);
    debug_assert!(d.len() >= len30);
    debug_assert!(m.len() >= len30);
    // D is in the range (-2.M, M). First, conditionally add M if D is negative, to bring it
    // into the range (-M, M). Then normalize by conditional negating (according to signF)
    // and/or then adding M, to bring it into the range [0, M).

    let last = len30 - 1;
    {
        let mut c = 0;
        let cond_add = d[last] >> 31;
        for i in 0..last {
            let mut di = d[i] + (m[i] & cond_add);
            di = (di ^ cond_negate) - cond_negate;
            c += di;
            d[i] = c & M30;
            c >>= 30;
        }
        {
            let mut di = d[last] + (m[last] & cond_add);
            di = (di ^ cond_negate) - cond_negate;
            c += di;
            d[last] = c;
        }
    }
    {
        let mut c = 0;
        let cond_add = d[last] >> 31;
        for i in 0..last {
            let di = d[i] + (m[i] & cond_add);
            c += di;
            d[i] = c & M30;
            c >>= 30;
        }
        {
            let di = d[last] + (m[last] & cond_add);
            c += di;
            d[last] = c;
        }

        debug_assert!(c >> 30 == 0)
    }
}
fn decode_30(bits: usize, x: &[i32], z: &mut [u32]) {
    let mut bits = bits as isize;
    debug_assert!(bits > 0);

    let mut avail = 0;
    let mut data = 0;
    let mut x_offset = 0;
    let mut z_offset = 0;

    while bits > 0 {
        while avail < bits.min(30) {
            data |= (x[x_offset] as u64) << avail;
            x_offset += 1;
            avail += 30;
        }

        z[z_offset] = data as u32;
        z_offset += 1;
        data >>= 32;
        avail -= 32;
        bits -= 32;
    }
}
fn division_30_var(eta: i32, f0: i32, g0: i32, t: &mut[i32]) -> i32 {
    let mut eta = eta;
    let mut u = 1;
    let mut v = 0;
    let mut q = 0;
    let mut r = 1;
    let mut f = f0;
    let mut g = g0;
    let mut m;
    let mut w;
    let mut x;
    let mut y;
    let mut z;
    let mut i = 30;
    let mut limit;
    let mut zeros;

    loop {
        zeros = (g | (-1 << i)).leading_zeros() as i32;

        g >>= zeros;
        u <<= zeros;
        v <<= zeros;
        eta -= zeros as i32;
        i -= zeros;

        if i <= 0 {
            break;
        }

        debug_assert!((f & 1) == 1);
        debug_assert!((g & 1) == 1);
        debug_assert!((u * f0 + v * g0) == f << (30 - i));
        debug_assert!((q * f0 + r * g0) == g << (30 - i));

        if eta <= 0 {
            eta = 2 - eta;
            x = f;
            f = g;
            g = -x;
            y = u;
            u = v;
            q = -y;
            z = v;
            v = r;
            r = -z;

            // Handle up to 6 div steps at once, subject to eta and i.
            limit = if eta > i { i } else { eta };
            m = (i32::MAX >> (32 - limit)) as i32 & 63;
            w = (f * g * (f * f - 2)) & m;
        } else {
            // Handle up to 4 div steps at once, subject to eta and i.
            limit = if eta > i { i } else { eta };
            m = (i32::MAX >> (32 - limit)) as i32 & 15;
            w = f + (((f + 1) & 4) << 1);
            w = (w * -g) & m;
        }

        g += f * w;
        q += u * w;
        r += v * w;

        debug_assert!((g & m) == 0);
    }

    t[0] = u;
    t[1] = v;
    t[2] = q;
    t[3] = r;
    eta
}
fn encode_30(bits: usize, x: &[u32], z: &mut [i32]) {
    let mut bits = bits;
    debug_assert!(bits > 0);

    let mut avail = 0;
    let mut data = 0;
    let mut x_offset = 0;
    let mut z_offset = 0;

    while bits > 0 {
        if avail < bits.min(30) {
            data |= (x[x_offset] & M32UL) << avail;
            x_offset += 1;
            avail += 32;
        }

        z[z_offset] = data as i32 & M30;
        z_offset += 1;
        data >>= 30;
        avail -= 30;
        bits -= 30;
    }
}
fn equal_to(len: usize, x: &[i32], y: i32) -> i32 {
    let mut d = x[0] ^ y;
    for i in 1..len {
        d |= x[i];
    }

    d = (d >> 1) | (d & 1);
    (d - 1) >> 31
}
fn equal_to_var(len: usize, x: &[i32], y: i32) -> bool {
    let mut d = x[0] ^ y;
    if d != 0 {
        return false;
    }

    for i in 1..len {
        d |= x[i];
    }

    d == 0
}
fn get_maximum_division_steps(bits: usize) -> usize {
    (188898 * bits + (if bits < 46 { 308405 } else { 181188 })) >> 16
}
fn get_maximum_half_division_steps(bits: usize) -> usize {
    (150964 * bits + 99243) >> 16
}
fn half_division_steps_30(theta: i32, f0: i32, g0: i32, t: &mut[i32]) -> i32 {
    let mut theta = theta;
    let mut u = 1 << 30;
    let mut v = 0;
    let mut q = 0;
    let mut r = 1 << 30;
    let mut f = f0;
    let mut g = g0;

    for i in 0..30 {
        debug_assert!((f & 1) == 1);
        debug_assert!(((u >> (30 - i)) * f0 + (v >> (30 - i)) * g0) == f << i);
        debug_assert!(((q >> (30 - i)) * f0 + (r >> (30 - i)) * g0) == g << i);

        let c1 = theta >> 31;
        let c2 = -(g & 1);

        let x = f ^ c1;
        let y = u ^ c1;
        let z = v ^ c2;

        g -= x & c2;
        q -= y & c2;
        r -= z & c2;

        let c3 = c2 & !c1;
        theta = (theta ^ c3) + 1;

        f += g & c3;
        u += q & c3;
        v += r & c3;

        g >>= 1;
        q >>= 1;
        r >>= 1;
    }

    t[0] = u;
    t[1] = v;
    t[2] = q;
    t[3] = r;
    theta
}
fn negate_30(len30: usize, d: &mut [i32]) -> i32 {
    debug_assert!(len30 > 0);
    debug_assert!(d.len() >= len30);

    let mut c = 0;
    let last = len30 - 1;

    for i in 0..last {
        c -= d[i];
        d[i] = c & M30;
        c >>= 30;
    }

    c -= d[last];
    d[last] = c;
    c >>= 30;
    c
}
fn trim_fg_30_var(len30: usize, f: &mut [i32], g: &mut [i32]) -> usize {
    let mut len30 = len30;
    debug_assert!(len30 > 0);
    debug_assert!(f.len() >= len30);
    debug_assert!(g.len() >= len30);

    let fn1 = f[len30 - 1];
    let gn1 = g[len30 - 1];

    let mut cond = ((len30 - 2) >> 31) as i32;
    cond |= fn1 ^ (fn1 >> 31);
    cond |= gn1 ^ (gn1 >> 31);

    if cond == 0 {
        f[len30 - 2] |= fn1 << 30;
        g[len30 - 2] |= gn1 << 30;
        len30 -= 1;
    }
    len30
}
fn update_de_30(len30: usize, d: &mut [i32], e: &mut [i32], t: &[i32], inv_32: i32, m: &[i32]) {
    debug_assert!(len30 > 0);
    debug_assert!(d.len() >= len30);
    debug_assert!(e.len() >= len30);
    debug_assert!(m.len() >= len30);
    debug_assert!((inv_32 * m[0]) == 0);

    let u = t[0];
    let v = t[1];
    let q = t[2];
    let r = t[3];
    let mut di;
    let mut ei;
    let mut md;
    let mut me;
    let mut mi;
    let sd;
    let se;

    let mut cd;
    let mut ce;

    // We accept D (E) in the range (-2.M, M) and conceptually add the modulus to the input
    // value if it is initially negative. Instead of adding it explicitly, we add u and/or v (q
    // and/or r) to md (me).

    sd = d[len30 - 1] >> 31;
    se = e[len30 - 1] >> 31;

    md = (u & sd) + (v & se);
    me = (q & sd) + (r & se);

    mi = m[0];
    di = d[0];
    ei = e[0];

    cd = (u as i64 * di as i64) + (v as i64 * ei as i64);
    ce = (q as i64 * di as i64) + (r as i64 * ei as i64);

    // Subtract from md/me an extra term in the range [0, 2^30) such that the low 30 bits of the
    // intermediate D/E values will be 0, allowing clean division by 2^30. The final D/E are
    // thus in the range (-2.M, M), consistent with the input constraint.

    md -= (inv_32 * cd as i32 + md) & M30;
    me -= (inv_32 * ce as i32 + me) & M30;

    cd += mi as i64 * md as i64;
    ce += mi as i64 * me as i64;

    debug_assert!((cd as i32 & M30) == 0);
    debug_assert!((ce as i32 & M30) == 0);

    for i in 1..len30 {
        mi = m[i];
        di = d[i];
        ei = e[i];

        cd += (u as i64 * di as i64) + (v as i64 * ei as i64) + (mi as i64 * md as i64);
        ce += (q as i64 * di as i64) + (r as i64 * ei as i64) + (mi as i64 * me as i64);

        d[i - 1] = cd as i32 & M30;
        cd >>= 30;
        e[i - 1] = ce as i32 & M30;
        ce >>= 30;
    }

    d[len30 - 1] = cd as i32;
    e[len30 - 1] = ce as i32;
}
fn update_fg_30(len30: usize, f: &mut [i32], g: &mut [i32], t: &[i32]) {
    debug_assert!(len30 > 0);
    debug_assert!(f.len() >= len30);
    debug_assert!(g.len() >= len30);

    let u = t[0];
    let v = t[1];
    let q = t[2];
    let r = t[3];
    let mut fi;
    let mut gi;
    let mut cf;
    let mut cg;

    fi = f[0];
    gi = g[0];

    cf = (u as u64 * fi as u64) + (v as u64 * gi as u64);
    cg = (q as u64 * fi as u64) + (r as u64 * gi as u64);

    cf >>= 30;
    cg >>= 30;

    for i in 1..len30 {
        fi = f[i];
        gi = g[i];

        cf += (u as u64 * fi as u64) + (v as u64 * gi as u64);
        cg += (q as u64 * fi as u64) + (r as u64 * gi as u64);

        f[i - 1] = cf as i32 & M30;
        cf >>= 30;

        g[i - 1] = cg as i32 & M30;
        cg >>= 30;
    }

    f[len30 - 1] = cf as i32;
    g[len30 - 1] = cg as i32;
}


