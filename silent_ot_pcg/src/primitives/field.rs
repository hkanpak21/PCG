use ark_ff::{Fp128, Fp64, MontBackend, MontConfig, Field, PrimeField};
use ark_ff::fields::models::fp64::Fp64Parameters;
use ark_ff::fields::models::fp128::Fp128Parameters;

// Use the galois crate for F_{2^128}
use galois::GF2e128;

// Define F_2 (useful for LPN matrix operations if not using bool/u8)
#[derive(MontConfig)]
#[modulus = "2"]
#[generator = "1"]
pub struct F2Config;
pub type F2 = Fp64<MontBackend<F2Config, 1>>;

// Define F_{2^64} (example binary extension field)
// Not directly used in standard Silent OT (which uses F_{2^128}), but good example.
// ark-ff primarily uses prime fields, but can represent F_{2^n} elements as polynomials over F2.
// However, ark-ff's direct support for optimal F_{2^n} arithmetic (like using PCLMULQDQ) might be limited.
// For high performance F_{2^128}, specialized crates might be better (e.g., `gf256` is F_{2^8}, `galois` crate exists but check status).

// Let's define F_{2^128} using the standard library approach for demonstration.
// We can represent elements as u128 and implement field arithmetic manually or use a dedicated crate.
// Using ark-ff's Fp128 is NOT F_{2^128}. It's a prime field F_p where p fits in 128 bits.

// Define F_{2^128} using the galois crate.
// This uses the standard polynomial x^128 + x^7 + x^2 + x + 1.
pub type Field128 = GF2e128;

// The galois crate implements standard traits like Add, Mul, etc.
// No need for placeholder functions add_f2_128, mul_f2_128.

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::UniformRand as ArkUniformRand; // Use full path to avoid clash
    use rand::{Rng, thread_rng};
    use std::ops::{Add, Mul, Sub, Div};
    use galois::Field;

    #[test]
    fn test_f2_arithmetic() {
        let zero = F2::from(0u64);
        let one = F2::from(1u64);

        assert_eq!(zero + zero, zero);
        assert_eq!(zero + one, one);
        assert_eq!(one + zero, one);
        assert_eq!(one + one, zero); // 1 + 1 = 0 in F2

        assert_eq!(zero * zero, zero);
        assert_eq!(zero * one, zero);
        assert_eq!(one * zero, zero);
        assert_eq!(one * one, one);
    }

    #[test]
    fn test_field128_arithmetic() {
        let mut rng = thread_rng();
        // Generate random elements using From<u128>
        let a = Field128::from(rng.gen::<u128>());
        let b = Field128::from(rng.gen::<u128>());
        let c = Field128::from(rng.gen::<u128>());
        let zero = Field128::ZERO;
        let one = Field128::ONE;

        // Test Addition (XOR)
        assert_eq!(a.add(zero), a);
        assert_eq!(a.add(a), zero); // a + a = 0 in GF(2^n)
        assert_eq!(a.add(b).add(c), a.add(b.add(c))); // Associativity
        assert_eq!(a.add(b), b.add(a)); // Commutativity

        // Test Multiplication
        assert_eq!(a.mul(zero), zero);
        assert_eq!(a.mul(one), a);
        assert_eq!(a.mul(b), b.mul(a)); // Commutativity
        assert_eq!(a.mul(b).mul(c), a.mul(b.mul(c))); // Associativity

        // Test Distributivity: a * (b + c) = a * b + a * c
        assert_eq!(a.mul(b.add(c)), a.mul(b).add(a.mul(c)));

        // Test Inverse (if not zero)
        if a != zero {
            let a_inv = a.inverse().unwrap();
            assert_eq!(a.mul(a_inv), one);
        }
         if b != zero {
            let b_inv = b.inverse().unwrap();
            assert_eq!(b.mul(b_inv), one);
        }

        println!("Field128 tests passed.");
    }
}
