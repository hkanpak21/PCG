use crate::pcg_core::{PcgError, PcgExpander, PcgSeedGenerator};
use crate::primitives::dpf::{Dpf, DpfKey};
use crate::primitives::field::{Field128};
use crate::primitives::lpn::{
    LpnMatrix, LpnParameters, CodeType,
    matrix_vector_multiply_f2, matrix_vector_multiply_fq,
};
use ark_ff::Field as ArkField; // Keep alias for DPF trait bounds if needed
use ark_std::vec::Vec;
use ark_std::marker::PhantomData;
use rand::{Rng, thread_rng};
use nalgebra::DVector;
use std::ops::{Add, Mul};
use galois::Field; // For Field128 ops

// --- Seed Structures (Revised based on Fig 3 / Sec 4.3) ---

// Seed for P0 (sVOLE Sender role in standard VOLE)
#[derive(Clone)]
pub struct SvoleSenderSeed {
    k_dpf: DpfKey, // DPF Key K_0 for e0
    s: DVector<u8>, // Secret k-bit vector s
}

// Seed for P1 (sVOLE Receiver role in standard VOLE)
#[derive(Clone)]
pub struct SvoleReceiverSeed {
    k_dpf: DpfKey, // DPF Key K_1 for e1
    x: DVector<Field128>, // Secret k-vector x over Field128
}

// Enum to hold either seed type
#[derive(Clone)]
pub enum SvoleSeed {
    Sender(SvoleSenderSeed),
    Receiver(SvoleReceiverSeed),
}

// --- Output Structures (Revised) ---

// Output for P0 (sVOLE Sender)
#[derive(Debug, PartialEq)]
pub struct SvoleSenderOutput {
    pub u: DVector<u8>,     // Vector u = e0 (n bits)
    pub v: DVector<Field128>, // Vector v = Map(H*s, delta) + u*delta (n elements)
}

// Output for P1 (sVOLE Receiver)
#[derive(Debug, PartialEq)]
pub struct SvoleReceiverOutput {
    pub delta: Field128,      // Chosen delta value
    pub w: DVector<Field128>, // Vector w = H*x + Map(e1, delta) (n elements)
}

// Enum to hold either output type
#[derive(Debug, PartialEq)]
pub enum SvoleOutput {
    Sender(SvoleSenderOutput),
    Receiver(SvoleReceiverOutput),
}


/// sVOLE PCG struct implementing the core traits.
/// Note: The Field F for DPF is fixed to u8 (F2 output).
pub struct SvolePcg {
    lpn_params: LpnParameters,
    h_matrix: LpnMatrix, // Store H (n x k) - generate once
}

impl SvolePcg {
    /// Creates a new SvolePcg instance, generating the LPN matrix H.
    /// Assumes H is n x k.
    pub fn new(lpn_params: LpnParameters) -> Result<Self, &'static str> {
        // Generate H (n x k)
        // Current generator makes k x n, so we need to adjust or transpose.
        // Let's modify the LPN parameters temporarily for generation.
        let gen_params = LpnParameters {
            n: lpn_params.k, // Swap n and k for generator
            k: lpn_params.n,
            t: lpn_params.t,
            code_type: lpn_params.code_type, // Assume code type works with swapped dims
        };
        println!("Warning: Generating LPN matrix with swapped dimensions (k x n) due to generator mismatch.");
        let h_matrix_gen = gen_params.generate_matrix()?; // Generates k x n

        // TODO: Properly generate n x k matrix or transpose h_matrix_gen.
        // For now, using the k x n matrix and assuming it's n x k conceptually.
        let h_matrix = h_matrix_gen;
        if h_matrix.nrows() != lpn_params.n || h_matrix.ncols() != lpn_params.k {
             // Re-generating with correct dimensions for placeholder test
             println!("Regenerating placeholder matrix with correct n x k dimensions.");
             let mut rng = thread_rng();
             match lpn_params.code_type {
                  CodeType::RandomLinear => {
                     let h_dense = DMatrix::from_fn(lpn_params.n, lpn_params.k, |_,_| rng.gen_range(0..=1));
                     h_matrix = LpnMatrix::Dense(h_dense);
                  }
                   _ => return Err("Cannot regenerate non-random matrix with correct dims yet.")
             }
        }

        Ok(Self { lpn_params, h_matrix })
    }

    /// Helper to map F2 vector to Field128 using delta.
    /// output[i] = delta if input[i] == 1, else 0.
    fn map_f2_to_fq(&self, input: &DVector<u8>, delta: Field128) -> DVector<Field128> {
        DVector::from_iterator(
            input.nrows(),
            input.iter().map(|&bit| {
                if bit == 1 { delta } else { Field128::ZERO }
            })
        )
    }
}

// Implementation of the PCG Seed Generator trait for sVOLE
// F is now fixed to u8 (DPF output for F2)
impl PcgSeedGenerator for SvolePcg {
    type Seed0 = SvoleSenderSeed;
    type Seed1 = SvoleReceiverSeed;

    /// sVOLE_PCG::Gen (Fig 3 / Sec 4.3)
    /// Outputs SvoleSenderSeed and SvoleReceiverSeed.
    fn gen(
        _security_param: usize, // Lambda (e.g., 128) - used implicitly by DPF security
        lpn_params: &LpnParameters,
    ) -> Result<(Self::Seed0, Self::Seed1), PcgError> {

        let k = lpn_params.k;
        let n = lpn_params.n;

        // 1. P1 samples random k-vector x over Field128
        let mut rng = thread_rng();
        let x: DVector<Field128> = DVector::from_fn(k, |_, _| Field128::from(rng.gen::<u128>()));

        // 2. P0 samples random k-vector s over F_2
        let s: DVector<u8> = DVector::from_fn(k, |_, _| rng.gen_range(0..=1));

        // 3. Parties use DPF.Gen to generate keys for the t-sparse error vector `e`.
        //    This requires *secure* DPF generation (FR5), not local DPF gen.
        //    The non-interactive `gen` here should produce seeds compatible with the
        //    interactive protocol's output.
        //    For now, create placeholder DPF keys directly.
        //    We need the DPF output to be F2 (u8).
        println!("Warning: Using placeholder DPF keys in sVOLE Gen. Secure generation needed.");
        let domain_bits = (n as f64).log2().ceil() as usize;
        let dpf = Dpf::<u8>::new(domain_bits);
        // Choose a random alpha and beta=1 for the placeholder DPF
        let alpha = rng.gen_range(0..n);
        let (k_dpf0, k_dpf1) = dpf.gen(alpha, 1u8)
            .map_err(|e| PcgError::SeedGenError(format!("DPF gen failed: {}", e)))?;

        let seed0 = SvoleSenderSeed { k_dpf: k_dpf0, s };
        let seed1 = SvoleReceiverSeed { k_dpf: k_dpf1, x };

        Ok((seed0, seed1))
    }
}

// Implementation of the PCG Expander trait for sVOLE
impl PcgExpander for SvolePcg {
    type Seed = SvoleSeed; // Use the enum
    type Output = SvoleOutput; // Use the enum

    /// sVOLE_PCG::Expand (Fig 3 / Sec 4.3)
    /// Takes the combined seed enum and delta (if receiver).
    fn expand(
        &self,
        party_index: u8,
        seed: &Self::Seed,
        // Delta is only needed by the receiver (P1)
        delta: Option<Field128>,
    ) -> Result<Self::Output, PcgError> {

        let dpf = Dpf::<u8>::new(self.lpn_params.n.ilog2() as usize);

        match seed {
            SvoleSeed::Sender(sender_seed) => {
                if party_index != 0 { return Err(PcgError::ExpandError("Sender seed used by wrong party index".to_string())); }
                let delta_val = delta.ok_or_else(|| PcgError::ExpandError("Sender requires delta value for expand".to_string()))?;

                // P0 (Sender) computation
                // 1. Get e0 = Eval(0, k_dpf0)
                let e0_f2 = dpf.full_eval(&sender_seed.k_dpf)
                    .map_err(|e| PcgError::ExpandError(format!("DPF eval 0 failed: {}", e)))?;
                let u = DVector::from_vec(e0_f2);

                // 2. Compute v0 = H * s (over F2)
                let v0_f2 = matrix_vector_multiply_f2(&self.h_matrix, &sender_seed.s)?;

                // 3. Map v0 and u to Field128 using delta
                let mapped_v0 = self.map_f2_to_fq(&v0_f2, delta_val);
                let mapped_u = self.map_f2_to_fq(&u, delta_val);

                // 4. Compute v = mapped_v0 + mapped_u
                let v = mapped_v0.add(mapped_u);

                Ok(SvoleOutput::Sender(SvoleSenderOutput { u, v }))
            }
            SvoleSeed::Receiver(receiver_seed) => {
                if party_index != 1 { return Err(PcgError::ExpandError("Receiver seed used by wrong party index".to_string())); }
                let delta_val = delta.ok_or_else(|| PcgError::ExpandError("Receiver requires delta value for expand".to_string()))?;

                // P1 (Receiver) computation
                // 1. Get e1 = Eval(1, k_dpf1)
                 let e1_f2 = dpf.full_eval(&receiver_seed.k_dpf)
                    .map_err(|e| PcgError::ExpandError(format!("DPF eval 1 failed: {}", e)))?;
                let e1 = DVector::from_vec(e1_f2);

                // 2. Compute v1 = H * x (over Field128)
                let v1 = matrix_vector_multiply_fq(&self.h_matrix, &receiver_seed.x)?;

                // 3. Map e1 to Field128 using delta
                let mapped_e1 = self.map_f2_to_fq(&e1, delta_val);

                // 4. Compute w = v1 + mapped_e1
                let w = v1.add(mapped_e1);

                Ok(SvoleOutput::Receiver(SvoleReceiverOutput { delta: delta_val, w }))
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::lpn::CodeType;
    use ark_ff::Zero;
    use galois::PrimeField; // For Field128::random

    #[test]
    fn test_svole_gen_expand_correctness() {
        let n = 64; // Smaller dimensions for testing
        let k = 10;
        let t = 3; // Ignored by placeholder DPF gen

        let lpn_params = LpnParameters {
            n, k, t,
            code_type: CodeType::RandomLinear,
        };

        // Create SvolePcg instance (generates H)
        let svole_pcg = SvolePcg::new(lpn_params.clone()).expect("Failed to create SVole PCG");

        // Test Gen
        let gen_result = SvolePcg::gen(128, &lpn_params);
        assert!(gen_result.is_ok());
        let (seed0, seed1) = gen_result.unwrap();

        // Test Expand
        let mut rng = thread_rng();
        let delta = Field128::random(&mut rng);

        // Wrap seeds in enum
        let sender_seed = SvoleSeed::Sender(seed0);
        let receiver_seed = SvoleSeed::Receiver(seed1);

        let expand_res_p0 = svole_pcg.expand(0, &sender_seed, Some(delta));
        let expand_res_p1 = svole_pcg.expand(1, &receiver_seed, Some(delta));

        assert!(expand_res_p0.is_ok(), "P0 expand failed: {:?}", expand_res_p0.err());
        assert!(expand_res_p1.is_ok(), "P1 expand failed: {:?}", expand_res_p1.err());

        let output0 = expand_res_p0.unwrap();
        let output1 = expand_res_p1.unwrap();

        // Extract outputs
        let (u, v) = match output0 {
            SvoleOutput::Sender(out) => (out.u, out.v),
            _ => panic!("P0 expand returned wrong type"),
        };
        let w = match output1 {
            SvoleOutput::Receiver(out) => {
                assert_eq!(out.delta, delta);
                out.w
            }
            _ => panic!("P1 expand returned wrong type"),
        };

        // Check dimensions
        assert_eq!(u.nrows(), n);
        assert_eq!(v.nrows(), n);
        assert_eq!(w.nrows(), n);

        // Check the core VOLE relation: v[i] = u[i]*delta + w[i]
        let mut correctness_check_passed = true;
        for i in 0..n {
            let u_i = u[i];
            let v_i = v[i];
            let w_i = w[i];

            let u_i_delta = if u_i == 1 { delta } else { Field128::ZERO };
            let expected_v = u_i_delta.add(w_i);

            if v_i != expected_v {
                 eprintln!("Correctness check failed at index {}:", i);
                 eprintln!("  u = {}", u_i);
                 eprintln!("  v = {:?}", v_i);
                 eprintln!("  w = {:?}", w_i);
                 eprintln!("  delta = {:?}", delta);
                 eprintln!("  u*delta+w = {:?}", expected_v);
                 correctness_check_passed = false;
                 // break; // Stop on first error
            }
        }

        assert!(correctness_check_passed, "sVOLE correctness check v = u*delta + w failed");
        println!("sVOLE correctness check passed!");
    }
}
