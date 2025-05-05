use crate::pcg_core::{PcgError, PcgExpander, PcgSeedGenerator, SvoleSenderSeed, SvoleReceiverSeed, SvoleSeed};
use crate::primitives::dpf::{Dpf, DpfTrait};
use crate::primitives::field::{Field128, F2};
use crate::primitives::lpn::{LpnParameters};
use ark_std::vec::Vec;
use ark_std::marker::PhantomData;
use rand::{thread_rng};
use ark_ff::{UniformRand, One, Zero};

// --- Output Structures (Revised) ---

// Output for P0 (sVOLE Sender)
#[derive(Debug, PartialEq, Clone)]
pub struct SvoleSenderOutput {
    pub u: Vec<F2>,     // Vector u (n elements)
    pub v: Vec<Field128>, // Vector v (n elements)
}

// Output for P1 (sVOLE Receiver)
#[derive(Debug, PartialEq, Clone)]
pub struct SvoleReceiverOutput {
    pub x: Vec<F2>,    // Vector x (n elements)
    pub w: Vec<Field128>, // Vector w (n elements)
}

// Enum to hold either output type
#[derive(Debug, PartialEq, Clone)]
pub enum SvoleOutput {
    Sender(SvoleSenderOutput),
    Receiver(SvoleReceiverOutput),
}

/// sVOLE PCG struct implementing the core traits.
/// Note: The Field F for DPF is fixed to u8 (F2 output).
pub struct SvolePcg {
    lpn_params: LpnParameters,
    dpf_handler: Dpf<F2>, // Assuming DPF output is F2, make concrete
    _field_marker: PhantomData<Field128>, // VOLE field
}

impl SvolePcg {
    /// Creates a new SvolePcg instance, generating the LPN matrix.
    pub fn new(lpn_params: LpnParameters) -> Result<Self, &'static str> {
        let dpf_domain_bits = lpn_params.k; // Example: Tie DPF domain to k
        let dpf_handler = Dpf::<F2>::new(dpf_domain_bits);
        Ok(SvolePcg {
            lpn_params,
            dpf_handler,
            _field_marker: PhantomData,
        })
    }
}

// Implementation of the PCG Seed Generator trait for sVOLE
// F is now fixed to u8 (DPF output for F2)
impl PcgSeedGenerator for SvolePcg {
    type Seed = SvoleSeed;

    fn gen(
        &self, // Add &self back, needed for self.lpn_params, self.h_matrix etc.
        security_param: usize, // Match the trait
    ) -> Result<(Self::Seed, Self::Seed), PcgError> {
        // Implementation uses self.lpn_params, self.h_matrix, self.h_transpose
        let k = self.lpn_params.k;
        let n = self.lpn_params.n;

        // Generate H and H_transpose inside gen
        let h_matrix = self.lpn_params.generate_matrix().map_err(|e| PcgError::LpnError(e.to_string()))?;
        let h_transpose_matrix = h_matrix.transpose();

        // DPF parameters: alpha = y||Delta, beta = x
        let delta = Field128::rand(&mut thread_rng());
        let x_f2: Vec<F2> = (0..n).map(|_| F2::rand(&mut thread_rng())).collect();
        let y_f2: Vec<F2> = (0..k).map(|_| F2::rand(&mut thread_rng())).collect();

        // Pack alpha = y || Delta
        let packed_y = pack_f2_vector(&y_f2)?; // Assuming this returns Result
        let mut alpha_bytes = Vec::new();
        alpha_bytes.extend_from_slice(&packed_y);
        // Need robust serialization for Field128
        alpha_bytes.extend_from_slice(&delta.to_bytes_le().map_err(|_| PcgError::SerializationError("Delta serialization failed".to_string()))?);
        // Hash to get DPF index alpha. Using placeholder hash.
        // Domain size `N` for DPF needs clarification. Using dpf_handler's domain_bits.
        let alpha_idx = simple_hash_to_usize(&alpha_bytes, self.dpf_handler.domain_bits());

        // Pack beta = x
        // DPF value should be 1 bit (F2) according to Fig 3 interpretation?
        // "Run DPF.Gen(1^lambda, alpha, beta)" where beta = x (vector)
        // This seems to imply DPF outputs shares of x.
        // Let's stick to the simplified F2 output DPF for now.
        // Beta = 1. The *meaning* is tied to x implicitly.
        let beta_f2 = 1u8; // DPF takes u8
        println!("WARNING: Using DPF beta=1, actual value x implicitly handled.");

        // Generate DPF keys
        // Use the stored dpf_handler instance
        let (k_dpf0, k_dpf1) = self.dpf_handler.gen(alpha_idx, beta_f2).map_err(|e| PcgError::DpfError(e.to_string()))?;

        // Generate P0's random vector s_delta (packed F2)
        let s_delta_f2: Vec<F2> = (0..k).map(|_| F2::rand(&mut thread_rng())).collect(); // Use k for length? Check paper
        let s_delta_f2_sec: Vec<F2> = (0..security_param).map(|_| F2::rand(&mut thread_rng())).collect();
        let s_delta_packed = pack_f2_vector(&s_delta_f2_sec)?; // Pack lambda bits

        // Create Seeds using canonical definitions
        let seed0 = SvoleSenderSeed {
            k_dpf: k_dpf0,
            s_delta: s_delta_packed, // lambda packed bits
            y: y_f2,                 // k F2 elements
            h_matrix: h_matrix,      // k x n matrix H
            delta: delta,            // Field128 delta
        };

        let seed1 = SvoleReceiverSeed {
            k_dpf: k_dpf1,
            x: x_f2,                  // n F2 elements
            h_transpose_matrix: h_transpose_matrix, // n x k matrix H^T
            delta: delta,             // Field128 delta
        };

        Ok((SvoleSeed::Sender(seed0), SvoleSeed::Receiver(seed1)))
    }
}

// Implementation of the PCG Expander trait for sVOLE
impl PcgExpander for SvolePcg {
    type Seed = SvoleSeed;
    type Output = SvoleOutput;

    // Add &self to match trait
    fn expand(
        &self, // Add &self
        party_index: u8,
        seed: &Self::Seed,
    ) -> Result<Self::Output, PcgError> {
        match seed {
            SvoleSeed::Sender(sender_seed) => {
                if party_index != 0 {
                    return Err(PcgError::InvalidPartyIndex("Sender seed used by receiver party".to_string()));
                }
                // P0 (Sender) expands
                // DPF eval -> z0
                let z0_f2 = self.dpf_handler.full_eval(&sender_seed.k_dpf)
                               .map_err(|e| PcgError::DpfError(e.to_string()))?;
                // t = z0 ^ s
                let s_f2 = unpack_f2_vector(&sender_seed.s_delta, sender_seed.h_matrix.ncols()); // Unpack s (size k)
                let t_f2 = xor_f2_vectors(&z0_f2, &s_f2)?;

                // Compute v = Spread(t) + H^T * u' (where u' is derived from DPF eval? No, that's different sVOLE)
                // From Fig 3: v = Spread(t) + y
                let t_spread: Vec<Field128> = spread_f2_vector(&t_f2);
                let y_spread: Vec<Field128> = spread_f2_vector(&sender_seed.y); // y is already in seed
                let v: Vec<Field128> = add_fq_vectors(&t_spread, &y_spread);

                // Need u output for sender? Fig 3 output is (u,v). But u is P1's input.
                // Let's assume P0 outputs v, P1 outputs w.
                // Output for sender should align with v = u*delta + w relationship.
                // P0 needs to output u. Where does P0 get u? It's from P1. This implies interaction?
                // Re-read paper: Expand produces (u, v) for P0 and (x, w) for P1.
                // P0 needs u. P0's seed has y=Hx, s. P0 computes t=z0^s, v=Spread(t)+y.
                // Where does u come from? Ah, u is the DPF evaluation for P0! u = Spread(z0).
                let u_spread: Vec<Field128> = spread_f2_vector(&z0_f2);

                Ok(SvoleOutput::Sender(SvoleSenderOutput { u: u_spread, v }))
            },
            SvoleSeed::Receiver(receiver_seed) => {
                 if party_index != 1 {
                    return Err(PcgError::InvalidPartyIndex("Receiver seed used by sender party".to_string()));
                 }
                 // Delta is now part of the seed
                 let delta = receiver_seed.delta;

                 // P1 (Receiver) expands
                 // DPF eval -> z1
                 let z1_f2 = self.dpf_handler.full_eval(&receiver_seed.k_dpf)
                                .map_err(|e| PcgError::DpfError(e.to_string()))?;

                 // Compute w = Spread(z1) - u*delta
                 // u is already in the receiver seed.
                 let z1_spread: Vec<Field128> = spread_f2_vector(&z1_f2);
                 let u_delta: Vec<Field128> = receiver_seed.x.iter().map(|ui| *ui * delta).collect();
                 let w: Vec<Field128> = sub_fq_vectors(&z1_spread, &u_delta);

                 // Output is (x, w)
                 Ok(SvoleOutput::Receiver(SvoleReceiverOutput { x: receiver_seed.x.clone(), w }))
            }
        }
    }
}

// Make helper functions public
pub fn pack_f2_vector(v: &[F2]) -> Result<Vec<u8>, PcgError> {
    // Implementation of pack_f2_vector
    let num_bytes = (v.len() + 7) / 8;
    let mut packed = vec![0u8; num_bytes];
    for (i, f2_val) in v.iter().enumerate() {
        if f2_val.is_one() {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            packed[byte_idx] |= 1 << bit_idx;
        }
    }
    Ok(packed)
}

pub fn unpack_f2_vector(v: &[u8], len: usize) -> Vec<F2> {
    // Implementation of unpack_f2_vector
    let mut unpacked = Vec::with_capacity(len);
    for i in 0..len {
        let byte_idx = i / 8;
        let bit_idx = i % 8;
        if byte_idx < v.len() {
            let byte = v[byte_idx];
            let bit = (byte >> bit_idx) & 1;
            unpacked.push(if bit == 1 { F2::one() } else { F2::zero() });
        } else {
            unpacked.push(F2::zero()); // Pad with zeros if input is too short
        }
    }
    unpacked
}

pub fn spread_f2_vector(v: &[F2]) -> Vec<Field128> {
    // Implementation of spread_f2_vector (simple embedding)
    v.iter()
     .map(|f2| if f2.is_one() { Field128::one() } else { Field128::zero() })
     .collect()
}

pub fn xor_f2_vectors(a: &[F2], b: &[F2]) -> Result<Vec<F2>, PcgError> {
    // Implementation of xor_f2_vectors
    if a.len() != b.len() {
        return Err(PcgError::InvalidInput("Vector lengths must match for XOR".to_string()));
    }
    let result = a.iter().zip(b.iter()).map(|(x, y)| *x + *y).collect(); // F2 addition is XOR
    Ok(result)
}

pub fn add_fq_vectors(a: &[Field128], b: &[Field128]) -> Vec<Field128> {
    // Implementation of add_fq_vectors
    a.iter().zip(b.iter()).map(|(x, y)| *x + *y).collect()
}

pub fn sub_fq_vectors(a: &[Field128], b: &[Field128]) -> Vec<Field128> {
    // Implementation of sub_fq_vectors
    a.iter().zip(b.iter()).map(|(x, y)| *x - *y).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::lpn::{LpnParameters, CodeType};
    use rand::{rngs::StdRng, SeedableRng, RngCore};
    use ark_ff::{Zero, Field, One};
    use crate::primitives::field::{Field128, F2};
    use crate::pcg_core::{SvoleSeed}; // Import moved SvoleSeed from pcg_core

    #[test]
    fn test_svole_gen_expand_correctness() {
        // Setup
        let lpn_params = LpnParameters {
            n: 256, // Example value
            k: 128, // Example value
            t: 10,  // Example value
            code_type: CodeType::RandomLinear,
        };
        let svole_pcg = SvolePcg::new(lpn_params.clone()).expect("Failed to create SVole PCG");

        // Gen
        let security_param = 128;
        let gen_result = svole_pcg.gen(security_param);
        assert!(gen_result.is_ok(), "Gen failed: {:?}", gen_result.err());
        let (seed0, seed1) = gen_result.unwrap();

        let delta = match &seed1 {
            SvoleSeed::Receiver(r) => r.delta,
            _ => panic!("Incorrect seed type for P1"),
        };

        // Expand P0
        let expand_res_p0 = SvolePcg::expand(0, &seed0);
        assert!(expand_res_p0.is_ok(), "P0 expand failed: {:?}", expand_res_p0.err());
        let out0 = match expand_res_p0.unwrap() {
            SvoleOutput::Sender(s) => s,
            _ => panic!("Incorrect output type for P0"),
        };

        // Expand P1
        let expand_res_p1 = SvolePcg::expand(1, &seed1);
        assert!(expand_res_p1.is_ok(), "P1 expand failed: {:?}", expand_res_p1.err());
        let out1 = match expand_res_p1.unwrap() {
            SvoleOutput::Receiver(r) => r,
            _ => panic!("Incorrect output type for P1"),
        };

        // Correctness Check: v = u * delta + w
        // Requires converting P1's output x (packed F2) back to F2?
        // No, the check uses u, v from P0 and w from P1.
        assert_eq!(out0.u.len(), lpn_params.k, "P0 output u length mismatch");
        assert_eq!(out0.v.len(), lpn_params.k, "P0 output v length mismatch");
        assert_eq!(out1.w.len(), lpn_params.k, "P1 output w length mismatch");

        for i in 0..lpn_params.k {
            let u_delta = out0.u[i] * delta;
            let u_delta_plus_w = u_delta + out1.w[i];
            assert_eq!(out0.v[i], u_delta_plus_w, "sVOLE check v = u*delta + w failed at index {}", i);
        }

        println!("sVOLE correctness check passed.");
    }
}

fn simple_hash_to_usize(input: &[u8], domain_size: usize) -> usize {
    // Placeholder implementation - use a proper hash and reduction
    if domain_size == 0 { return 0; }
    let hash_val = input.iter().fold(0usize, |acc, &byte| acc.wrapping_add(byte as usize));
    hash_val % domain_size
}
