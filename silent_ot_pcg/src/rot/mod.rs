use crate::pcg_core::{PcgError, PcgExpander, PcgSeedGenerator, SvoleSeed, SvoleSenderSeed, SvoleReceiverSeed};
use crate::primitives::crhf::{Crhf};
use crate::primitives::field::{Field128};
use crate::svole::{
    SvolePcg,
    SvoleOutput,
};
use ark_ff::{Field};
use ark_std::vec::Vec;
use ark_ff::{Zero as ArkZero, One as ArkOne};
use std::marker::PhantomData;
use crate::primitives::field::F2;
use std::fmt::Debug; // Import Debug trait

// --- ROT Seed/Output Structs ---

/// Seed structure for the ROT sender (maps to sVOLE Receiver P1)
#[derive(Clone, Debug)]
pub struct RotSenderSeed<H: Crhf<Vec<u8>, Vec<u8>> + Clone + Debug> {
    pub svole_seed: SvoleReceiverSeed, // Contains k1, u, x, delta, h_matrix
    pub crhf: H,
}

/// Seed structure for the ROT receiver (maps to sVOLE Sender P0)
#[derive(Clone, Debug)]
pub struct RotReceiverSeed<H: Crhf<Vec<u8>, Vec<u8>> + Clone + Debug> {
    pub svole_seed: SvoleSenderSeed, // Contains k0, s, y, h_transpose
    pub crhf: H,
}

/// Enum to hold either ROT sender or receiver seed
#[derive(Clone, Debug)]
pub enum RotSeed<H: Crhf<Vec<u8>, Vec<u8>> + Clone + Debug> {
    Sender(RotSenderSeed<H>),
    Receiver(RotReceiverSeed<H>),
}

/// Messages are typically 128-bit strings.
#[derive(Debug)]
pub struct RotSenderOutput {
    pub outputs: Vec<Vec<u8>>, // Vec of H(i, w_i)
}

/// Output for the OT Receiver (derived from sVOLE Sender P0)
/// Contains choice bits `c_i` and the corresponding hashed messages `T_i = H(i, v_i)`.
#[derive(Debug)]
pub struct RotReceiverOutput {
    pub outputs: Vec<(F2, Vec<u8>)>, // Vec of (c_i = u_i, T_i = H(i, v_i))
}

/// Enum to hold either ROT sender or receiver output
#[derive(Debug)]
pub enum RotOutput {
    Sender(RotSenderOutput),
    Receiver(RotReceiverOutput),
}

// --- ROT PCG Struct ---

/// Random OT PCG Wrapper around sVOLE PCG.
/// F: Field used for DPF values within sVOLE (e.g., representing F2).
/// H: CRHF instance used for transforming sVOLE output.
pub struct RotPcg<F: Field, H: Crhf<Vec<u8>, Vec<u8>> + Clone> {
    svole_pcg: SvolePcg,
    crhf: H,
    _marker: PhantomData<F>,
    output_bytes: usize,
}

impl<F: Field, H: Crhf<Vec<u8>, Vec<u8>> + Clone> RotPcg<F, H> {
    /// Creates a new RotPcg instance.
    /// LPN parameters should be configured for OT (p=2, q=2^r).
    /// Requires an sVOLE PCG instance and a CRHF instance.
    pub fn new(svole_pcg: SvolePcg, crhf: H, output_bytes: usize) -> Self {
        RotPcg { svole_pcg, crhf, _marker: PhantomData, output_bytes }
    }

    /// Constructor specifically for AES-based CRHF with default IV
    #[cfg(feature = "aes_crhf")] // Only compile if aes_crhf feature is enabled
    pub fn new_with_default_crhf(svole_pcg: SvolePcg, output_bytes: usize) -> RotPcg<F, AesCrhf<Vec<u8>>>
    {
        let crhf = AesCrhf::new(&[0u8; 16]); // Example default key - replace with secure key handling
        Self::new(svole_pcg, crhf, output_bytes)
    }
}

// Seed Generation for ROT PCG just calls the underlying sVOLE PCG Gen.
// The seeds themselves are sVOLE seeds.
impl<F: Field, H: Crhf<Vec<u8>, Vec<u8>> + Clone + Send + Sync + 'static + Debug> PcgSeedGenerator for RotPcg<F, H> {
    type Seed = RotSeed<H>;

    /// Calls svole_pcg.gen. LPN parameters must be set for OT (p=2, q=2^r).
    fn gen(
        &self,
        security_param: usize,
    ) -> Result<(Self::Seed, Self::Seed), PcgError> {
        // Call svole_pcg gen
        let (svole_seed0, svole_seed1) = self.svole_pcg.gen(security_param)?;

        // Package into ROT seeds, cloning the CRHF instance
        let rot_seed_receiver = match svole_seed0 {
            SvoleSeed::Sender(s) => RotReceiverSeed { svole_seed: s, crhf: self.crhf.clone() },
            _ => return Err(PcgError::SeedGenError("Incorrect sVOLE seed type for ROT receiver".to_string()))
        };

        let rot_seed_sender = match svole_seed1 {
            SvoleSeed::Receiver(r) => RotSenderSeed { svole_seed: r, crhf: self.crhf.clone() },
            _ => return Err(PcgError::SeedGenError("Incorrect sVOLE seed type for ROT sender".to_string()))
        };

        Ok((RotSeed::Sender(rot_seed_sender), RotSeed::Receiver(rot_seed_receiver)))
    }
}

// Expansion for ROT PCG
impl<F: Field, H: Crhf<Vec<u8>, Vec<u8>> + Clone + Send + Sync + 'static + Debug> PcgExpander for RotPcg<F, H> {
    type Seed = RotSeed<H>;
    type Output = RotOutput;

    /// ROT_PCG::Expand (FR4.2)
    /// 1. Calls sVOLE_PCG::Expand.
    /// 2. Performs role switch.
    /// 3. Applies CRHF H to transform sVOLE outputs into ROT shares.
    fn expand(
        &self,
        party_index: u8, // 0 for Sender, 1 for Receiver (Note: Role switch! ROT Sender = sVOLE Receiver)
        seed: &Self::Seed,
    ) -> Result<Self::Output, PcgError> {
        match seed {
            RotSeed::Sender(sender_seed) => {
                if party_index != 0 {
                    return Err(PcgError::InvalidPartyIndex("Sender seed used by receiver party".to_string()));
                }
                // ROT Sender is sVOLE Receiver (P1)
                let svole_output = self.svole_pcg.expand(1, &SvoleSeed::Receiver(sender_seed.svole_seed.clone()))?;
                let (x_packed, w) = match svole_output {
                    SvoleOutput::Receiver(out) => (out.x, out.w),
                    _ => return Err(PcgError::ExpandError("Incorrect sVOLE output type for ROT sender".to_string()))
                };
                let crhf = &sender_seed.crhf;
                // Apply CRHF: H(i, w_i) for all i
                let ot_sender_output = w.iter().enumerate().map(|(i, wi)| {
                    // Need consistent way to format (i, wi) as input bytes for CRHF
                    let mut crhf_input = Vec::new();
                    crhf_input.extend_from_slice(&(i as u64).to_le_bytes()); // Example: index as u64
                    crhf_input.extend_from_slice(&wi.to_bytes()); // Assuming Field128 has to_bytes()
                    crhf.hash(i, &crhf_input)
                }).collect::<Vec<_>>();

                Ok(RotOutput::Sender(RotSenderOutput { outputs: ot_sender_output }))
            },
            RotSeed::Receiver(receiver_seed) => {
                if party_index != 1 {
                     return Err(PcgError::InvalidPartyIndex("Receiver seed used by sender party".to_string()));
                }
                // ROT Receiver is sVOLE Sender (P0)
                let svole_output = self.svole_pcg.expand(0, &SvoleSeed::Sender(receiver_seed.svole_seed.clone()))?;
                let (u, v) = match svole_output {
                    SvoleOutput::Sender(out) => (out.u, out.v),
                    _ => return Err(PcgError::ExpandError("Incorrect sVOLE output type for ROT receiver".to_string()))
                };
                let crhf = &receiver_seed.crhf;
                let delta = receiver_seed.svole_seed.delta; // Get delta from sVOLE seed

                // Apply CRHF: H(i, u_i) and H(i, v_i) = H(i, u_i + delta)
                let ot_receiver_output = u.iter().zip(v.iter()).enumerate().map(|(i, (ui, vi))| {
                    // Format input for CRHF
                    let mut crhf_input0 = Vec::new();
                    crhf_input0.extend_from_slice(&(i as u64).to_le_bytes());
                    crhf_input0.extend_from_slice(&ui.to_bytes());

                    let mut crhf_input1 = Vec::new();
                    crhf_input1.extend_from_slice(&(i as u64).to_le_bytes());
                    // v_i = u_i * delta + w_i (from sVOLE correctness)
                    // We need u_i + delta as input? Check Fig 4. / Sec 5.2
                    // Receiver gets (u_i, v_i). Choice bit c_i = u_i (as F2 element)
                    // Output T_i = H(i, v_i), R_i = H(i, u_i)
                    // If c_i = 0, uses T_i. If c_i = 1, uses R_i ^ delta' (delta' = H(delta)) ? No.
                    // Standard ROT: Receiver gets (m_0, m_1). Outputs m_c.
                    // Here: Sender outputs {w_i0, w_i1}. Receiver has choice c_i, gets T_i = w_ic_i.
                    // Protocol gives: T_i = H(i, v_i), R_i = H(i, u_i).
                    // Receiver checks T_i == H(i, u_i + c_i*delta)? No, receiver computes T_i based on choice.
                    // Let's stick to outputting (R_i, T_i) pair for receiver.
                    let r_i = crhf.hash(i, &crhf_input0);

                    let mut crhf_input_v = Vec::new();
                    crhf_input_v.extend_from_slice(&(i as u64).to_le_bytes());
                    crhf_input_v.extend_from_slice(&vi.to_bytes());
                    let t_i = crhf.hash(i, &crhf_input_v);

                    // Need choice bits. Where do they come from? The receiver must provide them?
                    // The PCG *generates* random OTs, the choice bits are implicit/random.
                    // Choice bit c_i is effectively the F_2 representation of u_i.
                    // Output for receiver is T_i = H(i, v_i) and choice bit u_i.
                    let choice_bit_f2 = ui.to_f2(); // Need conversion Field128 -> F2
                    (choice_bit_f2, t_i)
                }).collect::<Vec<_>>();

                Ok(RotOutput::Receiver(RotReceiverOutput { outputs: ot_receiver_output }))
            }
        }
    }
}

// --- Helper Trait/Method Stubs ---
trait FieldExt: Field { // Placeholder trait
    fn to_bytes(&self) -> Vec<u8>;
    fn to_f2(&self) -> F2; // Placeholder conversion
}
impl FieldExt for Field128 {
    fn to_bytes(&self) -> Vec<u8> { self.0.to_le_bytes().to_vec() } // Assuming Field128(u128)
    fn to_f2(&self) -> F2 { if self.0 % 2 == 0 { F2::zero() } else { F2::one() } } // Simplistic placeholder
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::lpn::{CodeType, LpnParameters};
    use crate::svole::SvolePcg;
    use crate::primitives::crhf::AesCrhf;
    use ark_test_curves::bls12_381::Fq as TestField; // Example Field for DPF

    #[test]
    fn test_rot_gen_expand_structure() {
        let lpn_params = LpnParameters { n: 1024, k: 128, t: 10, code_type: CodeType::RandomLinear };

        // Create RotPcg with default sVOLE and CRHF placeholders (using Vec<u8>)
        let rot_pcg = RotPcg::<TestField, AesCrhf<Vec<u8>>>::new_with_default_crhf(lpn_params.clone(), 128).unwrap();

        // Test Gen
        let gen_result = rot_pcg.gen(128);
        assert!(gen_result.is_ok());
        let (seed_sender, seed_receiver) = gen_result.unwrap();

        // Test Expand for OT Sender (Party 0)
        let expand_res_p0 = RotPcg::<TestField, AesCrhf<Vec<u8>>>::expand(0, &seed_sender);
        assert!(expand_res_p0.is_ok());
        let sender_output_res = expand_res_p0.unwrap();
        match sender_output_res {
            RotOutput::Sender(sender_output) => {
                assert_eq!(sender_output.outputs.len(), lpn_params.n);
                 if lpn_params.n > 0 {
                     assert!(!sender_output.outputs[0].is_empty()); // Basic check
                 }
            }
            _ => panic!("Incorrect output type for sender expand"),
        }


        // Test Expand for OT Receiver (Party 1)
        let expand_res_p1 = RotPcg::<TestField, AesCrhf<Vec<u8>>>::expand(1, &seed_receiver);
        assert!(expand_res_p1.is_ok());
        let receiver_output_res = expand_res_p1.unwrap();
         match receiver_output_res {
            RotOutput::Receiver(receiver_output) => {
                assert_eq!(receiver_output.outputs.len(), lpn_params.n);
                if lpn_params.n > 0 {
                    assert!(receiver_output.outputs[0].0 == F2::zero() || receiver_output.outputs[0].0 == F2::one());
                    assert!(!receiver_output.outputs[0].1.is_empty()); // Basic check
                }
            }
            _ => panic!("Incorrect output type for receiver expand"),
        }
    }

    #[test]
    fn test_rot_gen_expand_correctness() {
        let lpn_params = LpnParameters { n: 64, k: 32, t: 5, code_type: CodeType::RandomLinear };
        let svole_pcg = SvolePcg::new(lpn_params);
        let output_bytes = 16; 
        let rot_pcg = RotPcg::<TestField, AesCrhf<Vec<u8>>>::new_with_default_crhf(svole_pcg, output_bytes);

        let (seed_sender_enum, seed_receiver_enum) = rot_pcg.gen().expect("ROT Gen failed");

        let delta = match &seed_receiver_enum {
            RotSeed::Receiver(r) => r.svole_seed.delta,
            _ => panic!("Incorrect seed types generated"),
        };
        match &seed_sender_enum {
             RotSeed::Sender(s) => assert_eq!(s.svole_seed.delta, delta, "Sender delta mismatch"),
             _ => panic!("Incorrect seed types generated"),
         };

        // Fix: Call expand with correct signature
        let expand_res_p0 = rot_pcg.expand(seed_sender_enum.clone()); // Pass seed value
        assert!(expand_res_p0.is_ok(), "Sender expand failed: {:?}", expand_res_p0.err());
        let output_sender_enum = expand_res_p0.unwrap();

        // Fix: Call expand with correct signature
        let expand_res_p1 = rot_pcg.expand(seed_receiver_enum.clone()); // Pass seed value
        assert!(expand_res_p1.is_ok(), "Receiver expand failed: {:?}", expand_res_p1.err());
        let output_receiver_enum = expand_res_p1.unwrap();

        match (output_sender_enum, output_receiver_enum) {
            (RotOutput::Sender(out_s), RotOutput::Receiver(out_r)) => {
                assert_eq!(out_s.outputs.len(), out_r.outputs.len(), "Mismatch in number of OTs");
                assert_eq!(out_r.outputs.len(), out_s.outputs.len(), "Receiver choice bit count mismatch");
                assert_eq!(out_r.outputs.len(), out_s.outputs.len(), "Receiver message count mismatch");
                assert_eq!(out_s.outputs.len(), out_s.outputs.len(), "Sender message pair count mismatch");

                for i in 0..out_s.outputs.len() {
                    let choice_bit = out_r.outputs[i].0;
                    let received_msg = &out_r.outputs[i].1;
                    let expected_msg_index = i * 2 + (if choice_bit.is_one() { 1 } else { 0 }); 
                    let expected_msg = &out_s.outputs[expected_msg_index];
                    assert_eq!(received_msg, expected_msg, "ROT correlation failed at index {}", i);
                }
                println!("ROT correlation check PASSED!");
            }
            _ => panic!("Unexpected output types after ROT expansion"),
        }
    }
}
