use crate::pcg_core::{PcgError, PcgExpander, PcgSeedGenerator};
use crate::primitives::crhf::{Crhf, AesCrhf};
use crate::primitives::field::F2_128;
use crate::primitives::lpn::LpnParameters;
use crate::svole::{SvolePcg, SvoleSeed0, SvoleSeed1, SvoleOutput0, SvoleOutput1};
use ark_ff::Field; // For DPF field F
use ark_ff::UniformRand; // For Svole Gen
use ark_std::vec::Vec;
use ark_std::marker::PhantomData;

// Define Output Structures for Random OT
// For `n` OTs, where `n` is the LPN codeword length.

/// Output for the OT Sender (derived from sVOLE Receiver P1)
/// Contains pairs of messages (m_{i,0}, m_{i,1}) for each OT i.
/// Messages are typically 128-bit strings.
pub struct RotSenderOutput {
    pub messages: Vec<(Vec<u8>, Vec<u8>)>, // Vec of (m_i0, m_i1)
}

/// Output for the OT Receiver (derived from sVOLE Sender P0)
/// Contains choice bits `c_i` and the corresponding received messages `m_{i, c_i}`.
pub struct RotReceiverOutput {
    pub choice_bits: Vec<u8>, // Vec of c_i (0 or 1)
    pub messages: Vec<Vec<u8>>, // Vec of m_{i, c_i}
}

/// Random OT PCG Wrapper around sVOLE PCG.
/// F: Field used for DPF values within sVOLE (e.g., representing F2).
/// H: CRHF instance used for transforming sVOLE output.
pub struct RotPcg<F: Field, H: Crhf<Input = [u8], Output = Vec<u8>>> {
    svole_pcg: SvolePcg<F>,
    crhf: H,
}

impl<F, H> RotPcg<F, H>
where
    F: Field + UniformRand + Sync + Send + From<u128>,
    H: Crhf<Input = [u8], Output = Vec<u8>> + Sync + Send,
{
    /// Creates a new RotPcg instance.
    /// LPN parameters should be configured for OT (p=2, q=2^r).
    /// Requires an sVOLE PCG instance and a CRHF instance.
    pub fn new(svole_pcg: SvolePcg<F>, crhf: H) -> Self {
        Self { svole_pcg, crhf }
    }

    /// Convenience constructor using default AesCrhf.
    pub fn new_with_default_crhf(lpn_params: LpnParameters) -> RotPcg<F, AesCrhf<[u8]>> {
         RotPcg::new(SvolePcg::new(lpn_params), AesCrhf::new())
    }
}

// Seed Generation for ROT PCG just calls the underlying sVOLE PCG Gen.
// The seeds themselves are sVOLE seeds.
impl<F, H> PcgSeedGenerator for RotPcg<F, H>
where
    F: Field + UniformRand + Sync + Send + From<u128>,
    H: Crhf<Input = [u8], Output = Vec<u8>> + Sync + Send,
{
    type Seed0 = SvoleSeed0<F>;
    type Seed1 = SvoleSeed1<F>;

    /// Calls SvolePcg::gen. LPN parameters must be set for OT (p=2, q=2^r).
    fn gen(
        security_param: usize,
        lpn_params: &LpnParameters,
    ) -> Result<(Self::Seed0, Self::Seed1), PcgError> {
        SvolePcg::<F>::gen(security_param, lpn_params)
    }
}

// Expansion for ROT PCG
impl<F, H> PcgExpander for RotPcg<F, H>
where
    F: Field + UniformRand + Sync + Send + From<u128>,
    H: Crhf<Input = [u8], Output = Vec<u8>> + Sync + Send,
{
    // Seed type is the sVOLE seed tuple
    type Seed = (u8, SvoleSeed0<F>, SvoleSeed1<F>);
    // Output type depends on the party index
    type Output = Result<RotSenderOutput, RotReceiverOutput>; // Ok for Sender, Err for Receiver

    /// ROT_PCG::Expand (FR4.2)
    /// 1. Calls sVOLE_PCG::Expand.
    /// 2. Performs role switch.
    /// 3. Applies CRHF H to transform sVOLE outputs into ROT shares.
    fn expand(
        &self,
        party_index: u8, // 0 or 1
        seed: &Self::Seed,
        // No explicit correlation parameters needed here beyond what's in sVOLE/LPN params
        // But sVOLE expand needs delta, which is internal to sVOLE P1.
        // We need to generate/pass delta appropriately.
        // For Random OT, delta can be random? Let's assume sVOLE P1 generates it.
    ) -> Result<Self::Output, PcgError> {

        let mut rng = rand::thread_rng();
        // Placeholder: Generate a random delta for sVOLE P1 (OT Sender)
        // In a real protocol, this might be derived or fixed.
        let delta: F2_128 = u128::rand(&mut rng);

        // 1. Call sVOLE expand
        let svole_outputs = self.svole_pcg.expand(party_index, seed, delta)?;
        let (svole_output0, svole_output1) = svole_outputs;

        let n = self.svole_pcg.lpn_params.n;

        // 2. Role Switch and 3. Apply CRHF (Fig 4 / Sec 5.2)
        if party_index == 1 { // sVOLE Receiver P1 -> OT Sender
            let mut ot_messages = Vec::with_capacity(n);
            // P1 has delta and w = (w_i)_{i=1..n}
            let w = svole_output1.w;
            let delta_bytes = delta.to_le_bytes();

            for i in 0..n {
                // m_{i,0} = H(i, w_i)
                let wi_bytes = w[i].to_le_bytes(); // Need consistent serialization
                let m_i0 = self.crhf.hash(i, &wi_bytes);

                // m_{i,1} = H(i, w_i ^ delta)
                let w_xor_delta = w[i] ^ delta;
                let w_xor_delta_bytes = w_xor_delta.to_le_bytes();
                let m_i1 = self.crhf.hash(i, &w_xor_delta_bytes);

                ot_messages.push((m_i0, m_i1));
            }
            Ok(Ok(RotSenderOutput { messages: ot_messages }))

        } else { // sVOLE Sender P0 -> OT Receiver
            let mut ot_choices = Vec::with_capacity(n);
            let mut ot_received_messages = Vec::with_capacity(n);
            // P0 has u = (u_i)_{i=1..n} and v = (v_i)_{i=1..n}
            let u = svole_output0.u;
            let v = svole_output0.v;

            for i in 0..n {
                let choice_bit = u[i]; // c_i = u_i
                ot_choices.push(choice_bit);

                // m_{i, c_i} = H(i, v_i)
                let vi_bytes = v[i].to_le_bytes(); // Need consistent serialization
                let m_i_ci = self.crhf.hash(i, &vi_bytes);
                ot_received_messages.push(m_i_ci);
            }
            Ok(Err(RotReceiverOutput { choice_bits: ot_choices, messages: ot_received_messages }))
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::lpn::{CodeType, LpnParameters};
    use crate::svole::SvolePcg;
    use crate::primitives::crhf::AesCrhf;
    use ark_test_curves::bls12_381::Fq as TestField; // Example Field for DPF

    #[test]
    fn test_rot_pcg_gen_expand_placeholders() {
        let lpn_params = LpnParameters {
            n: 256,
            k: 128,
            t: 10,
            code_type: CodeType::RandomLinear,
        };

        // Create RotPcg with default sVOLE and CRHF placeholders
        let rot_pcg = RotPcg::<TestField, AesCrhf<[u8]>>::new_with_default_crhf(lpn_params.clone());

        // Test Gen
        let gen_result = RotPcg::<TestField, AesCrhf<[u8]>>::gen(128, &lpn_params);
        assert!(gen_result.is_ok());
        let (seed0, seed1) = gen_result.unwrap();

        // Test Expand for OT Sender (Party 1)
        let seed_tuple_p1 = (1u8, seed0.clone(), seed1.clone()); // Clone seeds if needed
        let expand_res_p1 = rot_pcg.expand(1, &seed_tuple_p1);
        assert!(expand_res_p1.is_ok());
        let sender_output_res = expand_res_p1.unwrap();
        assert!(sender_output_res.is_ok());
        let sender_output = sender_output_res.unwrap();
        assert_eq!(sender_output.messages.len(), lpn_params.n);
        if lpn_params.n > 0 {
             assert_eq!(sender_output.messages[0].0.len(), 16); // Expect 128-bit messages from AES CRHF
             assert_eq!(sender_output.messages[0].1.len(), 16);
        }

        // Test Expand for OT Receiver (Party 0)
        let seed_tuple_p0 = (0u8, seed0, seed1);
        let expand_res_p0 = rot_pcg.expand(0, &seed_tuple_p0);
        assert!(expand_res_p0.is_ok());
        let receiver_output_res = expand_res_p0.unwrap();
        assert!(receiver_output_res.is_err());
        let receiver_output = receiver_output_res.err().unwrap();
        assert_eq!(receiver_output.choice_bits.len(), lpn_params.n);
        assert_eq!(receiver_output.messages.len(), lpn_params.n);
        if lpn_params.n > 0 {
             assert!(receiver_output.choice_bits[0] == 0 || receiver_output.choice_bits[0] == 1);
             assert_eq!(receiver_output.messages[0].len(), 16); // Expect 128-bit messages
        }

        // TODO: Add consistency check: H(i, w_i ^ (u_i * delta)) == H(i, v_i)
        // Requires access to internal values or careful reconstruction.
        println!("ROT PCG placeholder test passed basic structure checks.");
    }
}
