use crate::pcg_core::{PcgError, PcgSeedGenerator, PcgSeed, SvoleSenderSeed, SvoleReceiverSeed, SvoleSeed};
use crate::primitives::{dpf::{Dpf, DpfKey, DpfTrait}, lpn::LpnParameters, field::{Field128, F2}};
use crate::svole::{pack_f2_vector}; // Removed unpack_f2_vector
use ark_ff::{Field, One, Zero, UniformRand, PrimeField};
use ark_std::vec::Vec;
use ark_std::marker::PhantomData;
use std::sync::mpsc::{channel, Sender, Receiver}; // Removed TryRecvError
use rand::thread_rng;
use std::fmt::Debug;

// --- Communication Channel Abstraction ---
trait ChannelError: std::fmt::Debug + Send + Sync + 'static {}
impl ChannelError for String {}

pub trait Channel<T: Send + Clone + Debug>: Send {
    fn send(&self, message: T) -> Result<(), PcgError>;
    fn recv(&self) -> Result<T, PcgError>;
}

// --- Base OT Interface Placeholder ---
// The interactive seed generation protocol relies on a base OT implementation.
// Define a simple trait to represent the required functionality.

pub enum BaseOtType {
    Sender,
    Receiver,
}

/// Placeholder trait for Base OT functionality.
/// A real implementation would involve a specific OT protocol (e.g., IKNP, SimplestOT).
pub trait BaseOtOracle {
    // For sender
    fn send(&mut self, messages: &[(Vec<u8>, Vec<u8>)]) -> Result<(), PcgError>;
    // For receiver
    fn receive(&mut self, choices: &[u8]) -> Result<Vec<Vec<u8>>, PcgError>;
}

// --- Secure DPF Key Generation Placeholder ---
// The protocol also needs a mechanism to securely generate DPF keys.
// This itself is often built using OTs (e.g., [Ds17]).

/// Placeholder function representing secure DPF key generation.
/// Takes base OTs and returns DPF key shares.
fn secure_dpf_key_gen<C: Channel<Vec<u8>>>(
    _channel: &C, // Channel for the protocol
    _sec_param: usize, // Security parameter
    _dpf_domain_bits: usize, // n for DPF domain size N=2^n
    _alpha: usize, // The point index (derived from y || Delta)
    _beta: Vec<u8>, // The value (packed x)
) -> Result<DpfKey, PcgError> {
    println!("Warning: Using insecure placeholder DpfKey generation! Needs secure 2PC impl.");
    // Reinstating NotImplemented error as per instructions
    Err(PcgError::NotImplemented("Secure DPF Gen needs implementation".to_string()))
    /* // Previous placeholder logic using Dpf::gen:
    let mut rng = thread_rng();
    let (k0, _k1) = Dpf::gen(_dpf_domain_bits, _alpha, _beta, &mut rng)?;
    Ok(k0)
    */
}


// --- Interactive sVOLE Seed Generation Protocol ---

/// Placeholder for the interactive seed generation state and logic.
/// F is the DPF output field (likely F2).
/// C is the communication channel type.
pub struct InteractiveSeedGenerator<C: Channel<DpfKey>, F: Field> {
    party_id: u8, // 0 or 1
    lpn_params: LpnParameters,
    channel: C,
    _marker: PhantomData<F>,
}

impl<C: Channel<DpfKey> + Send + 'static, F: Field + Sync + Send + 'static + UniformRand + PrimeField>
    InteractiveSeedGenerator<C, F>
{
    /// Creates a pair of generators, one for each party, connected by channels.
    /// This is a helper for testing or setting up a 2-party computation.
    /// Updated to create MpscChannel internally, simplifying test setup.
    pub fn new_pair_mpsc(
        lpn_params: LpnParameters,
    ) -> (
        InteractiveSeedGenerator<MpscChannel<DpfKey>, F>,
        InteractiveSeedGenerator<MpscChannel<DpfKey>, F>,
    ) {
        let (sender0, receiver1) = channel::<DpfKey>(); // Chan P0 -> P1
        let (sender1, receiver0) = channel::<DpfKey>(); // Chan P1 -> P0

        let chan0 = MpscChannel { sender: sender0, receiver: receiver0 };
        let chan1 = MpscChannel { sender: sender1, receiver: receiver1 };

        // Use the standard `new` method with the created Mpsc channels
        (Self::new(0, lpn_params.clone(), chan0), Self::new(1, lpn_params, chan1))
    }

    /// Creates a new generator instance.
    pub fn new(party_id: u8, lpn_params: LpnParameters, channel: C) -> Self {
        Self {
            party_id,
            lpn_params,
            channel,
            _marker: PhantomData,
        }
    }

    /// Executes the interactive protocol to generate the sVOLE seed share.
    /// Corresponds to FR5.1.
    /// Returns the local sVOLE seed share (wrapped in SvoleSeed enum).
    pub fn generate_local_seed(&mut self) -> Result<SvoleSeed, PcgError> {
        println!("Starting interactive seed generation for Party {}...", self.party_id);
        // 1. Securely generate DPF keys for alpha=0, beta=1 (F2)
        // Assume secure_dpf_key_gen handles the interaction.
        let dpf_key = secure_dpf_key_gen(
            &self.channel,
            128, // security_param (placeholder)
            0,   // alpha (placeholder)
            vec![F2::one()], // beta = 1 (placeholder)
        )?;
        println!("Party {}: DPF key generated/received.", self.party_id);

        // 2. Generate other seed components locally
        let k = self.lpn_params.k;
        let n = self.lpn_params.n;
        let mut rng = thread_rng();

        // Generate H and H_transpose (needed for seeds)
        let h_matrix = self.lpn_params.generate_matrix().map_err(|e| PcgError::LpnError(e.to_string()))?;
        let h_transpose = h_matrix.transpose();

        if self.party_id == 0 { // P0 (sVOLE Sender)
            let s_f2: Vec<F2> = (0..k).map(|_| F2::rand(&mut rng)).collect();
            let s_packed: Vec<u8> = pack_f2_vector(&s_f2)?;
            // P0 needs y=Hx. This requires interaction or assumption.
            // Placeholder: P0 gets zero y.
            let y_f2 = vec![F2::zero(); k];
            println!("Warning: P0 using placeholder zero vector for y=Hx in interactive gen.");

            let seed0 = SvoleSenderSeed {
                k_dpf: dpf_key,
                s_delta: s_packed,
                h_matrix: h_matrix.clone(),
                delta: F2::zero(),
            };
            Ok(SvoleSeed::Sender(seed0))
        } else { // P1 (sVOLE Receiver)
            let u: Vec<Field128> = (0..k).map(|_| Field128::rand(&mut rng)).collect();
            let x_f2: Vec<F2> = (0..n).map(|_| F2::rand(&mut rng)).collect();
            let x_packed: Vec<u8> = pack_f2_vector(&x_f2)?;
            let delta = Field128::rand(&mut rng);

            let seed1 = SvoleReceiverSeed {
                k_dpf: dpf_key,
                u,
                x: x_packed,
                h_transpose_matrix: h_transpose,
            };
            Ok(SvoleSeed::Receiver(seed1))
        }
    }

    fn gen(&self, security_param: usize) -> Result<(<Self as PcgSeedGenerator>::Seed, <Self as PcgSeedGenerator>::Seed), PcgError> {
        println!("Starting non-mutating interactive seed generation for party {}", self.party_id);

        // Re-add variable definitions from previous step's context if needed
        // DPF key generation (placeholder)
        let dpf_domain_bits = self.lpn_params.k; // Use k for domain_bits
        let alpha = 0;
        let beta_value = vec![F2::one()];
        // Dpf::gen expects beta as u8 (0 or 1)
        let beta_u8 = if beta_value.is_empty() || beta_value[0].is_zero() { 0u8 } else { 1u8 };

        // Create a Dpf instance to call its gen method
        // Assuming the DPF works on F2 elements
        let dpf_instance = Dpf::<F2>::new(dpf_domain_bits);

        // Call the Dpf::gen METHOD
        let (k0, k1) = dpf_instance.gen(alpha, beta_u8)
                        .map_err(|e| PcgError::DpfError(e.to_string()))?;
        let (sender_dpf_key, receiver_dpf_key) = (k0, k1); // Assuming party IDs align this way

        // Other placeholder values
        let delta_field: F = F::rand(&mut thread_rng());
        let x_f2: Vec<F2> = (0..self.lpn_params.n).map(|_| F2::rand(&mut thread_rng())).collect();
        let x_field: Vec<F> = x_f2.iter().map(|f2| if f2.is_one() { F::one() } else { F::zero() }).collect();
        let s_delta_f2: Vec<F2> = (0..self.lpn_params.k).map(|_| F2::rand(&mut thread_rng())).collect();
        let s_delta_bytes = pack_f2_vector(&s_delta_f2)?;
        let h_matrix = self.lpn_params.generate_matrix().map_err(|e| PcgError::LpnError(e.to_string()))?;
        let h_transpose_matrix = h_matrix.transpose();

        // Define structs within this scope
        let sender_seed_struct = SvoleSenderSeed {
            k_dpf: sender_dpf_key.clone(),
            s_delta: s_delta_bytes.clone(),
            h_matrix: h_matrix.clone(),
            delta: delta_field.clone(),
        };
        let receiver_seed_struct = SvoleReceiverSeed {
            k_dpf: receiver_dpf_key.clone(),
            x: x_field,
            h_transpose_matrix: h_transpose_matrix.clone(),
            delta: delta_field, // Receiver seed now also has delta in pcg_core
        };

        // Wrap structs in PcgSeed enum
        let sender_seed = PcgSeed::SvoleSender(sender_seed_struct);
        let receiver_seed = PcgSeed::SvoleReceiver(receiver_seed_struct);

        // Return the PcgSeed variants
        Ok((sender_seed, receiver_seed))
    }
}

// Specify the associated type explicitly
impl<C: Channel<DpfKey> + Send + 'static, F: Field + Sync + Send + 'static + UniformRand + One + Zero + Debug + From<u64> + PrimeField>
    PcgSeedGenerator for InteractiveSeedGenerator<C, F>
{
    type Seed = PcgSeed<F>;

    // Change &mut self to &self to match trait
    // Qualify associated type Seed
    fn gen(&self, security_param: usize) -> Result<(<Self as PcgSeedGenerator>::Seed, <Self as PcgSeedGenerator>::Seed), PcgError> {
        println!("Starting non-mutating interactive seed generation for party {}", self.party_id);

        // Re-add variable definitions from previous step's context if needed
        // DPF key generation (placeholder)
        let dpf_domain_bits = self.lpn_params.k; // Use k for domain_bits
        let alpha = 0;
        let beta_value = vec![F2::one()];
        // Dpf::gen expects beta as u8 (0 or 1)
        let beta_u8 = if beta_value.is_empty() || beta_value[0].is_zero() { 0u8 } else { 1u8 };

        // Create a Dpf instance to call its gen method
        // Assuming the DPF works on F2 elements
        let dpf_instance = Dpf::<F2>::new(dpf_domain_bits);

        // Call the Dpf::gen METHOD
        let (k0, k1) = dpf_instance.gen(alpha, beta_u8)
                        .map_err(|e| PcgError::DpfError(e.to_string()))?;
        let (sender_dpf_key, receiver_dpf_key) = (k0, k1); // Assuming party IDs align this way

        // Other placeholder values
        let delta_field: F = F::rand(&mut thread_rng());
        let x_f2: Vec<F2> = (0..self.lpn_params.n).map(|_| F2::rand(&mut thread_rng())).collect();
        let x_field: Vec<F> = x_f2.iter().map(|f2| if f2.is_one() { F::one() } else { F::zero() }).collect();
        let s_delta_f2: Vec<F2> = (0..self.lpn_params.k).map(|_| F2::rand(&mut thread_rng())).collect();
        let s_delta_bytes = pack_f2_vector(&s_delta_f2)?;
        let h_matrix = self.lpn_params.generate_matrix().map_err(|e| PcgError::LpnError(e.to_string()))?;
        let h_transpose_matrix = h_matrix.transpose();

        // Define structs within this scope
        let sender_seed_struct = SvoleSenderSeed {
            k_dpf: sender_dpf_key.clone(),
            s_delta: s_delta_bytes.clone(),
            h_matrix: h_matrix.clone(),
            delta: delta_field.clone(),
        };
        let receiver_seed_struct = SvoleReceiverSeed {
            k_dpf: receiver_dpf_key.clone(),
            x: x_field,
            h_transpose_matrix: h_transpose_matrix.clone(),
            delta: delta_field, // Receiver seed now also has delta in pcg_core
        };

        // Wrap structs in PcgSeed enum
        let sender_seed = PcgSeed::SvoleSender(sender_seed_struct);
        let receiver_seed = PcgSeed::SvoleReceiver(receiver_seed_struct);

        // Return the PcgSeed variants
        Ok((sender_seed, receiver_seed))
    }
}

// --- Mpsc Channel Implementation for Testing ---
pub struct MpscChannel<T: Send + Clone + Debug> {
    sender: Sender<T>,
    receiver: Receiver<T>,
}

impl<T: Send + Clone + Debug> Channel<T> for MpscChannel<T> {
    fn send(&self, message: T) -> Result<(), PcgError> {
        self.sender.send(message).map_err(|e| PcgError::ChannelError(e.to_string()))
    }

    fn recv(&self) -> Result<T, PcgError> {
        self.receiver.recv().map_err(|e| PcgError::ChannelError(e.to_string()))
    }
}

// --- Mock Base OT for Testing ---
pub struct MockBaseOt {
    party_type: BaseOtType,
    // Add simple state if needed for mock interaction
}

impl MockBaseOt {
    pub fn new(party_type: BaseOtType) -> Self {
        Self { party_type }
    }
}

impl BaseOtOracle for MockBaseOt {
    fn send(&mut self, messages: &[(Vec<u8>, Vec<u8>)]) -> Result<(), PcgError> {
        if !matches!(self.party_type, BaseOtType::Sender) { return Err(PcgError::SeedGenError("Mock OT called as wrong type".to_string())) }
        println!("Mock OT Sender sending {} pairs.", messages.len());
        // In a real test, could use channels/shared state to simulate communication
        Ok(())
    }

    fn receive(&mut self, choices: &[u8]) -> Result<Vec<Vec<u8>>, PcgError> {
         if !matches!(self.party_type, BaseOtType::Receiver) { return Err(PcgError::SeedGenError("Mock OT called as wrong type".to_string())) }
        println!("Mock OT Receiver receiving for {} choices.", choices.len());
        // Return dummy messages based on choices
        let result = choices.iter().map(|c| vec![*c; 16]).collect(); // Dummy 16-byte messages
        Ok(result)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::lpn::{LpnParameters, CodeType};
    use ark_test_curves::bls12_381::Fq as TestField; // Use a concrete field
    use std::thread;
    use std::marker::PhantomData; 
    use ark_ff::{Field as ArkField, Zero, One};
    use crate::primitives::field::F2; // Import F2 for Default impl

    #[test]
    fn test_interactive_seed_gen_with_mpsc_channel() {
        let lpn_params = LpnParameters {
            n: 64, k: 32, t: 5, code_type: CodeType::RandomLinear
        };

        // Fix: Use new_pair_mpsc which creates the channels internally
        let (gen0, gen1) = 
            InteractiveSeedGenerator::<MpscChannel<DpfKey>, TestField>::new_pair_mpsc(lpn_params);

        // Spawn threads for each party
        let handle0 = thread::spawn(move || {
            println!("Party 0 starting seed generation...");
            let seed0 = gen0.generate_local_seed();
            println!("Party 0 finished seed generation: {:?}", seed0.is_ok());
            seed0
        });

        let handle1 = thread::spawn(move || {
            println!("Party 1 starting seed generation...");
            let seed1 = gen1.generate_local_seed();
            println!("Party 1 finished seed generation: {:?}", seed1.is_ok());
            seed1
        });

        // Wait for threads and get results
        let result0 = handle0.join().unwrap();
        let result1 = handle1.join().unwrap();

        assert!(result0.is_ok(), "Party 0 seed generation failed: {:?}", result0.err());
        assert!(result1.is_ok(), "Party 1 seed generation failed: {:?}", result1.err());

        let seed0 = result0.unwrap();
        let seed1 = result1.unwrap();

        assert!(matches!(seed0, SvoleSeed::Sender(_)), "Party 0 has wrong seed type");
        assert!(matches!(seed1, SvoleSeed::Receiver(_)), "Party 1 has wrong seed type");

        if let (SvoleSeed::Sender(s0), SvoleSeed::Receiver(s1)) = (seed0, seed1) {
            assert_eq!(s0.delta, s1.delta, "Delta mismatch between seeds");
            println!("Seed generation delta consistency check passed.");
        } else {
            panic!("Seed type error after match");
        }
    }

    // Fix: DpfKey Default impl uses correct fields
    impl Default for DpfKey {
        fn default() -> Self {
            DpfKey {
                s0: vec![0u8; 16], 
                cw: Vec::new(),
                last_cw: vec![F2::zero()], 
            }
        }
    }
}

// Placeholder packing function for generic field F
// Required by previous logic, keep placeholder
fn pack_field_element<F: Field + Debug>(f: &F) -> Result<Vec<u8>, PcgError> {
    // Very basic placeholder using Debug print - REPLACE with proper serialization
    println!("Warning: Using placeholder pack_field_element");
    Ok(format!("{:?}", f).into_bytes())
}

// Placeholder unpack function for F2 (already added)
// fn unpack_f2_vector(...) -> Result<Vec<F2>, PcgError> { ... }

// Placeholder pack function for F2 (already added)
// fn pack_f2_vector(...) -> Result<Vec<u8>, PcgError> { ... }
