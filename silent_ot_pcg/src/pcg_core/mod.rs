use serde::{Serialize, Deserialize};
use std::fmt::Debug;
use crate::primitives::dpf::DpfKey;
use crate::primitives::lpn::LpnMatrix;
use crate::primitives::field::{Field128, F2};
use ark_std::vec::Vec;
use ark_ff::Field;
use std::marker::PhantomData;

/// Errors that can occur during PCG operations.
#[derive(Debug)]
pub enum PcgError {
    SeedGenError(String),
    ExpandError(String),
    DpfError(String),
    LpnError(String),
    ChannelError(String),
    InvalidPartyIndex(String),
    MissingParameter(String),
    Custom(String),
    SvoleError(String),
    IoError(std::io::Error),
    FieldMismatch(String),
    InvalidInput(String),
    CrhfError(String),
    SerializationError(String),
    NotImplemented(String),
    Other(String),
    // Add more specific errors as needed
}

/// Trait for the PCG Seed Generator (Gen).
///
/// Takes security parameters and correlation-specific parameters,
/// and outputs a pair of seeds (k₀, k₁) for the two parties.
pub trait PcgSeedGenerator {
    type Seed: Send + Sync + Clone + Debug;
    // Add associated types for correlation parameters if needed

    fn gen(
        &self,
        security_param: usize, // e.g., lambda
        // Add correlation_params here...
    ) -> Result<(Self::Seed, Self::Seed), PcgError>;
}

/// Trait for the PCG Expander (Expand).
///
/// Takes a party's index (σ ∈ {0, 1}) and their seed (k_σ),
/// and outputs their share of the long (pseudo)random correlated output (r_σ).
pub trait PcgExpander {
    type Seed: Send + Sync + Clone + Debug;
    type Output: Send + Sync + Debug;
    // Add associated types for correlation parameters if needed

    fn expand(
        &self,
        party_index: u8, // σ = 0 or 1
        seed: &Self::Seed,
        // Add correlation_params here...
    ) -> Result<Self::Output, PcgError>;
}

/// Seed structure for the sVOLE sender (P0)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SvoleSenderSeed {
    pub k_dpf: DpfKey,
    pub s_delta: Vec<u8>,
    pub h_matrix: LpnMatrix,
    pub delta: Field128,
}

/// Seed structure for the sVOLE receiver (P1)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SvoleReceiverSeed {
    pub k_dpf: DpfKey,
    pub x: Vec<F2>,
    pub h_transpose_matrix: LpnMatrix,
    pub delta: Field128,
}

/// Enum to hold either sender or receiver seed
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SvoleSeed {
    Sender(SvoleSenderSeed),
    Receiver(SvoleReceiverSeed),
}

/// Generic PCG Seed wrapper
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum PcgSeed<F: Field + Debug> {
    SvoleSender(SvoleSenderSeed),
    SvoleReceiver(SvoleReceiverSeed),
    Placeholder,
    _Marker(PhantomData<F>),
}

/// Generic PCG Output wrapper
#[derive(Debug, PartialEq, Clone)]
pub enum PcgOutput<F: Field + Debug + PartialEq> {
    SvoleSenderOutput(crate::svole::SvoleSenderOutput),
    SvoleReceiverOutput(crate::svole::SvoleReceiverOutput),
    Placeholder,
    _Marker(PhantomData<F>),
}
