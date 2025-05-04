use ark_std::vec::Vec;

/// Errors that can occur during PCG operations.
#[derive(Debug)]
pub enum PcgError {
    SeedGenError(String),
    ExpandError(String),
    // Add more specific errors as needed
}

/// Trait for the PCG Seed Generator (Gen).
///
/// Takes security parameters and correlation-specific parameters,
/// and outputs a pair of seeds (k₀, k₁) for the two parties.
pub trait PcgSeedGenerator {
    type Seed0;
    type Seed1;
    // Add associated types for correlation parameters if needed

    fn gen(
        security_param: usize, // e.g., lambda
        // Add correlation_params here...
    ) -> Result<(Self::Seed0, Self::Seed1), PcgError>;
}

/// Trait for the PCG Expander (Expand).
///
/// Takes a party's index (σ ∈ {0, 1}) and their seed (k_σ),
/// and outputs their share of the long (pseudo)random correlated output (r_σ).
pub trait PcgExpander {
    type Seed;
    type Output;
    // Add associated types for correlation parameters if needed

    fn expand(
        party_index: u8, // σ = 0 or 1
        seed: &Self::Seed,
        // Add correlation_params here...
    ) -> Result<Self::Output, PcgError>;
}
