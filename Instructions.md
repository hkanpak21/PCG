Project Requirement Document (PRD): PCG & Silent OT Implementation
This document outlines the requirements for implementing Pseudorandom Correlation Generators (PCGs) and specifically the Silent Oblivious Transfer (Silent OT) extension functionality based on the paper "Efficient Pseudorandom Correlation Generators: Silent OT Extension and More".
1. Introduction
1.1 Purpose: To define the requirements for a software library implementing efficient PCGs, focusing on the Silent OT correlation using LPN-based techniques described in the reference paper. This library will serve as a core component for building advanced MPC protocols with silent preprocessing.
1.2 Scope:
Implementation of the core PCG framework (Gen, Expand).
Implementation of the sVOLE correlation PCG based on Dual LPN and DPFs.
Implementation of the transformation from sVOLE PCG output to Random OT output (including Δ-OT step and CRHF).
Implementation of an interactive protocol for secure generation of the sVOLE PCG seeds (leveraging DPF setup).
Underlying cryptographic primitives (DPF, CRHF, potentially finite field arithmetic).
1.3 Target Audience: Cryptographic engineers, software developers, AI agent tasked with code generation/implementation.
1.4 Reference: Boyle et al., "Efficient Pseudorandom Correlation Generators: Silent OT Extension and More" [BCG+19].
2. Goals
Implement a reusable library for generating correlations via PCGs.
Provide a high-performance implementation of Silent Random OT extension.
Achieve communication complexity for ROT generation significantly lower than traditional OT extension (sublinear in the number of OTs).
Ensure the implementation is secure under the assumptions stated in the paper (LPN, CRHF, PRG security, semi-honest adversaries for seed generation initially).
Provide clear APIs for integration into larger MPC systems.
3. Non-Goals
Implementation of a complete MPC protocol using the Silent OT library.
Malicious security for the interactive seed generation protocol (initially focus on semi-honest security as in [Ds17]). The expansion phase is inherently secure against malicious parties due to lack of interaction.
Graphical User Interface (GUI).
Support for all PCG types mentioned in the paper (e.g., lattice-based, Beaver triples initially out of scope unless specified).
4. Functional Requirements
FR1: PCG Core Framework
FR1.1: Implement an abstract interface/trait for a PCG Seed Generator (Gen).
Input: Security parameter λ, correlation-specific parameters.
Output: Pair of seeds (k₀, k₁) or error.
FR1.2: Implement an abstract interface/trait for a PCG Expander (Expand).
Input: Party index σ ∈ {0, 1}, seed k_σ, correlation-specific parameters.
Output: Long (pseudo)random correlated output r_σ or error.
FR2: Primitives
FR2.1: Distributed Point Function (DPF)
Implement DPF Gen(1^λ, point, value) -> (K₀, K₁) based on a standard PRG (e.g., AES-based). Parameterize for domain size N and output group/field (e.g., F_q).
Implement DPF FullEval(σ, K_σ) -> vector (evaluates DPF on all domain points).
Dependency: Secure PRG (e.g., AES-CTR).
FR2.2: Correlation-Robust Hash Function (CRHF)
Implement an interface for CRHF H(i, input) -> output_string.
Provide instantiation using a fixed-key block cipher (e.g., AES) modeled as a random permutation, or potentially SHA-3/Blake2 for practical purposes if analyzed carefully. Parameterize input/output sizes.
FR2.3: LPN Parameters & Codes
Allow configuration of Dual LPN parameters: n, n', noise weight t, field F_p (specifically F₂ for OT).
Implement generation of the LPN matrix H (transpose of parity check or generator matrix) based on chosen code type:
Random Linear Code (for baseline).
Quasi-Cyclic Code.
LDPC Code (specify sparsity d).
FR2.4: Finite Field Arithmetic
Requires arithmetic in F_q = F_{p^r} (specifically F_{2^r} for OT, where r is string OT length, e.g., 128). Needs addition, multiplication.
FR2.5: Matrix/Vector Operations
Implement efficient multiplication of vectors by the LPN matrix H (and potentially its transpose depending on LPN formulation/code choice). Optimize for sparse H if LDPC is used.
FR3: Subfield VOLE (sVOLE) PCG
FR3.1: Implement sVOLE_PCG::Gen based on Fig 3.
Input: λ, n, n', t, p, q, code C.
Internally uses DPF Gen.
Output: Seeds (k₀, k₁) containing DPF keys and associated randomness S, y, x.
FR3.2: Implement sVOLE_PCG::Expand based on Fig 3.
Input: σ, seed k_σ.
Internally uses DPF FullEval, spread function, matrix multiplication with H, field arithmetic.
Output: sVOLE share (u, v) if σ=0, or (x, w) if σ=1.
FR4: Random OT PCG (from sVOLE)
FR4.1: Implement wrapper ROT_PCG::Gen that calls sVOLE_PCG::Gen with p=2 and appropriate q=2^r.
FR4.2: Implement wrapper ROT_PCG::Expand that:
Calls sVOLE_PCG::Expand with p=2.
Performs the role switch (sVOLE sender -> OT receiver, sVOLE receiver -> OT sender) as described in Sec 5.1.1.
Applies the CRHF H locally to the Δ-OT outputs to produce the final Random OT shares (uᵢ, vᵢ) for receiver and {(w_{i,j})}_{i,j} for sender, as in Fig 4 / Sec 5.2.
FR5: Interactive Seed Generation
FR5.1: Implement the interactive 2-party protocol to securely compute the sVOLE_PCG::Gen function.
Leverage existing protocols for secure DPF key generation (e.g., based on [Ds17] which uses base OTs).
Input: Security parameter λ, sVOLE parameters, required number of base OTs.
Output: Each party receives their respective sVOLE PCG seed k_σ.
Dependency: Secure implementation of k base OTs (can be assumed as input for the library).
5. Non-Functional Requirements
NFR1: Performance:
Expand should be highly optimized, targeting >1 million 128-bit OTs per second per core (as per paper estimates). Measure and profile.
Seed generation communication cost should be minimized, aiming for < 5 bits per resulting OT (amortized, excluding one-time base OT cost).
Seed sizes should be minimized (< 10 KB for n=1M OTs).
NFR2: Security:
Target 128-bit computational security.
Must be secure against semi-honest adversaries (for both Expand and Seed Generation).
Cryptographic primitives (PRG, Hash, FF arithmetic) must use secure, constant-time implementations where applicable.
Security relies on the hardness of the specified Dual LPN variant and the security of the DPF/PRG/CRHF.
NFR3: API Design:
Provide a clear, concise, and well-documented API for all public modules (PCG Gen/Expand, Seed Gen Protocol).
API should be thread-safe where appropriate (e.g., Expand should be callable in parallel).
NFR4: Platform: Target modern 64-bit Linux systems. x86-64 architecture with AES-NI and relevant vector extensions (SSE, AVX) for performance.
NFR5: Code Quality: Code should be modular, maintainable, include unit tests for all components, and integration tests for the end-to-end OT generation.
6. Assumptions & Dependencies
Hardness of the specific Dual LPN variant chosen (e.g., Regular-LPN or LPN w/ sparse codes).
Security of the underlying PRG (e.g., AES-CTR).
Correlation-robustness property of the chosen hash function/instantiation.
Availability of secure base OT implementation for the interactive seed generation phase.
Availability of robust libraries for large finite field arithmetic (e.g., F_{2^128}) and potentially elliptic curve crypto (if base OTs need implementing).
7. Implementation Guidance: Language & Libraries
Language:
Rust: A strong choice due to its focus on safety, performance, excellent package management (Cargo), and growing ecosystem for cryptography. Many modern crypto projects use Rust.
C++: Also a viable choice, offering high performance. It has a mature crypto ecosystem, but requires more careful memory management.
Recommendation: Rust is likely preferable if starting fresh, due to safety guarantees and ecosystem momentum. C++ is suitable if integrating with existing C++ systems or if the team has deep C++ expertise.
Libraries: Leverage existing high-quality, audited libraries wherever possible. Do not re-implement standard primitives unless absolutely necessary.
Hashing/Symmetric Crypto: Standard libraries (like Rust's sha3, blake2, aes crates) or OpenSSL/LibSodium in C++.
Finite Field Arithmetic: This is critical. Look for libraries optimized for binary extension fields (F_{2^r}).
Rust: Crates like ff, ark-ff (part of arkworks), potentially Galois crate.
C++: Libraries like NTL, mcl.
Elliptic Curves (for Base OT):
Rust: ark-ec, curve25519-dalek, k256.
C++: mcl, RELIC toolkit.
DPF/OT Extension:
Look for existing implementations. Libraries like libOTe (C++), emp-tool (C++) contain OT extension and related primitives. Search for Rust implementations of DPFs (might be less common or research-grade). Implementing DPFs from scratch based on AES is feasible but requires care.
LPN Code Matrices: Standard matrix libraries might work, but specialized implementations for LDPC encoding/transpose might be needed for NFR1.
8. Success Metrics
Successful generation of correlated outputs passing statistical randomness/correlation tests.
End-to-end generation of Random OTs passing standard security simulation tests (if built).
Performance benchmarks for Expand meeting or exceeding targets based on paper estimates (Table 4 / Table 9).
Measured communication cost for seed generation meeting targets (Table 3).
Successful code review and potentially external security audit.
9. Open Questions / Future Work
Malicious security hardening for the interactive seed generation protocol.
Implementation of PCGs for other correlations (Beaver Triples, etc.).
Exploration of different LPN code families (performance/security trade-offs).
Integration with a full MPC framework.
Optimized implementation of LDPC matrix operations if chosen.
This PRD provides a detailed starting point. An AI agent or development team would use this to understand the scope, requirements, and dependencies for building the Silent OT functionality.