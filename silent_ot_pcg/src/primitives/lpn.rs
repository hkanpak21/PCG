use ark_ff::Field; // Although for OT we need F_2, keep general for now
use ark_std::vec::Vec;
use nalgebra::{DMatrix, DVector};
use nalgebra_sparse::csr::CsrMatrix; // For sparse representation (LDPC)
use rand::{Rng, thread_rng};
use crate::primitives::field::Field128; // Import the F_q type
use std::ops::Add; // For Field128 addition

/// LPN code types mentioned in the PRD.
pub enum CodeType {
    RandomLinear,
    QuasiCyclic, // Requires more specific parameters
    Ldpc { sparsity: usize }, // Low-Density Parity-Check
}

/// Parameters for the Learning Parity with Noise (LPN) assumption.
/// Defines the properties of the code used.
pub struct LpnParameters {
    /// Codeword length (output dimension of the generator matrix G).
    /// Corresponds to `n` in the Dual LPN context (Fig 3).
    pub n: usize,
    /// Message length (input dimension of G, output dimension of H).
    /// Corresponds to `k` (often implicitly defined by `n` and code rate).
    /// The matrix H used in Dual LPN is k x n.
    pub k: usize,
    /// Noise weight `t` (Hamming weight of the error vector `e`).
    pub t: usize,
    /// Type of code used to generate the matrix H.
    pub code_type: CodeType,
    // Field F_p (specifically F_2 for OT)
    // We might represent F_2 elements as bools or u8s.
}

/// Represents the LPN matrix H (k x n).
/// Can be dense or sparse.
pub enum LpnMatrix {
    Dense(DMatrix<u8>), // Using u8 for F_2 elements (0 or 1)
    Sparse(CsrMatrix<u8>), // CSR format for sparse matrices
}

impl LpnParameters {
    /// Generates the LPN matrix H based on the specified code type.
    /// H is a k x n matrix.
    ///
    /// Note: This is a placeholder. Actual code generation is complex,
    /// especially for structured codes like QC or LDPC.
    pub fn generate_matrix(&self) -> Result<LpnMatrix, &'static str> {
        println!(
            "Warning: LPN matrix generation is a placeholder. Using random matrix for now."
        );

        match self.code_type {
            CodeType::RandomLinear => {
                // Generate a random k x n matrix over F_2
                let mut rng = thread_rng();
                let matrix = DMatrix::from_fn(self.k, self.n, |_, _| rng.gen_range(0..=1));
                Ok(LpnMatrix::Dense(matrix))
            }
            CodeType::QuasiCyclic => {
                // TODO: Implement Quasi-Cyclic code generation
                Err("Quasi-Cyclic code generation not implemented")
            }
            CodeType::Ldpc { sparsity } => {
                // TODO: Implement LDPC code generation (guaranteeing sparsity)
                // This often involves graph-based methods (e.g., Gallager's construction)
                // For now, generating a random sparse matrix as a *very* rough placeholder.

                let density = sparsity as f64 / self.n as f64; // Rough density target per row
                let mut csrmat = CsrMatrix::<u8>::zeros(self.k, self.n);
                let mut rng = thread_rng();

                // Naive random sparse generation (not proper LDPC construction)
                for r in 0..self.k {
                    for _ in 0..sparsity {
                        let c = rng.gen_range(0..self.n);
                        // This doesn't guarantee exactly `sparsity` non-zeros per row/col
                        // or handle duplicates well, just a quick placeholder.
                        *csrmat.entry_mut(r, c) = 1;
                    }
                }
                 Ok(LpnMatrix::Sparse(csrmat))
                // Err("LDPC code generation not implemented")
            }
        }
    }
}

// Placeholder for matrix-vector multiplication optimized for F_2
// and potentially leveraging sparsity.
pub fn matrix_vector_multiply_f2(
    matrix: &LpnMatrix,
    vector: &DVector<u8>,
) -> Result<DVector<u8>, &'static str> {
    match matrix {
        LpnMatrix::Dense(h) => {
            if h.ncols() != vector.nrows() {
                return Err("Matrix(F2) and vector dimensions mismatch");
            }
            // Naive multiplication, assuming elements are 0 or 1 (F_2)
            // Result y = H * x (mod 2)
            let result = h * vector;
            // Apply modulo 2
            Ok(result.map(|val| val % 2))
        }
        LpnMatrix::Sparse(h) => {
            if h.ncols() != vector.nrows() {
                return Err("Matrix(F2 sparse) and vector dimensions mismatch");
            }
            // Use sparse matrix multiplication
            let result = h * vector;
            // Apply modulo 2
            Ok(result.map(|val| val % 2))
        }
    }
}

/// Matrix-vector multiplication: y = H^T * x
/// H^T: n x k (F_2)
/// x: k x 1 (F_q = Field128)
/// y: n x 1 (F_q = Field128)
/// Computes y_j = sum_{i where H_{i,j}=1} x_i
pub fn matrix_transpose_vector_multiply_fq(
    matrix: &LpnMatrix, // H (k x n)
    vector: &DVector<Field128>, // x (k x 1)
) -> Result<DVector<Field128>, &'static str> {
    let k = matrix.nrows();
    let n = matrix.ncols();

    if k != vector.nrows() {
        return Err("Matrix(H^T) and vector(Fq) dimensions mismatch");
    }

    let mut result = DVector::<Field128>::from_element(n, Field128::ZERO);

    match matrix {
        LpnMatrix::Dense(h) => {
            // Iterate through columns of H (rows of H^T)
            for j in 0..n {
                let mut sum_j = Field128::ZERO;
                // Iterate through rows of H (columns of H^T)
                for i in 0..k {
                    if h[(i, j)] == 1 {
                        sum_j = sum_j.add(vector[i]); // Add x_i
                    }
                }
                result[j] = sum_j;
            }
        }
        LpnMatrix::Sparse(h) => {
            // Iterate through the non-zero entries of H
            // h is k x n (CSR format: rows, cols, values)
            for i in 0..k { // Iterate rows of H
                 let row = h.row(i);
                 for (j, val) in row.col_indices().iter().zip(row.values().iter()) {
                     // H_{i,j} = val (which is 1 for F2)
                     if *val == 1 {
                         // Add x_i to result[j]
                         result[*j] = result[*j].add(vector[i]);
                     }
                 }
            }
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    // Helper to get matrix dimensions regardless of type
    impl LpnMatrix {
        fn nrows(&self) -> usize {
            match self {
                LpnMatrix::Dense(m) => m.nrows(),
                LpnMatrix::Sparse(m) => m.nrows(),
            }
        }
        fn ncols(&self) -> usize {
            match self {
                LpnMatrix::Dense(m) => m.ncols(),
                LpnMatrix::Sparse(m) => m.ncols(),
            }
        }
    }

    #[test]
    fn test_random_linear_matrix_gen() {
        let params = LpnParameters {
            n: 128, // Codeword length
            k: 64,  // Message length
            t: 10,  // Noise weight (not used in matrix gen)
            code_type: CodeType::RandomLinear,
        };

        let matrix_res = params.generate_matrix();
        assert!(matrix_res.is_ok());
        let matrix = matrix_res.unwrap();

        assert_eq!(matrix.nrows(), params.k);
        assert_eq!(matrix.ncols(), params.n);
        if let LpnMatrix::Dense(h) = matrix {
            assert!(h.iter().all(|&x| x == 0 || x == 1));
        } else {
            panic!("Expected Dense matrix");
        }
    }

    #[test]
    fn test_ldpc_matrix_gen_placeholder() {
        let sparsity = 5;
        let params = LpnParameters {
            n: 256,
            k: 128,
            t: 10,
            code_type: CodeType::Ldpc { sparsity },
        };

        let matrix_res = params.generate_matrix();
         assert!(matrix_res.is_ok());
         let matrix = matrix_res.unwrap();

        assert_eq!(matrix.nrows(), params.k);
        assert_eq!(matrix.ncols(), params.n);
        if let LpnMatrix::Sparse(h) = matrix {
            println!("Placeholder LDPC matrix NNZ: {}", h.nnz());
             assert!(h.nnz() <= sparsity * params.k); // Should be roughly this
             assert!(h.nnz() > 0);
        } else {
            panic!("Expected Sparse matrix for LDPC placeholder");
        }
    }

    #[test]
    fn test_matrix_vector_mult_f2() {
        let params = LpnParameters {
            n: 8,
            k: 4,
            t: 1,
            code_type: CodeType::RandomLinear,
        };
        let lpn_mat = params.generate_matrix().unwrap();
        let vector = DVector::from_vec(vec![1, 0, 1, 1, 0, 1, 0, 1]); // n x 1

        // Check y = H * x (k x 1 result)
        let result_res = matrix_vector_multiply_f2(&lpn_mat, &vector);
        assert!(result_res.is_ok());
        let result = result_res.unwrap();

        assert_eq!(result.nrows(), params.k);
        assert!(result.iter().all(|&x| x == 0 || x == 1));
        println!("Mat(F2)-Vec(F2) Mult Result (y=Hx): {:?}", result.transpose());

        // Example with sparse (placeholder)
        let sparse_params = LpnParameters {
            n: 8,
            k: 4,
            t: 1,
            code_type: CodeType::Ldpc { sparsity: 2 },
        };
         let sparse_mat = sparse_params.generate_matrix().unwrap();
         let sparse_res = matrix_vector_multiply_f2(&sparse_mat, &vector);
         assert!(sparse_res.is_ok());
         let sparse_result = sparse_res.unwrap();
         assert_eq!(sparse_result.nrows(), sparse_params.k);
         assert!(sparse_result.iter().all(|&x| x == 0 || x == 1));
         println!("Sparse Mat(F2)-Vec(F2) Mult Result (y=Hx): {:?}", sparse_result.transpose());
    }

    #[test]
    fn test_matrix_transpose_vector_mult_fq() {
        let mut rng = thread_rng();
        let params = LpnParameters {
            n: 8, // H is k x n, so H^T is n x k
            k: 4,
            t: 1,
            code_type: CodeType::RandomLinear,
        };
        let lpn_mat = params.generate_matrix().unwrap(); // H (k x n)
        let vector_fq: DVector<Field128> = DVector::from_fn(params.k, |_, _| Field128::from(rng.gen::<u128>())); // x (k x 1)

        // Check y = H^T * x (n x 1 result)
        let result_res = matrix_transpose_vector_multiply_fq(&lpn_mat, &vector_fq);
        assert!(result_res.is_ok());
        let result = result_res.unwrap();

        assert_eq!(result.nrows(), params.n);

        // Manual check for small case (optional)
        // ... create small H, x, calculate expected y, compare ...

        println!("Mat(H^T)-Vec(Fq) Mult Result (y=H^T x): {} rows", result.nrows());

        // Example with sparse (placeholder)
         let sparse_params = LpnParameters {
            n: 8,
            k: 4,
            t: 1,
            code_type: CodeType::Ldpc { sparsity: 2 },
        };
         let sparse_mat = sparse_params.generate_matrix().unwrap(); // H (k x n)
         let sparse_vector_fq: DVector<Field128> = DVector::from_fn(sparse_params.k, |_, _| Field128::from(rng.gen::<u128>())); // x (k x 1)
         let sparse_res = matrix_transpose_vector_multiply_fq(&sparse_mat, &sparse_vector_fq);
         assert!(sparse_res.is_ok());
         let sparse_result = sparse_res.unwrap();
         assert_eq!(sparse_result.nrows(), sparse_params.n);
         println!("Sparse Mat(H^T)-Vec(Fq) Mult Result (y=H^T x): {} rows", sparse_result.nrows());
    }
}
