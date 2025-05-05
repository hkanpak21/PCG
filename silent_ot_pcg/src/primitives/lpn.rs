use crate::primitives::field::{Field128, F2};
use nalgebra::{DMatrix, DVector};
use nalgebra_sparse::csr::CsrMatrix; // For sparse representation (LDPC)
use nalgebra_sparse::SparseEntryMut;
use nalgebra_sparse::SparseEntry;
use rand::{Rng, thread_rng};
// use std::ops::Add; // For Field128 addition
use ark_ff::{Field, Zero, One}; // Import Zero and One for F2
use serde::{Serialize, Deserialize}; // Add serde imports

/// LPN code types mentioned in the PRD.
#[derive(Clone, Debug)]
pub enum CodeType {
    RandomLinear,
    QuasiCyclic, // Requires more specific parameters
    Ldpc { sparsity: usize }, // Low-Density Parity-Check
}

/// Parameters for the Learning Parity with Noise (LPN) assumption.
/// Defines the properties of the code used.
#[derive(Clone, Debug)]
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
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum LpnMatrix {
    Dense(DMatrix<u8>), // Using u8 for F_2 elements (0 or 1)
    Sparse(CsrMatrix<u8>), // CSR format for sparse matrices
    Empty, // Added for the new matrix_vector_multiply_f2 function
}

impl LpnMatrix {
    pub fn nrows(&self) -> usize {
        match self {
            LpnMatrix::Dense(m) => m.nrows(),
            LpnMatrix::Sparse(m) => m.nrows(),
            LpnMatrix::Empty => 0,
        }
    }
    pub fn ncols(&self) -> usize {
        match self {
            LpnMatrix::Dense(m) => m.ncols(),
            LpnMatrix::Sparse(m) => m.ncols(),
            LpnMatrix::Empty => 0,
        }
    }
    pub fn transpose(&self) -> Self {
        match self {
            LpnMatrix::Sparse(m) => {
                let dense = DMatrix::<u8>::from_fn(m.nrows(), m.ncols(), |r, c| {
                    m.get_entry(r, c).map(|e| *e.value()).unwrap_or(0)
                });
                LpnMatrix::Dense(dense.transpose())
            },
            LpnMatrix::Dense(m) => LpnMatrix::Dense(m.transpose()),
            LpnMatrix::Empty => LpnMatrix::Empty, // Handle Empty case
        }
    }
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
                        // Use get_entry_mut and insert
                        match csrmat.get_entry_mut(r, c) {
                            Some(SparseEntryMut::NonZero(mut entry)) => {
                                *entry = (*entry ^ 1); // Toggle existing entry
                            }
                            Some(SparseEntryMut::Zero(_entry_ref)) => {
                                // If zero exists explicitly, toggle it
                                // entry_ref.insert(1); // This syntax was wrong before
                                // The API might not allow modifying Zero entry directly
                                // Easiest is often to remove and re-insert or use NonZero logic
                                csrmat.insert(r, c, 1); // Overwrite/insert 1
                            }
                            None => {
                                csrmat.insert(r, c, 1); // Insert 1 if no entry
                            }
                        }
                    }
                }
                 Ok(LpnMatrix::Sparse(csrmat))
                // Err("LDPC code generation not implemented")
            }
        }
    }

    // Hypothetical function to insert into a sparse triplet structure if needed
    // (This function replaces the context where the original error occurred)
    // Modify this based on actual sparse matrix construction logic
    fn insert_sparse_entry(&self, triplets: &mut CsrMatrix<u8>, i: usize, idx: usize, value: u8) {
        // Use get_entry_mut and insert
        match triplets.get_entry_mut(i, idx) {
            Some(SparseEntryMut::NonZero(mut entry)) => {
                *entry = (*entry ^ value); // XOR with existing non-zero
            }
            Some(SparseEntryMut::Zero(_)) => {
                triplets.insert(i, idx, value); // Overwrite/insert value
            }
            None => {
                triplets.insert(i, idx, value); // Insert value if no entry
            }
        }
    }
}

// Placeholder for matrix-vector multiplication optimized for F_2
// and potentially leveraging sparsity.
pub fn matrix_vector_multiply_f2(
    matrix: &LpnMatrix,
    vector: &[F2],
) -> Result<Vec<F2>, &'static str> {
    match matrix {
        LpnMatrix::Dense(h) => {
            if h.ncols() != vector.len() {
                return Err("Matrix(F2) and vector dimensions mismatch");
            }
            // Convert F2 vector to DVector<u8> for nalgebra
            let vec_nalgebra = DVector::from_iterator(vector.len(),
                 vector.iter().map(|f| if f.is_one() { 1u8 } else { 0u8 } )
             );
            let result_nalgebra = h * vec_nalgebra;
            // Convert result back to Vec<F2>
            Ok(result_nalgebra.iter().map(|&byte| F2::from(byte % 2)).collect())
        }
        LpnMatrix::Sparse(h) => {
            if h.ncols() != vector.len() {
                return Err("Matrix(F2 sparse) and vector dimensions mismatch");
            }
            // Convert F2 vector to DVector<u8> for nalgebra sparse
            let vec_nalgebra = DVector::from_iterator(vector.len(),
                vector.iter().map(|f| if f.is_one() { 1u8 } else { 0u8 } )
            );
            // The result is a Dense vector (DVector)
            let result_nalgebra: DVector<u8> = h * vec_nalgebra; // Type annotation clarifies
            // Iterate over the dense result vector
            Ok(result_nalgebra.iter().map(|&byte| F2::from(byte % 2)).collect())
        }
        LpnMatrix::Empty => {
            Err("Empty matrix encountered in matrix_vector_multiply_f2")
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
    vector: &[Field128], // x (k x 1)
) -> Result<Vec<Field128>, &'static str> {
    let k = matrix.nrows();
    let n = matrix.ncols();

    if k != vector.len() {
        return Err("Matrix(H^T) and vector(Fq) dimensions mismatch");
    }

    let mut result = vec![Field128::zero(); n]; // Use Field::zero()

    match matrix {
        LpnMatrix::Dense(h) => {
            for j in 0..n {
                let mut sum_j = Field128::zero(); // Use Field::zero()
                for i in 0..k {
                    if h[(i, j)] == 1 {
                        sum_j += vector[i]; // Use += operator
                    }
                }
                result[j] = sum_j;
            }
        }
        LpnMatrix::Sparse(h) => {
            for i in 0..k { // Iterate rows of H
                 let row = h.row(i);
                 for (j, val) in row.col_indices().iter().zip(row.values().iter()) {
                     if *val == 1 {
                         result[*j] += vector[i]; // Use += operator
                     }
                 }
            }
        }
        LpnMatrix::Empty => {
            return Err("Empty matrix encountered in matrix_transpose_vector_multiply_fq");
        }
    }
    Ok(result)
}

pub fn matrix_vector_multiply_fq(
     matrix: &LpnMatrix,
     vector: &[Field128],
 ) -> Result<Vec<Field128>, &'static str> {
    let n = matrix.nrows(); // H is kxn, result is kx1
    let k = matrix.ncols();

    if k != vector.len() {
        return Err("Matrix(Fq H) and vector(Fq) dimensions mismatch");
    }

    let mut result = vec![Field128::zero(); n]; // Use Field::zero()

     match matrix {
         LpnMatrix::Dense(h) => {
            for r in 0..n { // Output rows = n
                 let mut row_sum = Field128::zero(); // Use Field::zero()
                 for c in 0..k { // Input cols = k
                      let val = Field128::from(h[(r, c)] as u128);
                      row_sum += val * vector[c];
                  }
                  result[r] = row_sum;
              }
         }
         LpnMatrix::Sparse(h) => {
            for r in 0..n { // Output rows = n
                 let mut row_sum = Field128::zero(); // Use Field::zero()
                 let row_view = h.row(r);
                 for (c, val_ref) in row_view.col_indices().iter().zip(row_view.values().iter()) {
                    let val = Field128::from(*val_ref as u128);
                    row_sum += val * vector[*c]; // c corresponds to index in vector
                 }
                 result[r] = row_sum;
             }
         }
         LpnMatrix::Empty => {
             return Err("Empty matrix in matrix_vector_multiply_fq");
         }
     }
    Ok(result)
}

pub fn process_sparse_matrix(m: &CsrMatrix<u8>) {
    // ...
    for (r, c, entry_option) in m.iter() { // Use iter() instead of triplet_iter maybe?
        if let Some(entry) = entry_option {
            // Original line: let value = m.get_entry(r, c).map(|e: SparseEntry<'_, u8>| *e.value()).unwrap_or(&0);
            // Change value() to into_value()
            let value = entry.into_value(); // Assuming into_value returns value or ref
            println!("Entry ({}, {}): {}", r, c, value);
        }
    }
    // Correct way to iterate triplets and use into_value
    for (r, c, val_ref) in m.triplet_iter() {
         let value = *val_ref; // Triplet iter gives direct value reference
         // If using get_entry within loop:
         let _value_from_get = m.get_entry(r, c).map(|e: SparseEntry<'_, u8>| *e.into_value()).unwrap_or(&0);
         println!("Triplet ({}, {}): {}", r, c, value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use crate::primitives::field::Field128;
    use ark_ff::{Field, One, Zero}; // Ensure Field is imported here too

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
        let vector: Vec<F2> = vec![
            F2::one(), F2::zero(), F2::one(), F2::one(),
            F2::zero(), F2::one(), F2::zero(), F2::one()
        ];

        let result_res = matrix_vector_multiply_f2(&lpn_mat, &vector);
        assert!(result_res.is_ok());
        let result = result_res.unwrap();
        assert_eq!(result.len(), params.k);
        assert!(result.iter().all(|&x| x == F2::zero() || x == F2::one()));
        println!("Mat(F2)-Vec(F2) Mult Result (y=Hx): {:?}", result);

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
         assert_eq!(sparse_result.len(), sparse_params.k);
         assert!(sparse_result.iter().all(|&x| x == F2::zero() || x == F2::one()));
         println!("Sparse Mat(F2)-Vec(F2) Mult Result (y=Hx): {:?}", sparse_result);
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
        let vector_fq: Vec<Field128> = (0..params.k).map(|_| Field128::rand(&mut rng)).collect(); // Use Field::rand()

        let result_res = matrix_transpose_vector_multiply_fq(&lpn_mat, &vector_fq);
        assert!(result_res.is_ok());
        let result = result_res.unwrap();

        assert_eq!(result.len(), params.n);

        println!("Mat(H^T)-Vec(Fq) Mult Result (y=H^T x): {} rows", result.len());

        let sparse_params = LpnParameters {
            n: 8,
            k: 4,
            t: 1,
            code_type: CodeType::Ldpc { sparsity: 2 },
        };
         let sparse_mat = sparse_params.generate_matrix().unwrap(); // H (k x n)
         let sparse_vector_fq: Vec<Field128> = (0..sparse_params.k).map(|_| Field128::rand(&mut rng)).collect(); // Use Field::rand()
         let sparse_res = matrix_transpose_vector_multiply_fq(&sparse_mat, &sparse_vector_fq);
         assert!(sparse_res.is_ok());
         let sparse_result = sparse_res.unwrap();
         assert_eq!(sparse_result.len(), sparse_params.n);
         println!("Sparse Mat(H^T)-Vec(Fq) Mult Result (y=H^T x): {} rows", sparse_result.len());
    }

    #[test]
    fn test_matrix_vector_mult_fq() {
        let params = LpnParameters {
            n: 8,
            k: 4,
            t: 1,
            code_type: CodeType::RandomLinear,
        };
        let lpn_mat = params.generate_matrix().unwrap();
        let vector: Vec<Field128> = vec![
            Field128::one(), Field128::zero(), Field128::one(), Field128::one(), // Use Field::one/zero
            Field128::zero(), Field128::one(), Field128::zero(), Field128::one()
        ];

        let result_res = matrix_vector_multiply_fq(&lpn_mat, &vector);
        assert!(result_res.is_ok());
        let result = result_res.unwrap();
        assert_eq!(result.len(), params.n);
        assert!(result.iter().all(|&x| x == Field128::zero() || x == Field128::one())); // Use Field::zero/one
        println!("Mat(Fq H)-Vec(Fq) Mult Result (y=Hx): {:?}", result);

        let sparse_params = LpnParameters {
            n: 8,
            k: 4,
            t: 1,
            code_type: CodeType::Ldpc { sparsity: 2 },
        };
         let sparse_mat = sparse_params.generate_matrix().unwrap();
         let sparse_res = matrix_vector_multiply_fq(&sparse_mat, &vector);
         assert!(sparse_res.is_ok());
         let sparse_result = sparse_res.unwrap();
         assert_eq!(sparse_result.len(), sparse_params.n);
         println!("Sparse Mat(Fq H)-Vec(Fq) Mult Result (y=Hx): {:?}", sparse_result);
    }
}
