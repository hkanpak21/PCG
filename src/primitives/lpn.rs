use crate::primitives::field::Field128;
use ark_ff::Field;
use ark_std::vec::Vec;
use nalgebra::{DMatrix, DVector};
use nalgebra_sparse::csr::CsrMatrix;
use rand::{Rng, thread_rng};
use std::ops::Add;

/// Matrix-vector multiplication: y = H * x
/// H: n x k (F_2)
/// x: k x 1 (F_q = Field128)
/// y: n x 1 (F_q = Field128)
/// Computes y_i = sum_{j where H_{i,j}=1} x_j
pub fn matrix_vector_multiply_fq(
    matrix: &LpnMatrix, // H (n x k)
    vector: &DVector<Field128>, // x (k x 1)
) -> Result<DVector<Field128>, &'static str> {
    let n = matrix.nrows();
    let k = matrix.ncols();

    if k != vector.nrows() {
        return Err("Matrix(Fq H) and vector(Fq) dimensions mismatch");
    }

    let mut result = DVector::<Field128>::from_element(n, Field128::ZERO);

    match matrix {
        LpnMatrix::Dense(h) => {
            for i in 0..n { // Iterate rows of H
                let mut sum_i = Field128::ZERO;
                for j in 0..k { // Iterate columns of H
                    if h[(i, j)] == 1 {
                        sum_i = sum_i.add(vector[j]); // Add x_j
                    }
                }
                result[i] = sum_i;
            }
        }
        LpnMatrix::Sparse(h) => {
            // Assumes CSR format stores H (n x k) correctly.
            // Iterate through rows of H
            for i in 0..n {
                 let row = h.row(i);
                 let mut sum_i = Field128::ZERO;
                 // Sum x_j for non-zero columns j in this row
                 for (j, val) in row.col_indices().iter().zip(row.values().iter()) {
                     if *val == 1 {
                         sum_i = sum_i.add(vector[*j]); // Add x_j
                     }
                 }
                 result[i] = sum_i;
            }
        }
    }
    Ok(result)
}


#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use crate::primitives::field::Field128; // Ensure Field128 is imported

    // ... (existing helper, test_random_linear_matrix_gen, test_ldpc_matrix_gen_placeholder, test_matrix_vector_mult_f2, test_matrix_transpose_vector_mult_fq) ...

    #[test]
    fn test_matrix_vector_mult_fq() {
        let mut rng = thread_rng();
        // H: n x k
        let n_rows = 8;
        let k_cols = 4;
        let params = LpnParameters {
            n: n_rows,
            k: k_cols,
            t: 1,
            code_type: CodeType::RandomLinear,
        };
        // Note: Our generate_matrix makes k x n matrix. We need n x k.
        // Let's generate H manually for this test or transpose.
        let h_dense = DMatrix::from_fn(n_rows, k_cols, |_, _| rng.gen_range(0..=1));
        let lpn_mat = LpnMatrix::Dense(h_dense);

        let vector_fq: DVector<Field128> = DVector::from_fn(k_cols, |_, _| Field128::from(rng.gen::<u128>())); // x (k x 1)

        // Check y = H * x (n x 1 result)
        let result_res = matrix_vector_multiply_fq(&lpn_mat, &vector_fq);
        assert!(result_res.is_ok());
        let result = result_res.unwrap();

        assert_eq!(result.nrows(), n_rows);
        println!("Mat(H)-Vec(Fq) Mult Result (y=Hx): {} rows", result.nrows());

        // Could add sparse test here too if needed.
    }

} 