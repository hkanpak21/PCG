use aes::Aes128;
use aes::cipher::{BlockEncrypt, KeyInit, generic_array::GenericArray};
use ark_std::vec::Vec;
use crate::pcg_core::PcgError;
use std::marker::PhantomData;

/// Trait for Correlation-Robust Hash Functions (CRHF).\n/// H(i, x) -> y
pub trait Crhf<Input, Output> {
    /// Hashes the input `input` associated with index `i`.\n    /// Should be correlation-robust.
    fn hash(&self, index: usize, input: &Input) -> Result<Output, PcgError>;
}

/// Placeholder AES-based CRHF.\n/// H_K(x) = AES_K(x) XOR x (Davies-Meyer)
/// For H(i, x), we might use K = PRF(master_key, i) or AES_K(i || x)
/// Currently VERY simplified: Uses a fixed zero key and ignores index.
/// Input type `In` determines how input bytes are provided.
#[derive(Clone)] // Add Clone
pub struct AesCrhf<In: ?Sized> { // Keep In generic, maybe Vec<u8> or &[u8]
    cipher: Aes128,
    _marker: PhantomData<In>,
}

// Implement Default manually if needed, or use new_with_key.
impl Default for AesCrhf<Vec<u8>> { // Default for Vec<u8> input type
    fn default() -> Self {
        println!("WARNING: AesCrhf using insecure default ZERO key!");
        let zero_key = GenericArray::from([0u8; 16]);
        Self { 
            cipher: Aes128::new(&zero_key),
             _marker: PhantomData,
        }
    }
}


// Provide a constructor that takes a key
impl<In> AesCrhf<In> { // Make generic over In
    /// Creates a new AesCrhf instance with the given AES key.
    pub fn new_with_key(key: &[u8]) -> Result<Self, PcgError> {
        if key.len() != 16 { // Basic key length check
            return Err(PcgError::CrhfError("Invalid AES key length".into()));
        }
        let aes_key = GenericArray::from_slice(key);
        Ok(Self {
            cipher: Aes128::new(aes_key),
            _marker: PhantomData,
        })
    }
}


impl<In: AsRef<[u8]>> Crhf<In, Vec<u8>> for AesCrhf<In> {
    /// Simplified AES hash: Ignores index, assumes input fits one block.
    /// H(x) = AES_K(x) (truncated/padded)
    fn hash(&self, index: usize, input: &In) -> Result<Vec<u8>, PcgError> {
        let input_bytes = input.as_ref();
        if input_bytes.is_empty() {
            return Err(PcgError::CrhfError("Input cannot be empty".into()));
        }

        // Incorporate index: Simple approach - prefix input with index bytes
        // Requires careful domain separation analysis in practice.
        let mut block_input_vec = Vec::new();
        block_input_vec.extend_from_slice(&index.to_le_bytes()); // Add index
        // Pad/truncate input_bytes to fit remaining block space? Or hash index separately?
        // Let's hash (index || input) padded to block size.
        block_input_vec.extend_from_slice(input_bytes);

        // Pad to AES block size (16 bytes)
        let block_len = 16;
        let padded_len = ((block_input_vec.len() + block_len - 1) / block_len) * block_len;
        block_input_vec.resize(padded_len, 0u8); // Pad with zeros

        // Process blocks (e.g., using CBC mode or just encrypting first block)
        // Simplest: Encrypt the first padded block containing index and input.
        if block_input_vec.len() < block_len {
            return Err(PcgError::CrhfError("Internal padding error".into()));
        }
        let mut block = GenericArray::clone_from_slice(&block_input_vec[0..block_len]);

        self.cipher.encrypt_block(&mut block);

        // Output the encrypted block
        Ok(block.to_vec())
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use ark_std::vec;

    #[test]
    fn test_aes_crhf_instantiation() {
        let key = [0u8; 16];
        let crhf_res = AesCrhf::<Vec<u8>>::new_with_key(&key);
        assert!(crhf_res.is_ok());
        let crhf = crhf_res.unwrap();

        let index = 42;
        let input = vec![1, 2, 3, 4, 5];
        let hash_res = crhf.hash(index, &input);
        assert!(hash_res.is_ok());
        let hash_val = hash_res.unwrap();
        assert_eq!(hash_val.len(), 16);
        println!("AES CRHF Hash: {:?}", hash_val);

        let input2 = vec![6, 7, 8, 9, 10];
        let hash_res2 = crhf.hash(index, &input2);
        assert!(hash_res2.is_ok());
        let hash_val2 = hash_res2.unwrap();
        assert_eq!(hash_val2.len(), 16);
        assert_ne!(hash_val, hash_val2, "Hashes should differ for different inputs");

        let hash_res3 = crhf.hash(index + 1, &input);
        assert!(hash_res3.is_ok());
        let hash_val3 = hash_res3.unwrap();
        assert_eq!(hash_val3.len(), 16);
        assert_ne!(hash_val, hash_val3, "Hashes should differ for different indices");
    }

     #[test]
    fn test_aes_crhf_default() {
        let crhf = AesCrhf::<Vec<u8>>::default();
        let index = 1;
        let input = vec![10; 8];
        let hash_res = crhf.hash(index, &input);
        assert!(hash_res.is_ok());
        assert_eq!(hash_res.unwrap().len(), 16);
    }
}
