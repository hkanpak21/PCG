use aes::{Aes128, NewBlockCipher, cipher::{BlockEncrypt, KeyInit}};
use ark_std::vec::Vec;
use ark_std::marker::PhantomData;
use block_cipher::generic_array::GenericArray;
use core::convert::TryInto;

/// Trait for a Correlation-Robust Hash Function (CRHF).
///
/// Maps an index `i` and an `input` value to an `output_string`.
pub trait Crhf {
    type Input;
    type Output;

    /// Hashes the input along with an index.
    /// `index`: An index `i` used to potentially tweak the hash.
    /// `input`: The primary input to the hash function.
    fn hash(&self, index: usize, input: &Self::Input) -> Self::Output;
}

/// A CRHF implementation based on fixed-key AES.
/// Models AES as a random permutation.
/// H(i, x) = AES_K(i || x) ^ (i || x)  (Davies-Meyer like construction)
/// The key K is fixed globally (e.g., derived from a master seed or hardcoded for simplicity for now).
/// Output size is 128 bits (AES block size).
pub struct AesCrhf<In: ?Sized> {
    aes_cipher: Aes128,
    _input_type: PhantomData<In>,
    // In a real implementation, the key `K` might be configured or derived.
    // For now, we'll use a dummy fixed key.
}

impl<In: AsRef<[u8]>> AesCrhf<In> {
    /// Creates a new AES-based CRHF instance.
    /// Note: Uses a fixed, public key for demonstration.
    /// A real implementation should use a properly generated, secret key.
    pub fn new() -> Self {
        // Dummy key for demonstration purposes.
        // DO NOT USE IN PRODUCTION.
        let key = GenericArray::from([0u8; 16]);
        let aes_cipher = Aes128::new(&key);
        Self { aes_cipher, _input_type: PhantomData }
    }

    /// Prepares the AES block by combining index and input.
    /// Pads with zeros if necessary to fill a 128-bit block.
    /// Currently assumes index fits in u64 and input fits in remaining space.
    /// A more robust implementation would handle larger inputs (e.g., using a mode like CBC-MAC or hash).
    fn prepare_block(&self, index: usize, input: &In) -> GenericArray<u8, typenum::U16> {
        let mut block_bytes = [0u8; 16];
        let index_bytes = (index as u64).to_le_bytes();
        block_bytes[0..8].copy_from_slice(&index_bytes);

        let input_bytes = input.as_ref();
        let input_len = core::cmp::min(input_bytes.len(), 8);
        block_bytes[8..8 + input_len].copy_from_slice(&input_bytes[..input_len]);

        // TODO: Handle inputs larger than 8 bytes. Hashing or a proper MAC mode needed.
        if input_bytes.len() > 8 {
            println!("Warning: AES CRHF input truncated to 8 bytes!");
        }

        GenericArray::from(block_bytes)
    }
}

impl<In: AsRef<[u8]>> Crhf for AesCrhf<In> {
    // Assuming Input is something that can be represented as bytes.
    // Output is fixed to 128 bits (Vec<u8> of length 16).
    type Input = In;
    type Output = Vec<u8>;

    /// Implements H(i, x) = AES_K(i || x) ^ (i || x)
    fn hash(&self, index: usize, input: &Self::Input) -> Self::Output {
        let block = self.prepare_block(index, input);
        let mut encrypted_block = block.clone();

        self.aes_cipher.encrypt_block(&mut encrypted_block);

        // XOR output with input (Davies-Meyer)
        let result_bytes: Vec<u8> = encrypted_block
            .iter()
            .zip(block.iter())
            .map(|(enc_byte, in_byte)| enc_byte ^ in_byte)
            .collect();

        result_bytes
    }
}

impl<In: AsRef<[u8]>> Default for AesCrhf<In> {
    fn default() -> Self {
        Self::new()
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, thread_rng};

    #[test]
    fn test_aes_crhf_basic() {
        let crhf = AesCrhf::<[u8]>::new();
        let index = 12345;
        let input = b"test_input"; // 10 bytes, will be truncated

        let output = crhf.hash(index, input);

        assert_eq!(output.len(), 16); // AES block size

        // Hash of same input/index should be identical
        let output2 = crhf.hash(index, input);
        assert_eq!(output, output2);

        // Hash of different index should be different (likely)
        let output_diff_index = crhf.hash(index + 1, input);
        assert_ne!(output, output_diff_index);

        // Hash of different input should be different (likely)
        let output_diff_input = crhf.hash(index, b"other_inp");
        assert_ne!(output, output_diff_input);

        println!("CRHF Output ({} bytes): {:?}", output.len(), output);
    }

    #[test]
    fn test_aes_crhf_vectors() {
        let crhf = AesCrhf::<Vec<u8>>::new();
        let index = 987;
        let input_vec = vec![0, 1, 2, 3, 4, 5]; // 6 bytes

        let output = crhf.hash(index, &input_vec);
        assert_eq!(output.len(), 16);

        // Check determinism
        let output2 = crhf.hash(index, &input_vec);
        assert_eq!(output, output2);

        println!("CRHF Output Vec ({} bytes): {:?}", output.len(), output);
    }
}
