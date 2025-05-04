use aes::Aes128;
use ark_ff::Field;
use ark_std::marker::PhantomData;
use ark_std::vec::Vec;
use block_cipher::{
    BlockCipher, NewBlockCipher,
    generic_array::{GenericArray, typenum::U16},
};
use rand::{RngCore, thread_rng};

// Type alias for AES block
type Block = GenericArray<u8, U16>;

/// Structure for a DPF key ([GI14] construction).
/// Contains the initial seed and correction words for each level.
#[derive(Clone)] // Clone might be needed for seeds/outputs
pub struct DpfKey {
    key_id: u8, // 0 or 1 (sigma)
    domain_bits: usize,
    s0: Block, // Initial seed
    // Correction words: one block per level (except last)
    // cw[i] corresponds to level i+1
    cw: Vec<Block>,
    // Final correction value (single bit for F2 DPF)
    cw_final: u8,
}

// Added basic new method for placeholder usage if needed elsewhere
// Should ideally be generated only by `gen`
impl DpfKey {
     pub fn new() -> Self {
         // Only suitable for placeholders, not a valid key
          Self {
              key_id: 0,
              domain_bits: 0,
              s0: Default::default(),
              cw: vec![],
              cw_final: 0,
          }
     }
}

/// Structure for the DPF functionality.
/// F: The *output type* of the DPF (e.g., F2, or a field element).
/// We implement the core logic for F2 (0 or 1) output first.
pub struct Dpf<F> {
    domain_bits: usize,
    aes_cipher: Aes128, // Use a fixed key for AES for simplicity now
    _marker: PhantomData<F>,
}

// Helper: XOR two blocks
#[inline(always)]
fn xor_block(a: &Block, b: &Block) -> Block {
    let mut res = Block::default();
    for i in 0..16 {
        res[i] = a[i] ^ b[i];
    }
    res
}

// Helper: Evaluate PRG (Fixed-key AES) on a block
// Expands one block seed into two block seeds for the next level.
#[inline(always)]
fn prg(cipher: &Aes128, input: &Block) -> (Block, Block) {
    // Input block contains seed `s` and control bit `t` (LSB).
    // Left seed: PRG(s)
    let mut s_l = input.clone();
    s_l[15] &= 0xfe; // Ensure LSB is 0 before encryption for s_L
    cipher.encrypt_block(&mut s_l);

    // Right seed: PRG(s XOR 1...1) - approx, use fixed tweak for simplicity
    // Let's use PRG(s XOR FixedTweak) instead of relying on input LSB state
    let tweak = GenericArray::from([0xff; 16]); // Example fixed tweak
    let mut s_r = xor_block(input, &tweak);
    s_r[15] &= 0xfe; // Ensure LSB is 0 before encryption for s_R
    cipher.encrypt_block(&mut s_r);

    (s_l, s_r)
}

// Helper: Get the i-th bit of alpha (0-indexed from right)
#[inline(always)]
fn bit(alpha: usize, i: usize) -> u8 {
    ((alpha >> i) & 1) as u8
}

// Helper: Get the last bit of a block (LSB)
#[inline(always)]
fn lsb(block: &Block) -> u8 {
     block[15] & 1
}

impl<F> Dpf<F>
where F: From<u8> + std::ops::Add<Output = F> + Clone + Default + PartialEq + std::fmt::Debug // Traits for F2 testing
{
    /// Creates a new DPF instance.
    pub fn new(domain_bits: usize) -> Self {
        // Placeholder: Use a fixed zero key for AES. DO NOT USE IN PRODUCTION.
        let key = GenericArray::from([0u8; 16]);
        let aes_cipher = Aes128::new(&key);
        Self { domain_bits, aes_cipher, _marker: PhantomData }
    }

    /// DPF Gen algorithm ([GI14], Appendix D). Corrected version.
    /// Generates keys (k0, k1) for point `alpha` and value `beta` (must be 0 or 1).
    /// Output F is restricted to representing F2 values (0 or 1).
    pub fn gen(
        &self,
        alpha: usize, // The point (index)
        beta: u8,     // The value (0 or 1) at the point
    ) -> Result<(DpfKey, DpfKey), &'static str> {
        if beta > 1 {
            return Err("DPF Gen currently only supports beta = 0 or 1");
        }
        if alpha >= (1 << self.domain_bits) {
             return Err("Alpha out of domain bounds");
        }

        let n = self.domain_bits;
        let mut rng = thread_rng();

        // 1. Sample initial seeds s_0, s_1 and keep track of control bits t_0, t_1
        let mut s = [Block::default(), Block::default()];
        rng.fill_bytes(s[0].as_mut_slice());
        rng.fill_bytes(s[1].as_mut_slice());

        let mut t = [lsb(&s[0]), lsb(&s[1])];
        s[0][15] &= 0xfe; // Zero out LSB for the actual seed part
        s[1][15] &= 0xfe;

        let s_init = [s[0].clone(), s[1].clone()]; // Keep initial seeds for keys

        let mut cw = vec![Block::default(); n]; // Correction words for levels 1..n

        // 2. Iterate through levels i = 0 to n-1
        for i in 0..n {
            // PRG expansion for both seeds
            let (s_l_0, s_r_0) = prg(&self.aes_cipher, &s[0]);
            let (s_l_1, s_r_1) = prg(&self.aes_cipher, &s[1]);

            // Get control bits from expansions
            let t_l = [lsb(&s_l_0), lsb(&s_l_1)];
            let t_r = [lsb(&s_r_0), lsb(&s_r_1)];

            // Determine which side (left/right) corresponds to alpha at this level
            let alpha_i = bit(alpha, n - 1 - i); // Bit from left

            // Correction word cw_i is the XOR sum along the non-alpha path
            if alpha_i == 0 { // Non-alpha path is Right
                 cw[i] = xor_block(&s_r_0, &s_r_1);
            } else { // Non-alpha path is Left
                 cw[i] = xor_block(&s_l_0, &s_l_1);
            }

            // Apply correction word based on control bits t[0] and t[1]
            // If t[0] is 1, correct s_L/R[1] based on alpha_i
            // If t[1] is 1, correct s_L/R[0] based on alpha_i
            let mut s_l_corrected = [s_l_0, s_l_1];
            let mut s_r_corrected = [s_r_0, s_r_1];

            if alpha_i == 0 { // Non-alpha is Right
                if t[0] == 1 { s_r_corrected[1] = xor_block(&s_r_corrected[1], &cw[i]); }
                if t[1] == 1 { s_r_corrected[0] = xor_block(&s_r_corrected[0], &cw[i]); }
            } else { // Non-alpha is Left
                if t[0] == 1 { s_l_corrected[1] = xor_block(&s_l_corrected[1], &cw[i]); }
                if t[1] == 1 { s_l_corrected[0] = xor_block(&s_l_corrected[0], &cw[i]); }
            }

            // Update control bits t[0], t[1] for the next level
            for sigma in 0..=1 {
                 if alpha_i == 0 { // Keep Left seeds
                     s[sigma] = s_l_corrected[sigma];
                     // t_next = t_L[sigma] XOR (t_current[sigma] * LSB(cw_i))
                     // Here LSB(cw_i) = t_R[0] XOR t_R[1]
                     t[sigma] = t_l[sigma] ^ (t[sigma] * (t_r[0] ^ t_r[1]));
                 } else { // Keep Right seeds
                     s[sigma] = s_r_corrected[sigma];
                     // t_next = t_R[sigma] XOR (t_current[sigma] * LSB(cw_i))
                     // Here LSB(cw_i) = t_L[0] XOR t_L[1]
                     t[sigma] = t_r[sigma] ^ (t[sigma] * (t_l[0] ^ t_l[1]));
                 }
                 s[sigma][15] &= 0xfe; // Zero out LSB for next iteration seed
            }
        }

        // 3. Compute final correction word cw_final (single bit)
        let final_convert0 = lsb(&s[0]); // Convert(s_n^0)
        let final_convert1 = lsb(&s[1]); // Convert(s_n^1)

        // Last correction bit depends on beta and final control bits t[0], t[1]
        let cw_final_0 = final_convert1 ^ (if t[0] == 1 { beta } else { 0 });
        let cw_final_1 = final_convert0 ^ (if t[1] == 1 { beta } else { 0 });

        let k0 = DpfKey { key_id: 0, domain_bits: n, s0: s_init[0], cw: cw.clone(), cw_final: cw_final_0 };
        let k1 = DpfKey { key_id: 1, domain_bits: n, s0: s_init[1], cw: cw, cw_final: cw_final_1 };

        Ok((k0, k1))
    }

    /// DPF FullEval algorithm ([GI14]). Evaluates key on all 2^n domain points. Corrected version.
    /// Returns a vector representing the DPF evaluation over F2 (0s or 1s).
    pub fn full_eval(&self, key: &DpfKey) -> Result<Vec<F>, &'static str> {
        if key.domain_bits != self.domain_bits {
             return Err("Key domain size mismatch");
        }
        let n = self.domain_bits;
        let num_leaves = 1 << n;

        // Stores seeds for the current level being processed
        let mut current_s = Vec::with_capacity(1 << n);
        current_s.push(key.s0.clone());
        // Stores control bits for the current level
        let mut current_t = Vec::with_capacity(1 << n);
        current_t.push(lsb(&key.s0)); // Initial control bit

        // Traverse levels 0 to n-1
        for i in 0..n {
            let level_size = 1 << i;
            let mut next_s = Vec::with_capacity(level_size * 2);
            let mut next_t = Vec::with_capacity(level_size * 2);
            let cw_i = &key.cw[i];

            for j in 0..level_size {
                let s_parent = &current_s[j];
                let t_parent = current_t[j];

                // Expand seed using PRG
                let (mut s_l, mut s_r) = prg(&self.aes_cipher, s_parent);

                // Apply correction word cw[i] if t_parent is 1
                 if t_parent == 1 {
                     s_l = xor_block(&s_l, cw_i);
                     s_r = xor_block(&s_r, cw_i);
                 }

                 // Get control bits for children
                 let t_l = lsb(&s_l);
                 let t_r = lsb(&s_r);
                 s_l[15] &= 0xfe; // Zero out LSB for next seed
                 s_r[15] &= 0xfe;

                 // Store seeds and updated control bits for next level
                 next_s.push(s_l);
                 next_t.push(t_l ^ t_parent); // T_next = T_child XOR T_parent
                 next_s.push(s_r);
                 next_t.push(t_r ^ t_parent);
            }
            current_s = next_s;
            current_t = next_t;
        }

        // Final level n: apply final correction word
        let mut eval_vec: Vec<F> = Vec::with_capacity(num_leaves);
        for j in 0..num_leaves {
            // Convert uses LSB for F2
            let mut val = lsb(&current_s[j]);
             if current_t[j] == 1 { // If final control bit is 1
                 val ^= key.cw_final;
             }
            eval_vec.push(F::from(val));
        }

        Ok(eval_vec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    // Simple u8 type for F2 testing
    type TestFieldF2 = u8;

    #[test]
    fn test_dpf_gen_eval_f2() {
        let domain_bits = 10; // Example: domain size 2^10 = 1024
        let dpf = Dpf::<TestFieldF2>::new(domain_bits);
        let mut rng = thread_rng();

        let alpha = rng.gen_range(0..1 << domain_bits); // Random point
        let beta = 1u8; // Value is 1 at alpha

        // Test Gen
        let gen_result = dpf.gen(alpha, beta);
        assert!(gen_result.is_ok());
        let (k0, k1) = gen_result.unwrap();

        assert_eq!(k0.domain_bits, domain_bits);
        assert_eq!(k1.domain_bits, domain_bits);
        assert_eq!(k0.cw.len(), domain_bits);
        assert_eq!(k1.cw.len(), domain_bits);

        // Test FullEval
        let eval0_res = dpf.full_eval(&k0);
        let eval1_res = dpf.full_eval(&k1);
        assert!(eval0_res.is_ok(), "Eval0 failed: {:?}", eval0_res.err());
        assert!(eval1_res.is_ok(), "Eval1 failed: {:?}", eval1_res.err());

        let eval0 = eval0_res.unwrap();
        let eval1 = eval1_res.unwrap();

        let domain_size = 1 << domain_bits;
        assert_eq!(eval0.len(), domain_size);
        assert_eq!(eval1.len(), domain_size);

        // Check correctness: eval0[x] ^ eval1[x] should be beta if x=alpha, else 0
        let mut mismatch_found = false;
        for i in 0..domain_size {
            let sum = eval0[i] ^ eval1[i]; // XOR for F2 addition
            if i == alpha {
                if sum != beta {
                    eprintln!("Mismatch at alpha={}: eval0={}, eval1={}, sum={}, expected_beta={}", i, eval0[i], eval1[i], sum, beta);
                    mismatch_found = true;
                }
            } else {
                if sum != 0 {
                     eprintln!("Mismatch at non-alpha={}: eval0={}, eval1={}, sum={}", i, eval0[i], eval1[i], sum);
                     mismatch_found = true;
                }
            }
        }
        assert!(!mismatch_found, "DPF evaluation correctness check failed");
        println!("DPF F2 test passed for alpha={}, beta={}", alpha, beta);
    }

     #[test]
     fn test_dpf_gen_beta_zero() {
         let domain_bits = 8;
         let dpf = Dpf::<TestFieldF2>::new(domain_bits);
         let mut rng = thread_rng();
         let alpha = rng.gen_range(0..1 << domain_bits);
         let beta = 0u8; // Value is 0 at alpha

         let (k0, k1) = dpf.gen(alpha, beta).expect("Gen failed");
         let eval0 = dpf.full_eval(&k0).expect("Eval0 failed");
         let eval1 = dpf.full_eval(&k1).expect("Eval1 failed");

         let domain_size = 1 << domain_bits;
         for i in 0..domain_size {
             assert_eq!(eval0[i] ^ eval1[i], 0, "Sum should be 0 for all x when beta=0");
         }
         println!("DPF F2 test passed for beta=0");
     }

      #[test]
      fn test_dpf_alpha_out_of_bounds() {
          let domain_bits = 4;
          let dpf = Dpf::<TestFieldF2>::new(domain_bits);
          let alpha = 16; // 1 << domain_bits
          let beta = 1u8;
          let result = dpf.gen(alpha, beta);
          assert!(result.is_err());
          assert_eq!(result.err(), Some("Alpha out of domain bounds"));
      }
}
