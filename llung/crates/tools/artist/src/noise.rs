//! Deterministic value noise and fBm — the raw material for the
//! procedural styles. Everything derives from a u64 seed by integer
//! avalanche hashing: no RNG state, same seed ⇒ same art on every
//! machine (useful for tests and for "regenerate this one").

/// Lattice hash: coordinates + seed → [0, 1).
fn hash2(ix: i64, iy: i64, seed: u64) -> f64 {
    let mut h = (ix as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (iy as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
        ^ seed.wrapping_mul(0x1656_67B1_9E37_79F9);
    // splitmix64-style finalizer: thorough avalanche so neighbouring
    // lattice points look uncorrelated.
    h ^= h >> 30;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^= h >> 31;
    // 53-bit mantissa fit.
    (h >> 11) as f64 / (1u64 << 53) as f64
}

fn smoothstep(t: f64) -> f64 {
    t * t * (3.0 - 2.0 * t)
}

/// Smooth value noise in [0, 1] at frequency 1 (lattice spacing 1).
pub fn value_noise(x: f64, y: f64, seed: u64) -> f64 {
    let ix = x.floor() as i64;
    let iy = y.floor() as i64;
    let fx = smoothstep(x - ix as f64);
    let fy = smoothstep(y - iy as f64);

    let a = hash2(ix, iy, seed);
    let b = hash2(ix + 1, iy, seed);
    let c = hash2(ix, iy + 1, seed);
    let d = hash2(ix + 1, iy + 1, seed);

    let ab = a + (b - a) * fx;
    let cd = c + (d - c) * fx;
    ab + (cd - ab) * fy
}

/// Fractal Brownian motion: `octaves` doublings of value noise, amplitude
/// halving each octave, each with its own seed branch. Output roughly
/// centered on 0.5, clamped to [0, 1].
pub fn fbm(x: f64, y: f64, seed: u64, octaves: u32) -> f64 {
    let mut sum = 0.0;
    let mut amp = 0.5;
    let mut freq = 1.0;
    let mut norm = 0.0;
    for o in 0..octaves.max(1) {
        let branch = seed ^ (o as u64 + 1).wrapping_mul(0x517C_C1B7_2722_0A95);
        sum += amp * value_noise(x * freq, y * freq, branch);
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    (sum / norm).clamp(0.0, 1.0)
}

/// One avalanche round on the seed itself — picking per-seed constants
/// (e.g. which style "mix" resolves to) without an RNG.
pub fn derive(seed: u64, salt: u64) -> u64 {
    let mut h = seed ^ salt.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h ^= h >> 30;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^= h >> 31;
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_is_deterministic_and_seeded() {
        assert_eq!(fbm(1.25, 3.5, 42, 4), fbm(1.25, 3.5, 42, 4));
        assert_ne!(fbm(1.25, 3.5, 42, 4), fbm(1.25, 3.5, 43, 4));
        // In range.
        for i in 0..100 {
            let v = fbm(i as f64 * 0.37, i as f64 * 0.11, 7, 5);
            assert!((0.0..=1.0).contains(&v), "{v}");
        }
    }

    #[test]
    fn derive_is_stable() {
        assert_eq!(derive(42, 1), derive(42, 1));
        assert_ne!(derive(42, 1), derive(43, 1));
    }
}
