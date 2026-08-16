use image::{GenericImageView, ImageFormat, Rgba};
use std::io::Cursor;
use std::path::Path;

pub fn process_avatar_thumbnail<P: AsRef<Path>>(
    input_path: P,
    size: u32,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let img = image::open(input_path)?;

    // center crop to a 1:1 square
    let (width, height) = img.dimensions();
    let min_dim = width.min(height);
    let x = (width - min_dim) / 2;
    let y = (height - min_dim) / 2;
    let cropped = img.crop_imm(x, y, min_dim, min_dim);

    // down to target dimensions
    let resized = cropped.resize_exact(size, size, image::imageops::FilterType::Triangle);
    let mut rgba = resized.to_rgba8();

    // mask and anti-alias
    let center = size as f32 / 2.0;
    let radius = center;
    let feather = 1.2f32; // Transition band width in pixels (1.0 to 1.5 gives crisp, smooth edges)

    for (x, y, pixel) in rgba.enumerate_pixels_mut() {
        // Distance from pixel center to image center
        let dx = x as f32 + 0.5 - center;
        let dy = y as f32 + 0.5 - center;
        let dist = (dx * dx + dy * dy).sqrt();

        if dist >= radius {
            // Entirely outside the circle -> fully transparent
            *pixel = Rgba([0, 0, 0, 0]);
        } else if dist > radius - feather {
            // In the transition zone -> compute smooth anti-aliased alpha
            let t = (radius - dist) / feather;

            // Smoothstep curve (3t² - 2t³) for a natural visual falloff
            let alpha_factor = t * t * (3.0 - 2.0 * t);

            // Multiply existing pixel alpha by the factor
            let current_alpha = pixel[3] as f32;
            pixel[3] = (current_alpha * alpha_factor).round() as u8;
        }
        // inside (dist <= radius - feather) retains its original alpha
    }

    // Encode to PNG buffer
    let mut png_bytes = Vec::new();
    rgba.write_to(&mut Cursor::new(&mut png_bytes), ImageFormat::Png)?;

    Ok(png_bytes)
}

use cid::Cid;
use multihash::Multihash;
use sha2::{Digest, Sha256};

pub fn generate_cid_from_bytes(bytes: &[u8]) -> Result<String, Box<dyn std::error::Error>> {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let hash_result = hasher.finalize();

    const SHA256_MULTIHASH_CODE: u64 = 0x12;
    let multihash = Multihash::wrap(SHA256_MULTIHASH_CODE, &hash_result)?;

    const RAW_CODEC: u64 = 0x55;
    let cid = Cid::new_v1(RAW_CODEC, multihash);

    Ok(cid.to_string())
}
