use std::f32::consts::PI;
use textplots::{Chart, Plot, Shape}; //

struct Point3D {
    x: f32,
    y: f32,
    z: f32,
}

fn main() {
    // 1. Setup Camera Perspective Matrices (Parameters your LLM can control)
    let pitch = 40.0 * PI / 180.0; // Look down at a 40-degree angle
    let yaw = 35.0 * PI / 180.0; // Turn camera 35 degrees horizontally
    let camera_distance = 3.5;

    let (cos_p, sin_p) = (pitch.cos(), pitch.sin());
    let (cos_y, sin_y) = (yaw.cos(), yaw.sin());

    // Helper closure to project 3D coordinates into a 2D Viewport Space
    let project = |pt: &Point3D| -> (f32, f32) {
        // Rotate (Yaw & Pitch)
        let x_rot = pt.x * cos_y - pt.y * sin_y;
        let y_temp = pt.x * sin_y + pt.y * cos_y;
        let y_rot = y_temp * cos_p - pt.z * sin_p;
        let z_rot = y_temp * sin_p + pt.z * cos_p;

        // Apply perspective projection (Z-translation and perspective division)
        let z_final = z_rot + camera_distance;
        let proj_x = x_rot / z_final;
        let proj_y = y_rot / z_final;

        // Correct for terminal aspect ratio (terminal characters are roughly twice as tall as they are wide)
        (proj_x * 2.2, proj_y)
    };

    // 2. Generate the 3D Object Mesh Data (A Mathematical Ripple Topology)
    let grid_size = 15;
    let mut lines = Vec::new();

    // Pre-calculate 2D matrix of projected 3D points
    let mut mesh = vec![vec![(0.0f32, 0.0f32); grid_size]; grid_size];
    for i in 0..grid_size {
        for j in 0..grid_size {
            let x = (i as f32 / grid_size as f32) * 4.0 - 2.0;
            let y = (j as f32 / grid_size as f32) * 4.0 - 2.0;
            // Generate Z-height wave
            let z = ((x * x + y * y).sqrt() * 3.0).cos() * 0.4;

            mesh[i][j] = project(&Point3D { x, y, z });
        }
    }

    // 3. Construct Wireframe Edges (Connect points together into structural paths)
    for i in 0..grid_size {
        for j in 0..grid_size {
            // Draw horizontal rows
            if i < grid_size - 1 {
                lines.push(mesh[i][j]);
                lines.push(mesh[i + 1][j]);
            }
            // Draw vertical columns
            if j < grid_size - 1 {
                lines.push(mesh[i][j]);
                lines.push(mesh[i][j + 1]);
            }
        }
    }

    // 4. Stream to Textplots Braille Viewport Engine
    println!("\n--- 3D Wireframe Resolution via Textplots Braille Canvas ---");

    // Create an explicit bounded 120x60 subpixel bounding region
    Chart::new(120, 60, -1.8, 1.8)
        .lineplot(&Shape::Lines(&lines)) // Feed textplots our projected coordinates
        .display(); // Print out the beautiful, high-density Braille output string!
}
