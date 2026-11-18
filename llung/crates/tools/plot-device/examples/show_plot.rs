//! Print a plot to the terminal: `cargo run -p plot-device-tool --example show_plot -- '<json>'`
fn main() {
    let args: String = std::env::args()
        .skip(1)
        .next()
        .unwrap_or_else(|| {
            r#"{"csv": "x,y\n1,2\n2,4\n3,9\n4,16\n5,25\n", "kind": "scatter", "width": 60, "height": 18}"#
                .to_string()
        });
    let out = plot_device_tool::run(&args).unwrap();
    let doc: serde_json::Value = serde_json::from_str(&out).unwrap();
    println!("{}", doc["title"].as_str().unwrap_or(""));
    for row in doc["rows"].as_array().unwrap() {
        println!("{}", row.as_str().unwrap());
    }
}
