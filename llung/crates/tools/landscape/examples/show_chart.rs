//! Print a chart to the terminal: the braille rows of the render
//! document. Optional JSON args as the first CLI argument, e.g.
//! `cargo run -p landscape-tool --example show_chart -- '{"expr":"(sqrt(x^2+y^2)-1.2)^2+z^2-0.25","grid":36}'`
fn main() {
    let args: String = std::env::args()
        .skip(1)
        .next()
        .unwrap_or_else(|| r#"{"width": 76, "height": 26, "grid": 20}"#.to_string());
    let out = landscape_tool::run(&args).unwrap();
    let doc: serde_json::Value = serde_json::from_str(&out).unwrap();
    println!("{}", doc["title"].as_str().unwrap_or(""));
    for row in doc["rows"].as_array().unwrap() {
        println!("{}", row.as_str().unwrap());
    }
}
