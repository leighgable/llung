//! Print a chart to the terminal: the braille rows of the render document.
fn main() {
    let out = landscape_tool::run("{\"width\": 76, \"height\": 26, \"grid\": 20}").unwrap();
    let doc: serde_json::Value = serde_json::from_str(&out).unwrap();
    for row in doc["rows"].as_array().unwrap() {
        println!("{}", row.as_str().unwrap());
    }
}
