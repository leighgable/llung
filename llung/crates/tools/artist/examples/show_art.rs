//! Print a piece of art to the terminal: the braille rows of the render
//! document, exactly as the chat panel would lay them out.
fn main() {
    let args: String = std::env::args()
        .skip(1)
        .next()
        .unwrap_or_else(|| r#"{"width": 76, "height": 26}"#.to_string());
    let out = artist_tool::run(&args).unwrap();
    let doc: serde_json::Value = serde_json::from_str(&out).unwrap();
    println!("{}", doc["title"].as_str().unwrap());
    for row in doc["rows"].as_array().unwrap() {
        println!("{}", row.as_str().unwrap());
    }
}
