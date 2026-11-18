//! Print a diagram to the terminal: `cargo run -p diagram-tool --example show_diagram -- '<json>'`
fn main() {
    let args: String = std::env::args()
        .skip(1)
        .next()
        .unwrap_or_else(|| {
            r#"{"graph": "INTAKE: Intake\nbox INVESTIGATE: Investigate {\n  SUBA: Sub Agent A\n  SUBB: Sub Agent B\n}\nINTAKE -> INVESTIGATE\nSUBA -> PLAN: findings\nSUBB -> PLAN\nPLAN: Draft Plan\nPLAN -> APPROVAL: proposed plan\nAPPROVAL -> EXECUTE: approved\nEXECUTE -> REPORT\nREPORT: Report\n"}"#
                .to_string()
        });
    println!("{}", diagram_tool::run(&args).unwrap());
}
