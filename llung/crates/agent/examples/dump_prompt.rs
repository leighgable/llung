// Reconstruct the exact turn prompt the agent saw for a message in the
// shared account DB: DAG context chain -> HistoryCompressor (the real
// kimi-coding backend) -> flattened view, then the assembled prompt.
//
// Usage: cargo run -p agent --example dump_prompt -- <db-path> <substring-of-trigger-message>

use agent::history::HistoryCompressor;
use agent::interface::OpenAiCompatible;

fn kind_tag(k: &llung_core::network::message::MessageKind) -> String {
    let s = format!("{k:?}");
    s.split('{').next().unwrap_or(&s).to_string()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db_path = std::env::args().nth(1).unwrap_or_else(|| {
        "/home/leigh/.local/share/llung/account/node.db".to_string()
    });
    let needle = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "@kimi, I would like to test your glob".to_string());

    // Work on a copy: opening the real DB while the TUI runs is fine for
    // reads, but migrations/writes must not touch the live account.
    let tmp = std::env::temp_dir().join(format!("llung-dump-{}.db", std::process::id()));
    std::fs::copy(&db_path, &tmp)?;
    let db = llung_core::storage::db::Database::open(tmp.to_str().unwrap())?;

    let dag = db.load_dag()?;
    let trigger = dag
        .topic_linear("introductions")
        .into_iter()
        .find(|m| m.content.contains(&needle))
        .cloned()
        .ok_or("trigger message not found")?;
    println!("=== trigger [{}] from {} ===", kind_tag(&trigger.kind), trigger.sender_name);
    println!("{}\n", trigger.content);

    let chain: Vec<_> = dag.context_chain(&trigger.id).into_iter().cloned().collect();
    println!("=== context chain: {} messages ===", chain.len());
    for m in &chain {
        println!(
            "  {:<6} {:<10} {:>7} bytes  {}",
            kind_tag(&m.kind),
            m.sender_name,
            m.content.len(),
            m.content.lines().next().unwrap_or("").chars().take(70).collect::<String>(),
        );
    }

    // The exact compressor the agent loop uses: model-backed with the
    // agent's own backend (kimi-coding), same budget as AgentLoopContext.
    let key = llung_core::secrets::load_api_key("kimi").await?;
    let backend = OpenAiCompatible::kimi_coding("kimi-for-coding", None).with_api_key(key);
    let compressor = HistoryCompressor::with_model(Box::new(backend));

    println!("\n=== flattened view (what the model saw as history) ===");
    let tree = compressor.compress(&chain).await;
    let view = compressor.flatten_budgeted(&tree, compressor.context_budget);
    println!("{view}");
    println!("=== view: {} bytes ===", view.len());

    println!("\n=== assembled prompt ===");
    println!("[system] You are <name>, a helpful agent ... (fixed string)");
    println!("[user: context]\n{view}");
    let tool_block = agent::tools::tool_instructions();
    println!(
        "[user: new message]\nNew message: {}\n\n{}",
        trigger.content, tool_block
    );

    std::fs::remove_file(&tmp).ok();
    Ok(())
}
