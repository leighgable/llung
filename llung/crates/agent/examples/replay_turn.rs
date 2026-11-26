// Rebuild the exact first-ask prompt an agent sent for a message, and
// optionally replay it against the live model to see what it answers.
//
// Usage:
//   cargo run -p agent --example replay_turn -- <topic> <substring-of-trigger> \
//       [--agent sensei] [--db PATH] [--send [N]] [--port 9931] [--protocol line|xml|hermes]
//
// History compression uses the rule-based HeadlineSummarizer: identical
// to the live view for short chains (every message fits a verbatim
// node); long chains differ only in the merged summary lines. Places
// come from the usual resolution (LLUNG_WORKSPACE etc. / places.json).
// --send N asks the live llama-server N times (sampling varies) and
// reports which replies parse as tool calls.

use agent::history::{HeadlineSummarizer, HistoryCompressor, DEFAULT_CONTEXT_BUDGET};
use agent::interface::{Backend, OpenAiCompatible};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut take = |flag: &str| -> Option<String> {
        let i = args.iter().position(|a| a == flag)?;
        args.remove(i);
        if i < args.len() && !args[i].starts_with("--") {
            Some(args.remove(i))
        } else {
            Some(String::new())
        }
    };
    let agent_name = take("--agent").unwrap_or_else(|| "sensei".into());
    let db_path = take("--db").unwrap_or_else(|| {
        llung_core::identity::session::account_db_path().display().to_string()
    });
    let send = take("--send").map(|n| n.parse::<usize>().unwrap_or(1));
    let port: u16 = take("--port").and_then(|p| p.parse().ok()).unwrap_or(9931);
    let protocol_flag = take("--protocol").and_then(|p| agent::tools::ToolProtocol::parse(&p));
    let [topic, needle] = <[String; 2]>::try_from(args)
        .map_err(|_| "usage: replay_turn <topic> <trigger-substring> [flags]")?;

    // Work on a copy: never touch the live account DB.
    let tmp = std::env::temp_dir().join(format!("llung-replay-{}.db", std::process::id()));
    std::fs::copy(&db_path, &tmp)?;
    let db = llung_core::storage::db::Database::open(tmp.to_str().unwrap())?;
    let dag = db.load_dag()?;
    let trigger = dag
        .topic_linear(&topic)
        .into_iter()
        .rev()
        .find(|m| m.content.contains(&needle))
        .cloned()
        .ok_or("trigger message not found")?;
    let chain: Vec<_> = dag.context_chain(&trigger.id).into_iter().cloned().collect();
    let compressor = HistoryCompressor::new(Box::new(HeadlineSummarizer::new()), DEFAULT_CONTEXT_BUDGET);
    let tree = compressor.compress(&chain).await;
    let view = compressor.flatten_budgeted(&tree, compressor.context_budget);
    let trigger_text = format!("{}: {}", trigger.sender_name, trigger.content);
    let places = agent::places::current();
    // Same choice the agent loop makes: flag, else the live model's chat
    // template, else the TOOL line.
    let mut backend = OpenAiCompatible::llamacpp(port);
    backend.discover_model().await;
    let protocol = match protocol_flag {
        Some(p) => p,
        None => backend
            .chat_template()
            .await
            .map(|t| agent::tools::ToolProtocol::from_chat_template(&t))
            .unwrap_or_default(),
    };
    eprintln!("tool protocol: {protocol:?}");
    let prompt = agent::orchestrator::assemble_turn_prompt(
        &agent_name,
        &places,
        protocol,
        &[],
        &view,
        &trigger_text,
        &[],
        None,
        None,
    );
    for m in &prompt {
        println!("===== [{}] =====\n{}\n", m.agent_role(), m.content);
    }
    std::fs::remove_file(&tmp).ok();

    let Some(n) = send else { return Ok(()) };
    let mut parsed = 0;
    for i in 1..=n.max(1) {
        let reply = backend.chat(&prompt).await.map_err(|e| e.to_string())?;
        let call = agent::tools::parse_tool_call(&reply);
        if call.is_some() {
            parsed += 1;
        }
        println!("===== reply {i} — parsed: {:?} =====\n{reply}\n", call.map(|c| (c.name, c.arguments)));
    }
    println!("{parsed}/{} replies parsed as tool calls", n.max(1));
    Ok(())
}
