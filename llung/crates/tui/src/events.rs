use crossterm::event::{Event, EventStream};
use futures::StreamExt;
use tokio::sync::mpsc;

pub fn spawn_terminal_reader(tx: mpsc::Sender<Event>) {
    tokio::spawn(async move {
        let mut reader = EventStream::new();
        loop {
            match reader.next().await {
                Some(Ok(ev)) => {
                    if tx.send(ev).await.is_err() {
                        break; // reciever dropped
                    }
                }
                Some(Err(e)) => {
                    eprintln!("Terminal read error: {e}");
                    break;
                }
                None => break,
            }
        }
    });
}
