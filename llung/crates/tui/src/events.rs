use crossterm::event::{Event, KeyEvent};
use tokio::sync::mpsc;

pub fn spawn_terminal_reader(tx: mpsc::Sender<Event>) {
    tokio::spawn(async move {
        loop {
            if let Ok(ev) = crossterm::event::read() {
                if tx.send(ev).await.is_err() {
                    break;
                }
            }
        }
    });
}
