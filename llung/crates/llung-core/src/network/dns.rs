//! DNS resolver setup for the libp2p transport.
//!
//! The system resolver config (`/etc/resolv.conf`) can be empty while the
//! machine is offline or still joining a network. That must not stop the
//! node: mDNS and direct `/ip4`/`/ip6` addresses work without DNS. So we
//! wait briefly for the config to appear, and otherwise run with a
//! resolver that has no nameservers. `/dns*` addresses then fail to dial
//! individually instead of the whole node failing to start.
//!
//! Privacy: there is deliberately no fallback to public resolvers
//! (Cloudflare, Google), which would send lookups off the local network.

use libp2p::dns::{ResolverConfig, ResolverOpts};
use std::time::{Duration, Instant};

/// How long to wait for a usable system config at startup.
pub const DEFAULT_WAIT: Duration = Duration::from_secs(2);
const POLL: Duration = Duration::from_millis(250);

/// Resolver configuration for `SwarmBuilder::with_dns_config`.
pub struct DnsSetup {
    pub config: ResolverConfig,
    pub opts: ResolverOpts,
    /// Why DNS is unavailable; `None` when the system config was usable.
    pub unavailable: Option<String>,
}

impl DnsSetup {
    pub fn available(&self) -> bool {
        self.unavailable.is_none()
    }

    /// A resolver with no nameservers: every lookup fails, nothing else
    /// is affected.
    pub fn none(reason: impl Into<String>) -> Self {
        Self {
            config: ResolverConfig::new(),
            opts: ResolverOpts::default(),
            unavailable: Some(reason.into()),
        }
    }
}

/// The system resolver config, waiting up to `wait` for it to become
/// usable (e.g. while Wi-Fi is still connecting).
pub async fn system(wait: Duration) -> DnsSetup {
    wait_for(
        || hickory_resolver::system_conf::read_system_conf().map_err(|e| e.to_string()),
        wait,
        POLL,
    )
    .await
}

async fn wait_for(
    mut read: impl FnMut() -> Result<(ResolverConfig, ResolverOpts), String>,
    wait: Duration,
    poll: Duration,
) -> DnsSetup {
    let deadline = Instant::now() + wait;
    loop {
        match read() {
            Ok((config, opts)) => {
                return DnsSetup { config, opts, unavailable: None };
            }
            Err(e) if Instant::now() + poll <= deadline => {
                tracing::debug!("system DNS config not usable yet ({e}), retrying");
                tokio::time::sleep(poll).await;
            }
            Err(e) => {
                tracing::warn!("DNS unavailable, continuing without it: {e}");
                return DnsSetup::none(e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use libp2p::{Transport, core::transport::DialOpts, multiaddr::Multiaddr};

    fn ok() -> Result<(ResolverConfig, ResolverOpts), String> {
        Ok((ResolverConfig::default(), ResolverOpts::default()))
    }

    #[tokio::test]
    async fn immediate_success() {
        let setup = wait_for(ok, Duration::from_secs(1), Duration::from_millis(10)).await;
        assert!(setup.available());
    }

    #[tokio::test]
    async fn retries_until_config_appears() {
        let mut calls = 0;
        let setup = wait_for(
            || {
                calls += 1;
                if calls < 3 { Err("no nameservers found in config".into()) } else { ok() }
            },
            Duration::from_secs(1),
            Duration::from_millis(10),
        )
        .await;
        assert!(setup.available());
        assert_eq!(calls, 3);
    }

    #[tokio::test]
    async fn gives_up_without_failing() {
        let start = Instant::now();
        let setup = wait_for(
            || Err("no nameservers found in config".into()),
            Duration::from_millis(100),
            Duration::from_millis(10),
        )
        .await;
        assert!(!setup.available());
        assert!(setup.unavailable.unwrap().contains("no nameservers"));
        assert!(start.elapsed() < Duration::from_secs(1));
    }

    /// The offline fallback must only break DNS dials: building the
    /// transport works, and a `/dns4` dial errors instead of panicking.
    #[tokio::test]
    async fn empty_resolver_fails_dns_dials_only() {
        let setup = DnsSetup::none("offline");
        let mut transport = libp2p::dns::tokio::Transport::custom(
            libp2p::tcp::tokio::Transport::new(libp2p::tcp::Config::default()),
            setup.config,
            setup.opts,
        );
        let addr: Multiaddr = "/dns4/example.invalid/tcp/80".parse().unwrap();
        let dial = transport.dial(
            addr,
            DialOpts {
                role: libp2p::core::Endpoint::Dialer,
                port_use: libp2p::core::transport::PortUse::New,
            },
        );
        let failed = match dial {
            Ok(fut) => tokio::time::timeout(Duration::from_secs(5), fut)
                .await
                .expect("DNS dial must fail fast without nameservers")
                .is_err(),
            Err(_) => true,
        };
        assert!(failed);
    }
}
