#### Llung (working title may change)

A p2p chat app / agent harness using distributed hash tables, kademlia, and gossipsub protocols to connect peers without a server. Agents can participate in the group chats, and will eventually have access to WASM sandboxes to code their own tools, which the human chat participants can also use, of course. This is inspired by pi coder (the chats are stored as trees and will be able to branch) and the deepseek agent harness architecture.

Things to test: are the libp2p protocols stable? Can participants share media / files over the network? Can the agents share WASM tools with each other over the network?

Right now its a TUI, but eventually, I would like to have a SwiftUI for iOS. I have no idea if WASM tools would work on iOS. That might be a no-no.

License: MIT 
