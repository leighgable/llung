#### Implementation notes for Sandboxes / Tools

Use a mixture of Wasm tools and bubblewrap + nix.

Audit the current implementation of the Aukora Kernel clone in `crates/agent/src/aukora/`, and actually put it into effect in the agent crate. Integrate the following into the Aukora Kernal permissions framework. And then add a command, like `/grant` in the tui chat implementation. It might be nice to have `perpetual` and `once` variants? Does the harness ask for permissions, or does the agent? Maybe I need a human-feedback tool?

To provide tools for code editing, data exploration, visualization:
  - glob and grep: no external processes. Use the `ignore` crate (the core library behind ripgrep) and grep-searcher directly inside the harness portion of llung. It handles .gitignore rules automatically and runs at native speed.
  - diff & edit: Use the `similar` crate for generating unified diffs and enforcing exact search-and-replace matches.
  - bash: Nix and bubblewrap subprocesses. Real compilers, test runners, and CLI tools. Use bubblewrap (bwrap) to mount a minimal, isolated Nix store closure and the target workspace directory. This gets us hermetic environments, zero-overhead isolation, resource limits.    
  - Idea: back the workspace with an overlayfs mount or hidden Git worktree commits before every bash tool call so the harness can instantly roll back destructive shell commands, or is this overkill?
  - I want to provide a single STATEFUL repl tool with python repl, chibi-scheme repl and futhark repl. Llung spawns long-running child processes (python3 -i, chibi-scheme, futhark repl) inside the bwrap sandbox and pipes stdin/stdout across turns.
      - Sentinel Delimiters: Interactive REPLs do not exit after a command, so Llung harness needs to know when output finishes. Append a unique sentinel print statement after every input (e.g., print("__END_OF_TURN__") in Python or (display "__END_OF_TURN__") in Scheme) and read stdout until that token appears.
      - Cgroup Isolation Per Process: Even when sharing a filesystem, spawn each persistent REPL inside its own systemd scope so a runaway Futhark memory allocation or Python infinite loop can be killed cleanly without taking down the other interpreters.
  - For Futhark, we will just use the agent for type checking and small test inputs, so the futhark repl will default to the c compiler. But we could put the cuda / opencl backend behind a strict Aukora Kernel Grant, so that I have to approve any run on the GPU.  
  - Llung will need to build on non-NixOS systems in the future, so I will need a `cargo build`.   
  - For bubblewrap, do we use it directly in the nix development flake, or use nixwrap or nix-bwrap?
  - NixOS Driver Directory: Bind /run/opengl-driver read-only. NixOS symlinks active NVIDIA/AMD/Intel ICD loaders, libcuda.so, and OpenCL libraries here.
  - OpenCL ICD Discovery: Bind /etc/OpenCL (if present on your host) or set OCL_ICD_VENDORS=/run/opengl-driver/etc/OpenCL/vendors in the sandbox environment. Without this, futhark opencl will report zero platforms found.
  - Device Nodes: Use --dev-bind (not --ro-bind) for GPU character devices so ioctl calls work.
  - IPC & Shared Memory: Mount /dev/shm (--bind /dev/shm /dev/shm or a fresh tmpfs) because CUDA and OpenCL drivers use shared memory for host-device synchronization.
