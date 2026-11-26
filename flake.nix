{
  description = "uv2nix and rust for a chat/harness";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    # llama-cpp = {
    #   url = "github:ggml-org/llama.cpp";
    #   inputs.nixpkgs.follows = "nixpkgs";
    # };
    llama-cpp = {
      url = "github:blueswhen/llama.cpp/sensenova-loop";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    # Prefix sliding attention (Triton kernels) for experiments/models'
    # LatentLM. Source only (flake = false): its own nixpkgs / torch pins
    # stay out. Built by projectOverlay below into BOTH python envs (dev
    # venv + the agent's torch-python). Locked to a commit; after pushing
    # to the prefix repo:
    #   nix flake update prefix-src && direnv reload
    prefix-src = {
      url = "github:leighgable/prefix";
      flake = false;
    };

  };

  outputs =
    {
      nixpkgs,
      pyproject-nix,
      uv2nix,
      pyproject-build-systems,
      rust-overlay,
      llama-cpp,
      prefix-src,
      ...
    }:
    let
      inherit (nixpkgs) lib;
      forAllSystems = lib.genAttrs lib.systems.flakeExposed;

      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

      overlay = workspace.mkPyprojectOverlay {
        sourcePreference = "wheel";
      };

      editableOverlay = workspace.mkEditablePyprojectOverlay {
        root = "$REPO_ROOT";
      };

      pythonSets = forAllSystems (
        system:
        let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [ rust-overlay.overlays.default ];
          };
          python = pkgs.python314;
          hacks = pkgs.callPackage pyproject-nix.build.hacks { };

          projectOverlay = final: prev: {
            # Not in uv.lock: uv2nix cannot read a path dependency outside
            # this flake, so it is built here from the prefix-src input.
            # Runtime deps are listed by hand: torch + triton (kernels),
            # transformers (prefix.modeling; pinned in ./pyproject.toml).
            # prefix's own pyproject also lists accelerate / ipython /
            # pytest, which the kernels never import.
            prefix = final.stdenv.mkDerivation {
              pname = "prefix";
              version = "0.1.0";
              src = prefix-src;
              nativeBuildInputs = [ final.pyprojectHook ] ++ final.resolveBuildSystem { uv-build = [ ]; };
              passthru.dependencies = {
                torch = [ ];
                triton = [ ];
                transformers = [ ];
              };
            };
            torch =
              (hacks.nixpkgsPrebuilt {
                from = python.pkgs.torchWithRocm;
                prev = prev.torch;
              }).overrideAttrs
                (old: {
                  passthru = (old.passthru or { }) // {
                    dependencies = lib.filterAttrs (
                      name: _: !(lib.hasPrefix "nvidia-" name || lib.hasPrefix "cuda-" name)
                    ) (old.passthru.dependencies or { });
                  };
                });
            torchvision =
              (hacks.nixpkgsPrebuilt {
                from = python.pkgs.torchvision;
                prev = prev.torch;
              }).overrideAttrs
                (old: {
                  passthru = (old.passthru or { }) // {
                    dependencies = lib.filterAttrs (
                      name: _: !(lib.hasPrefix "nvidia-" name || lib.hasPrefix "cuda-" name)
                    ) (old.passthru.dependencies or { });
                  };
                });
          };
        in
        (pkgs.callPackage pyproject-nix.build.packages {
          inherit python;
        }).overrideScope
          (
            lib.composeManyExtensions [
              pyproject-build-systems.overlays.wheel
              overlay
              projectOverlay
            ]
          )
      );

      # Shared per-system pieces: the devShell and the packaged app use
      # the SAME nixpkgs, Rust toolchain and repl python closure.
      sharedFor =
        system:
        let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [ rust-overlay.overlays.default ];
          };
          # Python for the agent's repl tool, defined with pyproject-nix:
          # a real pyproject.toml (./python-tool) rendered into a
          # withPackages closure, so adding deps is a pyproject edit.
          pythonToolProject = pyproject-nix.lib.project.loadPyproject {
            projectRoot = ./python-tool;
          };
        in
        {
          inherit pkgs;
          rustToolchain = pkgs.rust-bin.selectLatestNightlyWith (
            toolchain:
            toolchain.default.override {
              extensions = [
                "rust-src"
                "rustfmt"
                "clippy"
              ];
            }
          );
          python_tool = pkgs.python314.withPackages (
            pythonToolProject.renderers.withPackages {
              python = pkgs.python314;
            }
          );
        };

      # The packaged app: `nix build` / `nix run` / a flake input.
      # Launch it from any directory: that directory is the agents'
      # workspace; everything else is located without the cwd.
      llungFor =
        system:
        let
          inherit (sharedFor system) pkgs rustToolchain python_tool;
          inherit (pkgs) lib;
          rustPlatform = pkgs.makeRustPlatform {
            cargo = rustToolchain;
            rustc = rustToolchain;
          };
          # What the agent sandboxes run: resolved from PATH at run time
          # (agent/src/sandbox/closure.rs). Pinning them here makes the
          # sandbox a hermetic store closure even when the app runs
          # outside the devShell — with the system PATH alone it would
          # fall back to (useless, on NixOS) host /usr and /bin binds.
          # systemd-run is deliberately NOT pinned: it must match the
          # running systemd, so it comes from the system PATH (--prefix
          # keeps that searchable after these).
          sandboxRuntime =
            with pkgs;
            [
              bash
              coreutils
              findutils
              gnugrep
              gnused
              gawk
              uv
              git
              # repl interpreters (python, chibi-scheme, futhark + the cc
              # its c backend shells out to). The ROCm torch research
              # interpreter stays devShell-only: too heavy to ship here.
              python_tool
              chibi
              futhark
              gcc
            ]
            ++ lib.optionals stdenv.hostPlatform.isLinux [ bubblewrap ];
        in
        rustPlatform.buildRustPackage {
          pname = "llung";
          version = "0.1.0";
          # Only what the Rust build needs: keeps target/, logs and the
          # python/nix side out of the source hash.
          src = lib.fileset.toSource {
            root = ./llung;
            fileset = lib.fileset.unions [
              ./llung/Cargo.toml
              ./llung/Cargo.lock
              ./llung/crates
              ./llung/runtime-tools
            ];
          };
          cargoLock.lockFile = ./llung/Cargo.lock;
          cargoBuildFlags = [
            "-p"
            "tui"
          ];
          # The suites need bubblewrap, a user systemd, network stubs and
          # model servers; they run in the devShell (`cargo test`).
          doCheck = false;
          nativeBuildInputs = [
            pkgs.makeWrapper
            pkgs.pkg-config
          ];
          # nixpkgs' cargo setup hook un-vendors C libraries on purpose
          # (LIBSQLITE3_SYS_USE_PKG_CONFIG=1, ZSTD_SYS_USE_PKG_CONFIG=1):
          # link nixpkgs' sqlite and zstd, so their security updates flow
          # through nixpkgs. Plain `cargo build` still uses rusqlite's
          # bundled sqlite.
          buildInputs = [
            pkgs.sqlite
            pkgs.zstd
          ];
          postInstall = ''
            # wasm tools next to the binary, where agent::paths finds them
            # (<exe dir>/../share/llung/runtime-tools) — the same
            # exe-relative lookup an app bundle uses on macOS/iOS.
            mkdir -p $out/share/llung
            cp -r runtime-tools $out/share/llung/
            wrapProgram $out/bin/tui --prefix PATH : ${lib.makeBinPath sandboxRuntime}
            ln -s tui $out/bin/llung
          '';
          meta = {
            description = "llung: p2p chat with local, sandboxed agents";
            mainProgram = "llung";
            platforms = lib.platforms.linux;
          };
        };
    in
    {
      devShells = forAllSystems (
        system:
        let
          inherit (sharedFor system) pkgs rustToolchain python_tool;
          pythonSet = pythonSets.${system}.overrideScope editableOverlay;
          virtualenv = pythonSet.mkVirtualEnv "llung-dev-env" (workspace.deps.all // { prefix = [ ]; });
          rocmEnv = pkgs.symlinkJoin {
            name = "rocm-combined";
            paths = with pkgs.rocmPackages; [
              rocblas
              hipblas
              clr # Contains hipcc and the HIP runtime
              clr.icd # Contains the OpenCL ICD
              rocminfo # Useful for verifying ROCm detection
            ];
          };
          llama-cpp-gpu =
            (llama-cpp.packages.${system}.rocm.override { useWebUi = false; }).overrideAttrs
              (old: {
                webui = null;
                cmakeFlags = (old.cmakeFlags or [ ]) ++ [
                  "-DAMDGPU_TARGETS=gfx1150"
                  "-DGPU_TARGETS=gfx1150"
                ];
              });
          # llama-cpp-gpu = pkgs.llama-cpp.override {
          #   rocmSupport = true;
          #   rocmGpuTargets = [ "gfx1150" ];
          # };
          # llama-cpp-gpu = pkgs.llama-cpp-vulkan;
          # python_tool (the repl's python) comes from sharedFor.
          # Phase 0 research harness: the torch trainer interpreter.
          # The SAME uv2nix package set as the dev virtualenv (torch
          # with the ROCm nixpkgs build via hacks.nixpkgsPrebuilt,
          # tensorboard, tbparse, pyyaml, numpy — see ./pyproject.toml),
          # rendered as its own non-editable env so experiments get the
          # installed packages. The wrapper's only job is a distinct
          # binary name: repl deps resolve interpreters by name from
          # PATH, and `python3` must keep meaning the lean python_tool
          # closure.
          python_torch = pythonSets.${system}.mkVirtualEnv "llung-torch-env" (
            workspace.deps.default // { prefix = [ ]; }
          );
          torch-interp = pkgs.runCommand "torch-interp" { } ''
            mkdir -p $out/bin
            ln -s ${python_torch}/bin/python3 $out/bin/torch-python
          '';
        in
        {
          default = pkgs.mkShell.override { stdenv = pkgs.clangStdenv; } {
            buildInputs = [
              rustToolchain
              pkgs.openssl
              pkgs.pkg-config
              rocmEnv
              pkgs.vulkan-tools
              pkgs.clinfo # Useful for verifying GPU detection
              pkgs.ocl-icd # OpenCL loader
              pkgs.perf
              pkgs.rust-analyzer
              pkgs.rustfmt
              # Phase 3 sandbox: proc:bash tool
              pkgs.bubblewrap
            ];
            packages = [
              virtualenv
              python_tool
              torch-interp
              pkgs.uv
              pkgs.nodejs
              pkgs.chibi
              pkgs.futhark
              llama-cpp-gpu
            ];
            env = {
              UV_NO_SYNC = "1";
              UV_PYTHON = pythonSet.python.interpreter;
              UV_PYTHON_DOWNLOADS = "never";
            };
            shellHook = ''
              unset PYTHONPATH
              export REPO_ROOT=$(git rev-parse --show-toplevel)
              # Ensure HIP can find the ROCm path if you use the HIP backend
              export HIP_PATH=${pkgs.rocmPackages.clr}
              # Tell the OpenCL loader where to find the AMD ICD
              export OCL_ICD_VENDORS=${pkgs.rocmPackages.clr.icd}/etc/OpenCL/vendors
              # Ensure libraries can find OpenCL and ROCm at runtime
              export LD_LIBRARY_PATH=${pkgs.ocl-icd}/lib:${rocmEnv}/lib:$LD_LIBRARY_PATH
              export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1             
              export HSA_OVERRIDE_GFX_VERSION="11.5.0"
              # Triton: persist autotune results (prefix kernels re-tune
              # for minutes per process otherwise); keep fp fusion on.
              export TRITON_CACHE_AUTOTUNING=1
              export TRITON_DEFAULT_FP_FUSION=1
            '';
          };
        }
      );

      packages = forAllSystems (system: {
        # The app (`nix run`, `nix build`, or a flake input's
        # packages.${system}.default).
        default = llungFor system;
        llung = llungFor system;
        # The python research env that used to be `default`.
        python-env = pythonSets.${system}.mkVirtualEnv "llung-env" workspace.deps.default;
      });
    };
}

# export PYTHONPATH="$REPO_ROOT/external/nanochat''${PYTHONPATH:+:$PYTHONPATH}"
