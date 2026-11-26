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

      # Research harness pieces (Linux + ROCm only). Nix is lazy: nothing
      # here — torch, ROCm, triton, prefix — is evaluated or fetched
      # unless a research output (llung-research, the research devShell,
      # torch-python) is actually built.
      researchFor =
        system:
        let
          inherit (sharedFor system) pkgs;
          # Phase 0 research harness: the torch trainer interpreter.
          # The SAME uv2nix package set as the dev virtualenv (torch
          # with the ROCm nixpkgs build via hacks.nixpkgsPrebuilt,
          # tensorboard, tbparse, pyyaml, numpy — see ./pyproject.toml),
          # rendered as its own non-editable env so experiments get the
          # installed packages. The wrapper's only job is a distinct
          # binary name: repl deps resolve interpreters by name from
          # PATH, and `python3` must keep meaning the lean python_tool
          # closure. (A symlink, not a makeWrapper script: the venv
          # finds its pyvenv.cfg via argv[0].)
          python_torch = pythonSets.${system}.mkVirtualEnv "llung-torch-env" (
            workspace.deps.default // { prefix = [ ]; }
          );
        in
        {
          inherit python_torch;
          torch-interp = pkgs.runCommand "torch-interp" { } ''
            mkdir -p $out/bin
            ln -s ${python_torch}/bin/python3 $out/bin/torch-python
          '';
          # gfx1150 iGPU; ROCm needs the override to pick its kernels.
          gfxVersion = "11.5.0";
        };

      # Rust sources: only what cargo needs, so target/, logs and the
      # python/nix side stay out of the source hash.
      llungSrc = lib.fileset.toSource {
        root = ./llung;
        fileset = lib.fileset.unions [
          ./llung/Cargo.toml
          ./llung/Cargo.lock
          ./llung/crates
          ./llung/runtime-tools
        ];
      };

      # PP-OCRv6 Small, pinned to the exact files and SHA-256s that
      # pdf-inspector's model manifest (vision/models.rs) verifies.
      ocrModelsFor =
        system:
        let
          inherit (sharedFor system) pkgs;
          release = "https://github.com/GreatV/oar-ocr/releases/download/v0.7.0";
          model = name: sha256: {
            inherit name;
            path = pkgs.fetchurl {
              url = "${release}/${name}";
              inherit sha256;
            };
          };
        in
        pkgs.linkFarm "pp-ocrv6-small" [
          (model "pp-ocrv6_small_det.onnx" "d73e0058b7a8086bbd57f3d10b8bcd4ff95363f67e06e2762b5e814fe9c9410e")
          (model "pp-ocrv6_small_rec.onnx" "5435fd747c9e0efe15a96d0b378d5bd157e9492ed8fd80edf08f30d02fa24634")
          (model "ppocrv6_dict.txt" "b5f2bfe2bdd9448429e3e82b51c789775d9b42f2403d082b00662eb77e401c5d")
        ];

      # The pdf_read helper: PDF -> Markdown (pdf-inspector), with
      # PDFium + ONNX Runtime + the OCR models wired in by the wrapper.
      # The agent runs it inside bubblewrap; its whole store closure
      # (libraries, models) is what the sandbox binds read-only.
      llungPdfFor =
        system:
        let
          inherit (sharedFor system) pkgs rustToolchain;
          rustPlatform = pkgs.makeRustPlatform {
            cargo = rustToolchain;
            rustc = rustToolchain;
          };
        in
        rustPlatform.buildRustPackage {
          pname = "llung-pdf";
          version = "0.1.0";
          src = llungSrc;
          cargoLock.lockFile = ./llung/Cargo.lock;
          cargoBuildFlags = [
            "-p"
            "llung-pdf"
          ];
          doCheck = false;
          nativeBuildInputs = [
            pkgs.makeWrapper
            pkgs.pkg-config
            pkgs.removeReferencesTo
          ];
          # The OCR stack's tokenizer (onig_sys): nixpkgs' cargo hook
          # links the system oniguruma instead of vendoring it.
          buildInputs = [ pkgs.oniguruma ];
          # Panic locations embed std's source path, which drags the
          # whole toolchain (~1.6 GB, docs included) into the closure.
          postInstall = ''
            remove-references-to -t ${rustToolchain} $out/bin/llung-pdf
            wrapProgram $out/bin/llung-pdf \
              --set PDFIUM_LIB_PATH ${pkgs.pdfium-binaries}/lib/libpdfium.so \
              --set ORT_DYLIB_PATH ${pkgs.onnxruntime}/lib/libonnxruntime.so \
              --set LLUNG_OCR_MODELS ${ocrModelsFor system}
          '';
          meta = {
            description = "PDF to Markdown (with OCR) for llung's pdf_read tool";
            mainProgram = "llung-pdf";
          };
        };

      # The Rust app, unwrapped: built ONCE and shared by every variant
      # (lean, research, later app bundles), so switching variants never
      # recompiles.
      llungUnwrappedFor =
        system:
        let
          inherit (sharedFor system) pkgs rustToolchain;
          inherit (pkgs) lib;
          rustPlatform = pkgs.makeRustPlatform {
            cargo = rustToolchain;
            rustc = rustToolchain;
          };
        in
        rustPlatform.buildRustPackage {
          pname = "llung-unwrapped";
          version = "0.1.0";
          src = llungSrc;
          cargoLock.lockFile = ./llung/Cargo.lock;
          cargoBuildFlags = [
            "-p"
            "tui"
          ];
          # The suites need bubblewrap, a user systemd, network stubs and
          # model servers; they run in the devShell (`cargo test`).
          doCheck = false;
          nativeBuildInputs = [
            pkgs.pkg-config
            pkgs.removeReferencesTo
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
            # current_exe() resolves symlinks, so the wrapped variants
            # below still find them here.
            mkdir -p $out/share/llung
            cp -r runtime-tools $out/share/llung/
            # std source paths in panic messages would otherwise keep the
            # whole Rust toolchain (~1.6 GB) in the app's closure.
            remove-references-to -t ${rustToolchain} $out/bin/*
          '';
          meta = {
            description = "llung: p2p chat with local, sandboxed agents (unwrapped binary)";
            mainProgram = "tui";
            platforms = lib.platforms.linux;
          };
        };

      # The packaged app: `nix build` / `nix run` / a flake input.
      # Launch it from any directory: that directory is the agents'
      # workspace; everything else is located without the cwd.
      #
      #   research = false → lean: no torch, no ROCm in the closure. The
      #     gpu:torch tool reports torch-python missing at call time.
      #   research = true  → + torch-python (ROCm torch, triton, prefix)
      #     on the sandbox PATH and the iGPU override in the env.
      llungFor =
        system:
        {
          research ? false,
        }:
        let
          inherit (sharedFor system) pkgs python_tool;
          inherit (pkgs) lib;
          unwrapped = llungUnwrappedFor system;
          rs = researchFor system;
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
              # its c backend shells out to; triton's JIT uses gcc too).
              python_tool
              chibi
              futhark
              gcc
              # pdf_read's helper (PDFium + ONNX Runtime OCR, ~0.5 GB).
              (llungPdfFor system)
            ]
            ++ lib.optionals stdenv.hostPlatform.isLinux [ bubblewrap ]
            ++ lib.optionals research [ rs.torch-interp ];
          wrapArgs = [
            "--prefix PATH : ${lib.makeBinPath sandboxRuntime}"
          ]
          ++ lib.optionals research [
            # repl.rs forwards this into the GPU sandbox; --set-default
            # so a different GPU can override it from the environment.
            "--set-default HSA_OVERRIDE_GFX_VERSION ${rs.gfxVersion}"
          ];
        in
        pkgs.symlinkJoin {
          name = "${if research then "llung-research" else "llung"}-${unwrapped.version}";
          paths = [ unwrapped ];
          nativeBuildInputs = [ pkgs.makeWrapper ];
          postBuild = ''
            wrapProgram $out/bin/tui ${lib.concatStringsSep " " wrapArgs}
            ln -s tui $out/bin/llung
          '';
          passthru = { inherit unwrapped research; };
          meta = unwrapped.meta // {
            description =
              "llung: p2p chat with local, sandboxed agents"
              + lib.optionalString research " (+ torch research harness)";
            mainProgram = "llung";
            platforms = if research then [ "x86_64-linux" ] else lib.platforms.linux;
          };
        };
    in
    {
      devShells = forAllSystems (
        system:
        let
          inherit (sharedFor system) pkgs rustToolchain python_tool;
          inherit (pkgs) lib;

          # Everything needed to hack on the app itself: Rust, the
          # sandbox and the lean repl interpreters. No torch, no ROCm.
          appInputs = [
            rustToolchain
            pkgs.openssl
            pkgs.pkg-config
            pkgs.rust-analyzer
            pkgs.rustfmt
          ]
          ++ lib.optionals pkgs.stdenv.hostPlatform.isLinux [
            pkgs.perf
            # Phase 3 sandbox: proc:bash tool
            pkgs.bubblewrap
          ];
          appPackages = [
            python_tool
            # pdf_read helper (iterate on it with LLUNG_PDF_BIN=target/…)
            (llungPdfFor system)
            pkgs.uv
            pkgs.nodejs
            pkgs.chibi
            pkgs.futhark
          ];
          mkShell = pkgs.mkShell.override { stdenv = pkgs.clangStdenv; };

          app = mkShell {
            buildInputs = appInputs;
            packages = appPackages;
          };

          research =
            let
              rs = researchFor system;
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
            in
            mkShell {
              buildInputs = appInputs ++ [
                rocmEnv
                pkgs.vulkan-tools
                pkgs.clinfo # Useful for verifying GPU detection
                pkgs.ocl-icd # OpenCL loader
              ];
              packages = appPackages ++ [
                virtualenv
                rs.torch-interp
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
                export HSA_OVERRIDE_GFX_VERSION="${rs.gfxVersion}"
                # Triton: persist autotune results (prefix kernels re-tune
                # for minutes per process otherwise); keep fp fusion on.
                export TRITON_CACHE_AUTOTUNING=1
                export TRITON_DEFAULT_FP_FUSION=1
              '';
            };
        in
        {
          inherit app;
          # `use flake` (.envrc) picks default: the full research shell
          # where ROCm exists, the lean app shell elsewhere (macOS).
          default = if pkgs.stdenv.hostPlatform.isLinux then research else app;
        }
        // lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux { inherit research; }
      );

      packages = forAllSystems (system: {
        # The app (`nix run`, `nix build`, or a flake input's
        # packages.${system}.default). Lean: no torch.
        default = llungFor system { };
        llung = llungFor system { };
        # `nix run .#llung-research`: + the torch research harness.
        llung-research = llungFor system { research = true; };
        llung-unwrapped = llungUnwrappedFor system;
        # The trainer interpreter on its own (`nix shell .#torch-python`).
        torch-python = (researchFor system).torch-interp;
        # The pdf_read helper on its own (`nix run .#llung-pdf -- x.pdf`).
        llung-pdf = llungPdfFor system;
        # The python research env that used to be `default`.
        python-env = pythonSets.${system}.mkVirtualEnv "llung-env" workspace.deps.default;
      });
    };
}

# export PYTHONPATH="$REPO_ROOT/external/nanochat''${PYTHONPATH:+:$PYTHONPATH}"
