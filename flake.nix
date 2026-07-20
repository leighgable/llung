{
  description = "uv2nix and rust for the kitchen";

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
  };

  outputs =
    {
      nixpkgs,
      pyproject-nix,
      uv2nix,
      pyproject-build-systems,
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
          pkgs = nixpkgs.legacyPackages.${system};
          python = pkgs.python314;
          hacks = pkgs.callPackage pyproject-nix.build.hacks { };

          projectOverlay = final: prev: {
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
            #   antlr4-python3-runtime = prev.antlr4-python3-runtime.overrideAttrs (old: {
            #     nativeBuildInputs =
            #       (old.nativeBuildInputs or [ ])
            #       ++ final.resolveBuildSystem {
            #         setuptools = [ ];
            #       };
            #   });
            #   pylatexenc = prev.pylatexenc.overrideAttrs (old: {
            #     nativeBuildInputs =
            #       (old.nativeBuildInputs or [ ])
            #       ++ final.resolveBuildSystem {
            #         setuptools = [ ];
            #       };
            #   });
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
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          pythonSet = pythonSets.${system}.overrideScope editableOverlay;
          virtualenv = pythonSet.mkVirtualEnv "kitch-dev-env" workspace.deps.all;
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
        in
        {
          default = pkgs.mkShell.override { stdenv = pkgs.clangStdenv; } {
            buildInputs = [
              rocmEnv
              pkgs.vulkan-tools
              pkgs.clinfo # Useful for verifying GPU detection
              pkgs.ocl-icd # OpenCL loader
              pkgs.perf
            ];
            packages = [
              virtualenv
              pkgs.uv
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
            '';
          };
        }
      );

      packages = forAllSystems (system: {
        default = pythonSets.${system}.mkVirtualEnv "kitch-env" workspace.deps.default;
      });
    };
}

# export PYTHONPATH="$REPO_ROOT/external/nanochat''${PYTHONPATH:+:$PYTHONPATH}"
