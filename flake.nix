{
  description = "Comparing denomamba-jax and denomamba-torch against RED-CNN for low-dose CT.";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs =
    { nixpkgs, ... }:
    let
      inherit (nixpkgs) lib;
      forAllSystems = lib.genAttrs lib.systems.flakeExposed;
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          default = pkgs.mkShell {
            packages = [
              pkgs.python3
              pkgs.uv
              pkgs.ty
              pkgs.ruff
              pkgs.nodejs # Add Node.js for language servers
              pkgs.nix-ld # Add the nix-ld compatibility layer
            ];

            NIX_LD_LIBRARY_PATH = lib.makeLibraryPath [
              pkgs.glibc
              pkgs.zlib
              pkgs.stdenv.cc.cc.lib
            ];

            shellHook = ''
              unset PYTHONPATH
              uv sync
              . .venv/bin/activate
            '';
          };
        }
      );
    };
}
