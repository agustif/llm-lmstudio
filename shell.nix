let
  pkgs = import <nixpkgs> { };
in
pkgs.mkShell {
  # Basic development environment with Python 3.13 and Rust toolchain
  buildInputs = [
    # maturin / py03 does not support python 3.14 yet
    pkgs.python313
    pkgs.uv
  ];

  # Optional: make cargo and rustup available in the shell and ensure a stable toolchain
  shellHook = ''
    uv sync --all-extras
    source ".venv/bin/activate"
  '';
}
