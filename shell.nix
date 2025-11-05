{ pkgs ? import <nixpkgs> { } }:

let
    pyfastnoisesimd = import ./pyfastnoisesimd.nix {
        inherit (pkgs) lib fetchPypi;
        buildPythonPackage = pkgs.python312Packages.buildPythonPackage;
        numpy = pkgs.python312Packages.numpy;
    };
in

with pkgs;

mkShell {
    buildInputs = [
        python312
        python312Packages.pyopengl
        python312Packages.pyopengl-accelerate
        python312Packages.glfw
        python312Packages.pyglm
        python312Packages.numpy
        python312Packages.pillow
        pyfastnoisesimd
    ];
}
