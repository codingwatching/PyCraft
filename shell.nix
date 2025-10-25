{ pkgs ? import <nixpkgs> { } }:

with pkgs;

mkShell rec {
    buildInputs = [
        python313
        python313Packages.pyopengl
        python313Packages.pyopengl-accelerate
        python313Packages.glfw
        python313Packages.pyglm
        python313Packages.numpy
        python313Packages.pillow
        python313Packages.noise
    ];
    LD_LIBRARY_PATH = lib.makeLibraryPath buildInputs;
}
