# NN Model to CPP Tests
Here are some tests for compiling neural network models to c++ code

## Generate mhlo from jax
First run `python test2.py` to generate a mhlo file

## Setup emitc repo
I used the emitc repo from https://github.com/iml130/mlir-emitc to compile mhlo to c++.

I used clang11 in ubuntu 18.04 inside WSL2:
```bash
# Install the GPG Key for https://apt.llvm.org/
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -

#dd the repo for Clang 11 stable-old for Ubuntu 18.04 Bionic
echo "deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-11 main" | sudo tee -a /etc/apt/sources.list
sudo apt-get update

sudo apt-get install libllvm-11-ocaml-dev libllvm11 llvm-11 llvm-11-dev llvm-11-doc llvm-11-examples llvm-11-runtime \
clang-11 clang-tools-11 clang-11-doc libclang-common-11-dev libclang-11-dev libclang1-11 clang-format-11 clangd-11 \
libfuzzer-11-dev lldb-11 lld-11 libc++-11-dev libc++abi-11-dev libomp-11-dev -y

# set the following env vars:
export CC=/usr/bin/clang
export CXX=/usr/bin/clang++
```
    
Follow the instructions provided there. But i needed to modify the `build_tools/build_mlir.sh` to include
`-DLLVM_ENABLE_OCAMLDOC=Off` in the first cmake call. Otherwise i could not install llvm. For some reason ocaml is not 
installed in the specified directory and it will not build the ocaml docs anyway. If this flag is not enough because of permission errors
on copying set write permissions for all in /usr/lib. Don't forget to change that back.

If your system can't handle the multiple compile threads, modify the line `cmake --build "$build_dir" --target install` to
`cmake --build "$build_dir" --target install -- -j 2`

## Run conversions
Inside mlir-emitc/build/bin dir run:
```bash
./emitc-opt --insert-emitc-mhlo-include \
--convert-mhlo-region-ops-to-emitc \
--convert-mhlo-to-emitc \
--insert-emitc-std-include \
--convert-std-to-emitc example.mhlo > example.emitc
```

```bash
./emitc-translate --mlir-to-cpp example.emitc > model_generated.h
```