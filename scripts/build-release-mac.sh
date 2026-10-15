#!/usr/bin/env bash

set -e

VERSION=$(cat ./VERSION)
export MACOSX_DEPLOYMENT_TARGET=10.14

#rm -fr release
mkdir -p release

for TARGET in aarch64-apple-darwin x86_64-apple-darwin; do
  # standalone, c, java, nodejs
  CARGO_PROFILE_RELEASE_LTO=fat cargo build --release -p kleos-bin -p kleos_c -p kleos_java -p kleos-node -F compact -F storage-rocksdb --target $TARGET
  cp target/$TARGET/release/kleos-bin release/kleos-$VERSION-$TARGET # standalone
  cp target/$TARGET/release/libkleos_c.a release/libkleos_c-$VERSION-$TARGET.a # c static
  cp target/$TARGET/release/libkleos_c.dylib release/libkleos_c-$VERSION-$TARGET.dylib # c dynamic
  cp target/$TARGET/release/libkleos_java.dylib release/libkleos_java-$VERSION-$TARGET.dylib # java
  cp target/$TARGET/release/libkleos_node.dylib release/libkleos_node-$VERSION-$TARGET.dylib # nodejs

  # python
  cd kleos-lib-python
  CARGO_PROFILE_RELEASE_LTO=fat PYO3_NO_PYTHON=1 maturin build -F compact -F storage-rocksdb --release --strip --target $TARGET
  cd ..
done

# copy python
cp target/wheels/*.whl release/

# swift
cd kleos-lib-swift
CARGO_PROFILE_RELEASE_LTO=fat ./build-rust.sh
cd ..

# with TiKV
#for TARGET in aarch64-apple-darwin x86_64-apple-darwin; do
#  CARGO_PROFILE_RELEASE_LTO=fat cargo build --release -p kleos-bin \
#    -F compact -F storage-rocksdb -F storage-tikv -F storage-sled --target $TARGET
#  cp target/$TARGET/release/kleos-bin release/cozo_all-$VERSION-$TARGET # standalone
#done

# WASM
cd kleos-lib-wasm
CARGO_PROFILE_RELEASE_LTO=fat ./build.sh
cd ..

cp kleos-lib-c/kleos_c.h release/

zip release/cozo_wasm-$VERSION-wasm32-unknown-unknown.zip $(find kleos-lib-wasm/pkg)