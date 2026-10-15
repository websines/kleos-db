#!/usr/bin/env bash
set -e

THISDIR=$(dirname $0)
cd $THISDIR

export SWIFT_BRIDGE_OUT_DIR="$(pwd)/generated"
# Build the project for the desired platforms:
cargo build -p kleos-swift -F compact --target x86_64-apple-darwin --release
cargo build -p kleos-swift -F compact --target aarch64-apple-darwin --release
mkdir -p ../target/universal-macos/release

lipo \
    ../target/aarch64-apple-darwin/release/libkleos_swift.a \
    ../target/x86_64-apple-darwin/release/libkleos_swift.a -create -output \
    ../target/universal-macos/release/libkleos_swift.a

cargo build -p kleos-swift -F compact --target aarch64-apple-ios --release
cargo build -p kleos-swift -F compact --target x86_64-apple-ios --release
cargo build -p kleos-swift -F compact --target aarch64-apple-ios-sim --release

mkdir -p ../target/universal-ios/release

lipo \
    ../target/aarch64-apple-ios-sim/release/libkleos_swift.a \
    ../target/x86_64-apple-ios/release/libkleos_swift.a -create -output \
    ../target/universal-ios/release/libkleos_swift.a

swift-bridge-cli create-package \
    --bridges-dir ./generated \
    --out-dir CozoSwiftBridge \
    --ios ../target/aarch64-apple-ios/release/libkleos_swift.a \
    --simulator ../target/universal-ios/release/libkleos_swift.a \
    --macos ../target/universal-macos/release/libkleos_swift.a \
    --name CozoSwiftBridge

cp CozoDB.swift CozoSwiftBridge/Sources/CozoSwiftBridge
cp Package.swift CozoSwiftBridge/
cp CozoSwiftBridge.podspec CozoSwiftBridge/

VERSION=$(cat ../VERSION)

cd CozoSwiftBridge

tar cvzf ../../release/CozoSwiftBridge-$VERSION.tgz .

cd ..