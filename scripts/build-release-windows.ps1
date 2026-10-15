$ErrorActionPreference="Stop"

$VERSION = cat .\VERSION
$env:CARGO_PROFILE_RELEASE_LTO = "fat"
$TARGET = "x86_64-pc-windows-msvc"
# $env:PYO3_NO_PYTHON = 1

mkdir -force release > $null

cd kleos-lib-python
maturin build -F compact -F storage-rocksdb --release --strip --target $TARGET
cd ..

cargo build --release -p kleos-bin -p kleos_c -p kleos_java -p kleos-node -F compact -F storage-rocksdb --target $TARGET
cp target/$TARGET/release/kleos-bin.exe release/kleos-$VERSION-$TARGET.exe # standalone
cp target/$TARGET/release/kleos_c.lib release/libkleos_c-$VERSION-$TARGET.lib # c static
cp target/$TARGET/release/kleos_c.dll release/libkleos_c-$VERSION-$TARGET.dll # c dynamic
cp target/$TARGET/release/kleos_java.dll release/libkleos_java-$VERSION-$TARGET.dll # java
cp target/$TARGET/release/kleos_node.dll release/libkleos_node-$VERSION-$TARGET.dll # nodejs

cp target/wheels/*.whl release/

# $TARGET = "x86_64-pc-windows-gnu"
# cargo build --release -p kleos-bin -p kleos_c -p kleos_java -p kleos-node -F compact -F storage-rocksdb --target $TARGET
# cp target/$TARGET/release/kleos-bin.exe release/kleos-$VERSION-$TARGET.exe # standalone
# cp target/$TARGET/release/libkleos_c.a release/libkleos_c-$VERSION-$TARGET.a # c static
# cp target/$TARGET/release/kleos_c.dll release/libkleos_c-$VERSION-$TARGET.dll # c dynamic
# cp target/$TARGET/release/kleos_java.dll release/libkleos_java-$VERSION-$TARGET.dll # java
# cp target/$TARGET/release/kleos_node.dll release/libkleos_node-$VERSION-$TARGET.dll # nodejs
