fn main() {
    let mut build = cc::Build::new();
    build
        .file("src/setback.c")
        // Keep a frame pointer to ease debugging of the shim.
        .flag_if_supported("-fno-omit-frame-pointer");

    build.compile("setback");

    println!("cargo:rerun-if-changed=src/setback.c");
}
