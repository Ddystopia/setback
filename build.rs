fn main() {
    let mut build = cc::Build::new();
    build
        .file("src/setback.c")
        // `setback_longjmp` runs on the recovery stack, so the shim is
        // optimized for size in every profile.
        .opt_level_str("s")
        // Keep a frame pointer to ease debugging of the shim.
        .flag_if_supported("-fno-omit-frame-pointer");

    build.compile("setback");

    println!("cargo:rerun-if-changed=src/setback.c");
}
