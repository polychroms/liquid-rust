fn main() {
    println!("cargo:rerun-if-changed=assets/icons/liquid-rust.ico");
    #[cfg(windows)]
    {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("assets/icons/liquid-rust.ico");
        res.compile().expect("Failed to compile Windows icon resource");
    }
}
