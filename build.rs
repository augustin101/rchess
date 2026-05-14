fn main() {
    let config = "engine.toml";
    println!("cargo:rerun-if-changed={config}");

    let content   = std::fs::read_to_string(config).unwrap_or_default();
    let nnue_path = parse_nnue_path(&content).unwrap_or_else(|| "networks/nnue.bin".to_string());

    // Bake the configured path into the binary as a compile-time env var.
    println!("cargo:rustc-env=RCHESS_NNUE_PATH={nnue_path}");
    // Rebuild whenever the network file itself changes.
    println!("cargo:rerun-if-changed={nnue_path}");

    // Enable the embed_nnue cfg flag when the Cargo feature is active.
    if std::env::var("CARGO_FEATURE_EMBED_NNUE").is_ok() {
        println!("cargo:rustc-cfg=embed_nnue");
    }
}

/// Minimal parser for the one field we need from engine.toml.
/// Handles lines of the form:  nnue = "path/to/file.bin"
fn parse_nnue_path(toml: &str) -> Option<String> {
    for line in toml.lines() {
        let line = line.trim();
        if line.starts_with('#') { continue; }
        let Some(rest) = line.strip_prefix("nnue") else { continue };
        let rest = rest.trim();
        let Some(rest) = rest.strip_prefix('=') else { continue };
        let val = rest.trim().trim_matches('"').trim_matches('\'');
        if !val.is_empty() {
            return Some(val.to_string());
        }
    }
    None
}
