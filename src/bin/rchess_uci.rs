fn main() {
    let use_nnue = !std::env::args().any(|a| a == "--no-nnue");
    rchess::uci::run(use_nnue);
}
