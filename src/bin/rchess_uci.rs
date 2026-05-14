fn main() {
    // Ignore SIGPIPE so that a GUI closing its end of the pipe returns EPIPE
    // from write() instead of delivering a fatal signal.  The UCI loop then
    // handles the error and exits cleanly rather than panicking.
    #[cfg(unix)]
    unsafe { libc::signal(libc::SIGPIPE, libc::SIG_IGN); }

    let use_nnue = !std::env::args().any(|a| a == "--no-nnue");
    rchess::uci::run(use_nnue);
}
