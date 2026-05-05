#[tokio::main]
async fn main() {
    let addr = "127.0.0.1:3000";
    println!("rchess running at http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, rchess::web::router()).await.unwrap();
}
