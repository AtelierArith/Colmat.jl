use faer::sparse::{SparseColMat, Triplet};
use faer::sparse::linalg::{SupernodalThreshold, qr::{factorize_symbolic_qr, QrSymbolicParams, SymbolicQr, SymbolicQrRaw}};
fn algorithm(s: &SymbolicQr<usize>) -> &'static str {
    match s.raw() {
        SymbolicQrRaw::Simplicial(_) => "simplicial",
        SymbolicQrRaw::Supernodal(_) => "supernodal",
    }
}
fn main() {
    let a=SparseColMat::<usize,f64>::try_new_from_triplets(2,2,&[
        Triplet{row:0,col:0,val:2.0},Triplet{row:1,col:1,val:3.0}]).unwrap();
    for (threshold,expected) in [(SupernodalThreshold::FORCE_SIMPLICIAL,"simplicial"),(SupernodalThreshold::FORCE_SUPERNODAL,"supernodal")] {
        let s=factorize_symbolic_qr(a.symbolic(),QrSymbolicParams{supernodal_flop_ratio_threshold:threshold,..Default::default()}).unwrap();
        assert_eq!(algorithm(&s),expected);
        assert_eq!(s.nrows(),2);
        println!("selected {expected}");
    }
}
