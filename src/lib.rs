use amd::{order, Control, Status};
use pyo3::prelude::*;
use rlu::{factor, solve, Options};

/// Performs LU factorization of a sparse matrix in compressed column format
/// and solves for one or more right-hand-side vectors.
#[pyfunction]
fn factor_solve(
    n: usize,
    rowind: Vec<usize>,
    colptr: Vec<usize>,
    nz: Vec<f64>,
    b: Vec<f64>,
    // rhs: Vec<Vec<f64>>,
    trans: bool,
) -> PyResult<Vec<f64>> {
    let control = Control::default();

    let (p, _p_inv, info) = order::<usize>(n, &colptr, &rowind, &control).unwrap();
    assert_eq!(info.status, Status::OK);

    let options = Options::default();
    let lu = factor(n, &rowind, &colptr, &nz, Some(&p), &options).unwrap();

    let mut x = b.clone();
    let mut rhs: Vec<&mut [f64]> = vec![&mut x];

    solve(&lu, &mut rhs, trans).unwrap();

    Ok(x)
}

/// Provides sparse LU factorization with partial pivoting as described in
/// "Sparse Partial Pivoting in Time Proportional to Arithmetic Operations"
/// by John R. Gilbert and Tim Peierls.
#[pymodule]
fn pyrlu(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(factor_solve, m)?)?;
    Ok(())
}
