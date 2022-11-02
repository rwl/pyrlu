use amd::{order, Control, Status};
use numpy::{Complex64, Element, PyReadonlyArray1, PyReadwriteArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rlu::{factor, solve, Options, Scalar};

/// Performs LU factorization of a sparse matrix in compressed column format
/// and solves for one or more right-hand-side vectors.
#[pyfunction]
#[pyo3(text_signature = "(n, rowind, colptr, nz, b, trans, /)")]
fn factor_solve(
    n: i32,
    rowind: PyReadonlyArray1<i32>,
    colptr: PyReadonlyArray1<i32>,
    nz: PyReadonlyArray1<f64>,
    b: PyReadwriteArrayDyn<f64>,
    trans: bool,
) -> PyResult<()> {
    order_factor_solve(n, rowind, colptr, nz, b, trans)
}

/// Performs LU factorization of a sparse complex matrix in compressed column format
/// and solves for one or more complex right-hand-side vectors.
#[pyfunction]
#[pyo3(text_signature = "(n, rowind, colptr, nz, b, trans, /)")]
fn z_factor_solve(
    n: i32,
    rowind: PyReadonlyArray1<i32>,
    colptr: PyReadonlyArray1<i32>,
    nz: PyReadonlyArray1<Complex64>,
    b: PyReadwriteArrayDyn<Complex64>,
    trans: bool,
) -> PyResult<()> {
    order_factor_solve(n, rowind, colptr, nz, b, trans)
}

fn order_factor_solve<S: Element + Scalar>(
    n: i32,
    rowind: PyReadonlyArray1<i32>,
    colptr: PyReadonlyArray1<i32>,
    nz: PyReadonlyArray1<S>,
    mut b: PyReadwriteArrayDyn<S>,
    trans: bool,
) -> PyResult<()> {
    let a_i = rowind
        .as_slice()
        .map_err(|err| PyValueError::new_err(format!("rowind: {}", err)))?;
    let a_p = colptr
        .as_slice()
        .map_err(|err| PyValueError::new_err(format!("colptr: {}", err)))?;

    let control = Control::default();

    let (p, _p_inv, info) = order::<i32>(n, a_p, a_i, &control)
        .map_err(|err| PyValueError::new_err(format!("amd status: {:?}", err)))?;
    assert_eq!(info.status, Status::OK);

    let a_x = nz
        .as_slice()
        .map_err(|err| PyValueError::new_err(format!("nz: {}", err)))?;

    let options = Options::default();
    let lu = factor(n, a_i, a_p, a_x, Some(&p), &options)
        .map_err(|err| PyValueError::new_err(format!("factor error: {}", err)))?;

    let x = b
        .as_slice_mut()
        .map_err(|err| PyValueError::new_err(format!("b: {}", err)))?;

    solve(&lu, x, trans).map_err(|err| PyValueError::new_err(format!("solve error: {}", err)))?;

    Ok(())
}

/// Provides sparse LU factorization with partial pivoting as described in
/// "Sparse Partial Pivoting in Time Proportional to Arithmetic Operations"
/// by John R. Gilbert and Tim Peierls.
#[pymodule]
fn pyrlu(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(factor_solve, m)?)?;
    m.add_function(wrap_pyfunction!(z_factor_solve, m)?)?;
    Ok(())
}
