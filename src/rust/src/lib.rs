//! lib.rs – public R bindings & orchestration layer
//! =================================================
//! This crate exposes several Rust back‑ends to R via **extendr**:
//! 
//! * MAGIC diffusion kernel    → `diffuse_expr_r()` magic.rs
//! * Sparse matrix conversion   → `convert_dgcmatrix_to_csr()`
//! * Sequence alignment        → `align_rust()` align.rs
//! * Sequence alignment pretty → `pretty()` align.rs
//! * Principal Convex Hull Analytics (PCHA) pcha.rs
//! * Sequence consensus helper, pairwise alignments, etc. getconsensus.rs
//! s
//! 
//! -------------------------------------------------------------------------
//! Copyright © 2025 Scott Furlan – MIT OR Apache‑2.0.

use extendr_api::prelude::*;
use ndarray::{Array2, ArrayView2, ShapeBuilder};
use sprs::CsMat;

// ---- project modules ----------------------------------------------------
mod getconsensus;
mod align;
mod pcha;
mod magic;

use crate::align::Aligner;
use magic::diffuse_expr;            // numerical kernel
use pcha::PchaOptions;        // PCHA options

//=========================================================================//
//  dgCMatrix  ↔︎  sprs::CsMat converters                                   //
//=========================================================================//

/// Convert an R **dgCMatrix** (genes × cells, CSC) → `sprs::CsMat` (CSC) and
/// then flip to **CSR** (cells × genes).  Performs extensive sanity checks so
/// upstream caller can trust the result.

pub fn convert_dgcmatrix_to_csr(mat: &Robj) -> extendr_api::Result<CsMat<f64>> {
    if !mat.inherits("dgCMatrix") {
        return Err(Error::Other("Input must inherit from 'dgCMatrix'".into()));
    }
    let s4: S4 = mat.try_into()?;

    let i = s4
        .get_slot("i")
        .ok_or_else(|| Error::Other("missing 'i'".into()))?
        .as_integer_vector()
        .ok_or_else(|| Error::Other("'i' not integer".into()))?;
    let p = s4
        .get_slot("p")
        .ok_or_else(|| Error::Other("missing 'p'".into()))?
        .as_integer_vector()
        .ok_or_else(|| Error::Other("'p' not integer".into()))?;
    let x = s4
        .get_slot("x")
        .ok_or_else(|| Error::Other("missing 'x'".into()))?
        .as_real_vector()
        .ok_or_else(|| Error::Other("'x' not numeric".into()))?;
    let dim = s4
        .get_slot("Dim")
        .ok_or_else(|| Error::Other("missing 'Dim'".into()))?
        .as_integer_vector()
        .ok_or_else(|| Error::Other("'Dim' not integer".into()))?;
    if dim.len() < 2 {
        return Err(Error::Other("'Dim' slot must have length 2".into()));
    }
    let (nrows, ncols) = (dim[0] as usize, dim[1] as usize);
    if p.len() != ncols + 1 {
        return Err(Error::Other(format!("length(p)={} != ncols+1", p.len()).into()));
    }

    // ----- row indices: shift to 0-based if necessary --------------------
    let mut row_indices: Vec<usize> = i.iter().map(|&v| v as usize).collect();
    if let Some(&max_row) = row_indices.iter().max() {
        if max_row == nrows {            // 1-based indices detected
            for idx in &mut row_indices { *idx -= 1; }
        } else if max_row > nrows - 1 {
            return Err(Error::Other("row index exceeds nrows".into()));
        }
    }

    let col_ptrs: Vec<usize> = p.iter().map(|&v| v as usize).collect();
    let values: Vec<f64> = x.to_vec();

    let csc = CsMat::<f64>::new_csc((nrows, ncols), col_ptrs, row_indices, values);
    Ok(csc.to_other_storage()) // → CSR
}


/// Convert a CSR back to R **dgCMatrix** (genes × cells, CSC).  Because Seurat
/// expects genes in rows, we transpose back to CSC on the way out.
fn _build_dgc_from_csr(mat: &CsMat<f64>) -> extendr_api::Result<Robj> {
    let csc = mat.to_other_storage(); // CSR → CSC

    // Slots: p (indptr), i (indices), x (data)
    let indptr = csc.indptr();
    let p_vec: Vec<i32> = indptr.as_slice().unwrap().iter().map(|&v| v as i32).collect();
    let i_vec: Vec<i32> = csc.indices().iter().map(|&v| v as i32).collect();
    let x_vec: Vec<f64> = csc.data().to_vec();

    let robj = list!(
        Dim      = r!([csc.rows() as i32, csc.cols() as i32]),
        Dimnames = NULL,
        x        = x_vec,
        i        = i_vec,
        p        = p_vec
    );
    call!("new", "dgCMatrix", robj)
}

//=========================================================================//
//  MAGIC diffusion: public R wrapper                                      //
//=========================================================================//

/// @export
#[extendr]
fn diffuse_expr_r(p: Robj, x: Robj, t: i32, chunk: i32) -> extendr_api::Result<List> {
    // Convert P (cells × cells)
    let p_csr = convert_dgcmatrix_to_csr(&p)?;            // cells × cells (rows = cells)

    // Convert X (genes × cells) – may need transpose to cells × genes
    let mut x_csr = convert_dgcmatrix_to_csr(&x)?;        // genes × cells (rows = genes)
    if x_csr.rows() != p_csr.rows() {          // <-- change here
        if x_csr.cols() == p_csr.rows() {
            x_csr = x_csr.transpose_view().to_owned();
        } else {
            return Err(Error::Other(
                "X must be genes×cells or cells×genes".into()));
        }
    }
    // Call numerical kernel
    let out_csr = diffuse_expr(&p_csr, &x_csr, t as usize, chunk as usize);

    // Back to dgCMatrix for R (transpose back so genes are rows)
    let out_csr_gene_rows = if out_csr.rows() == p_csr.cols() {
        out_csr.transpose_view().to_owned()
    } else {
        out_csr
    };

    // build_dgc_from_csr(&out_csr_gene_rows)
    // after you have `out_csr` (cells × genes, CSR)
    let csc = out_csr_gene_rows.to_other_storage();            // -> CSC so genes are rows

    // grab raw slices
    let p_vec:  Vec<i32> = csc.indptr()
                            .as_slice().unwrap()
                            .iter().map(|&v| v as i32).collect();
    let i_vec:  Vec<i32> = csc.indices()
                            .iter().map(|&v| v as i32).collect();
    let x_vec:  Vec<f64> = csc.data().to_vec();
    let dim_vec = vec![csc.rows() as i32, csc.cols() as i32];

    // **return a simple list**
    Ok(list!(
        p   = p_vec,
        i   = i_vec,
        x   = x_vec,
        Dim = dim_vec         // no Dimnames; add if you have them
    ))
}


//=========================================================================//
//  Test helper                                                             //
//=========================================================================//

#[extendr]
fn test_conversion(x: Robj) -> extendr_api::Result<bool> {
    convert_dgcmatrix_to_csr(&x)?; // returns Err on failure
    Ok(true) // explicit TRUE so tryCatch gets logical instead of NULL
}


/// @export
#[extendr]
fn pcha_rust(
    input_mat: Robj,
    k: Robj,
    c_init: Robj,
    s_init: Robj,
    max_iter_arg: Robj,
    conv_crit_arg: Robj
) -> Robj {
    // Convert input to Rust ndarray
    let mat: RMatrix<f64> = input_mat
        .try_into()
        .expect("`input_mat` must be a numeric matrix");
    let (nrow, ncol) = (mat.nrows(), mat.ncols());

    // Parse k
    let k = if let Some(v) = k.as_real_vector() {
        // eprintln!("using k = {:?}; processed as real vector", k);
        v[0] as usize
    } else if let Some(v) = k.as_integer_vector() {
        // eprintln!("using k = {:?}; processed as integer vector", k);
        v[0] as usize
    } else {
        panic!("`k` must be numeric or integer");
    };
    assert!(k <= ncol, "`k` (k) must be ≤ number of columns of input_mat");

    // Parse max_iter
    let max_iter_arg = if let Some(v) = max_iter_arg.as_real_vector() {
        // eprintln!("using max_iter = {:?}; processed as real vector", max_iter_arg);
        v[0] as usize
    } else if let Some(v) = max_iter_arg.as_integer_vector() {
        // eprintln!("using max_iter = {:?}; processed as integer vector", max_iter_arg);
        v[0] as usize
    } else {
        panic!("`max_iter` must be numeric or integer");
    };
    assert!(max_iter_arg > 0, "`max_iter` must be > 0");

    // Parse conv_crit
    let conv_crit_arg = if let Some(v) = conv_crit_arg.as_real_vector() {
        // eprintln!("using conv_crit = {:?}; processed as real vector", conv_crit_arg);
        v[0]
    } else if let Some(v) = conv_crit_arg.as_integer_vector() {
        // eprintln!("using conv_crit = {:?}; processed as integer vector", conv_crit_arg);
        v[0] as f64
    } else {
        panic!("`conv_crit` must be numeric or integer");
    };
    assert!(conv_crit_arg > 0.0, "`conv_crit` must be > 0");
    assert!(conv_crit_arg < 1.0, "`conv_crit` must be < 1");
    assert!(conv_crit_arg.is_finite(), "`conv_crit` must be finite");

    // Build the data array (p × n)
    let arr = ArrayView2::from_shape((nrow, ncol).f(), mat.data())
        .unwrap()
        .to_owned();

        // Build options
    
    let mut opts = PchaOptions::default();
    opts.max_iter  = max_iter_arg;    // from R
    opts.conv_crit = conv_crit_arg;   // from R

    // Parse optional c_init
    if !c_init.is_null() {
        // Convert to Array2<f64>
        let rmat: RMatrix<f64> = c_init
            .try_into()
            .expect("`c_init` must be a numeric matrix of shape (n × k)");
        let (cr, cc) = (rmat.nrows(), rmat.ncols());
        assert_eq!(cr, ncol, "`c_init` must have ncol(input_mat) rows");
        assert_eq!(cc, k,     "`c_init` must have k columns");
        opts.c_init = Some(ArrayView2::from_shape((cr, cc).f(), rmat.data())
            .unwrap()
            .to_owned());
    }
    if !s_init.is_null() {
        // Convert to Array2<f64>
        let rmat: RMatrix<f64> = s_init
                .try_into()
                .expect("`s_init` must be a numeric matrix of shape (k × n)");
        let (sr, sc) = (rmat.nrows(), rmat.ncols());
        assert_eq!(sr, k,    "`s_init` must have k rows");
        assert_eq!(sc, ncol, "`s_init` must have ncol(input_mat) columns");
        opts.s_init = Some(ArrayView2::from_shape((sr, sc).f(), rmat.data())
                .unwrap()
                .to_owned());
    }
    // Check for valid options
    // eprintln!("⚙️  PCHA starting with max_iter={}  conv_crit={}", opts.max_iter, opts.conv_crit);
    // Call into Rust PCHA
    let result = pcha::pcha(&arr, k, None, None, Some(opts))
        .expect("PCHA failed");

    // Return as an R list
    list!(
        C       = robj_matrix(&result.c),
        S       = robj_matrix(&result.s),
        XC      = robj_matrix(&result.xc),
        sse     = result.sse,
        varExpl = result.var_expl
    )
    .into()
}

fn robj_matrix(a: &Array2<f64>) -> Robj {
    let (nrow, ncol) = a.dim();
    let mut data:  Vec<f64> = Vec::with_capacity(nrow * ncol);
    for col in a.columns() { data.extend(col); }      // col-major copy
    let robj = r!(data);
    robj.set_attrib("dim", r!([nrow as i32, ncol as i32])).unwrap();
    robj
}


//------------------ consensus & alignment -------------------------------//

/// @export
#[extendr]
fn getconsensus(rstring: Robj, index_add: Robj) -> Robj {
    let seq = rstring.as_str().unwrap().to_string();
    let idx = *index_add.as_real_vector().unwrap().first().unwrap() as usize;
    getconsensus::getconsensus(seq, idx).into_iter().collect_robj()
}

/// @export
#[extendr]
fn align_rust(rstring1: Robj, rstring2: Robj, atype: Robj, verbose: Robj) -> Robj {
    let atype = atype.as_str_vector().unwrap().first().unwrap().to_string();
    let x = rstring1.as_str_vector().unwrap().first().unwrap().to_string().into_bytes();
    let y = rstring2.as_str_vector().unwrap().first().unwrap().to_string().into_bytes();
    let verbose = verbose.as_logical_vector().unwrap().first().unwrap().to_bool();
    let score_fn = |a: u8, b: u8| if a == b { 1 } else { -1 };
    let mut aln = Aligner::with_capacity(x.len(), y.len(), -5, -1, &score_fn);
    let result = match atype.as_str() {
        "global" => aln.global(&x, &y),
        "local"  => aln.local(&x, &y),
        _         => aln.semiglobal(&x, &y),
    };
    if verbose { println!("alignment: {:?}", result); }
    result.score.into_robj()
}

/// @export
#[extendr]
fn pretty(rstring1: Robj, rstring2: Robj, atype: Robj, verbose: Robj) {
    let atype = atype.as_str_vector().unwrap().first().unwrap().to_string();
    let x = rstring1.as_str_vector().unwrap().first().unwrap().to_string().into_bytes();
    let y = rstring2.as_str_vector().unwrap().first().unwrap().to_string().into_bytes();
    let verbose = verbose.as_logical_vector().unwrap().first().unwrap().to_bool();
    let score_fn = |a: u8, b: u8| if a == b { 1 } else { -1 };
    let mut aln = Aligner::with_capacity(x.len(), y.len(), -5, -1, &score_fn);
    let result = match atype.as_str() {
        "global" => aln.global(&x, &y),
        "local"  => aln.local(&x, &y),
        _         => aln.semiglobal(&x, &y),
    };
    if verbose { println!("{}", result.pretty(&x, &y, 10)); }
}

//=========================================================================//
//  extendr export table                                                   //
//=========================================================================//

extendr_module! {
    mod rustytools;
    fn test_conversion;
    fn diffuse_expr_r;
    fn getconsensus;
    fn align_rust;
    fn pcha_rust;
}
