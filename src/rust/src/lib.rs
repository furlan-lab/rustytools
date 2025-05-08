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

//=========================================================================//
//  PCHA, getconsensus, alignment bindings                                   //
//=========================================================================//

/// Call PCHA from R
///
/// @param input_mat   numeric matrix (variables in rows, samples in columns)
/// @param noc   integer, number of archetypes/components
/// @return list of matrices
/// @export
#[extendr]
fn pcha(input_mat: Robj, noc: Robj) -> Robj {
    let mat: RMatrix<f64> = input_mat.try_into().expect("`x` must be numeric matrix");
    let noc = if let Some(v) = noc.as_real_vector() {
        v[0] as usize
    } else if let Some(v) = noc.as_integer_vector() {
        v[0] as usize
    } else {
        return extendr_api::Error::from("`noc` must be numeric").into();
    };
    let (nrow, ncol) = (mat.nrows(), mat.ncols());
    assert!(noc <= ncol, "k (noc) must be ≤ ncol(x)");

    let arr = ArrayView2::from_shape((nrow, ncol).f(), mat.data()).unwrap().to_owned();
    let res = pcha::pcha(&arr, noc, None, None, None).unwrap();

    list!(
        C       = robj_matrix(&res.c),
        S       = robj_matrix(&res.s),
        XC      = robj_matrix(&res.xc),
        sse     = res.sse,
        varExpl = res.var_expl
    ).into()
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
    fn pcha;
}


// use extendr_api::prelude::*;
// // use extendr_api::wrapper::externalptr::ExternalPtr;

// mod getconsensus;
// mod align;
// mod pcha;
// mod magic;
// use crate::align::Aligner;
// use ndarray::{Array2, ArrayView2};
// use ndarray::ShapeBuilder;
// use sprs::CsMat;
// use magic::diffuse_expr;

// #[extendr]
// fn test_conversion(x: Robj) -> () {
//     // Convert P (cells × cells)
//     let _x_csr: sprs::CsMatBase<f64, usize, Vec<usize>, Vec<usize>, Vec<f64>> = {
//         let csc: sprs::CsMatBase<f64, usize, Vec<usize>, Vec<usize>, Vec<f64>> = convert_dgcmatrix_to_csr(&x).unwrap(); // now cells × genes **CSR**
//         csc
//     };
//     return ()
// }

// #[extendr]
// fn diffuse_expr_r(p: Robj, x: Robj, t: i32, chunk: i32) -> extendr_api::Result<Robj> {
//     // Convert P (cells × cells)
//     let p_csr: sprs::CsMatBase<f64, usize, Vec<usize>, Vec<usize>, Vec<f64>> = convert_dgcmatrix_to_csr(&p)?;

//     // Convert X (genes × cells) → transpose storage to CSR (cells × genes)
//     let x_csr: sprs::CsMatBase<f64, usize, Vec<usize>, Vec<usize>, Vec<f64>> = {
//         let csc: sprs::CsMatBase<f64, usize, Vec<usize>, Vec<usize>, Vec<f64>> = convert_dgcmatrix_to_csr(&x)?; // now cells × genes **CSR**
//         csc
//     };

//     let out_csr: sprs::CsMatBase<f64, usize, Vec<usize>, Vec<usize>, Vec<f64>> = diffuse_expr(&p_csr, &x_csr, t as usize, chunk as usize);
//     build_dgc_from_csr(&out_csr)
// }

// fn build_dgc_from_csr(mat: &CsMat<f64>) -> extendr_api::Result<Robj> {
//     let csc = mat.to_other_storage();
//     let binding = csc.indptr();
//     let indptr: &[usize] = binding.as_slice().expect("REASON");
//     let p_vec: Vec<i32> = indptr.iter().map(|&v| v as i32).collect();
//     let i_vec: Vec<i32> = csc.indices().iter().map(|&x| x as i32).collect();
//     let x_vec: Vec<f64> = csc.data().to_vec();

//     let robj = list!(
//         Dim = r!([csc.rows() as i32, csc.cols() as i32]),
//         Dimnames = NULL,
//         x = x_vec,
//         i = i_vec,
//         p = p_vec
//     );
//     call!("new", "dgCMatrix", robj)
// }


// pub fn convert_dgcmatrix_to_csr(mat: &Robj) -> extendr_api::Result<CsMat<f64>> {
//     // 1) sanity check ---------------------------------------------------------
//     if !mat.inherits("dgCMatrix") {
//         return Err(Error::Other("Input must inherit from 'dgCMatrix'".into()));
//     }

//     // Convert to S4 type to access slots
//     let s4_matrix: S4 = mat.try_into()?;

//     // Extract the components of the dgCMatrix using S4 methods with error handling
//     let i = s4_matrix.get_slot("i")
//         .ok_or_else(|| Error::Other("Failed to get 'i' slot".into()))?
//         .as_integer_vector()
//         .ok_or_else(|| Error::Other("Failed to convert 'i' slot to integer vector".into()))?;
    
//     let p = s4_matrix.get_slot("p")
//         .ok_or_else(|| Error::Other("Failed to get 'p' slot".into()))?
//         .as_integer_vector()
//         .ok_or_else(|| Error::Other("Failed to convert 'p' slot to integer vector".into()))?;
    
//     let x = s4_matrix.get_slot("x")
//         .ok_or_else(|| Error::Other("Failed to get 'x' slot".into()))?
//         .as_real_vector()
//         .ok_or_else(|| Error::Other("Failed to convert 'x' slot to real vector".into()))?;
    
//     let dim = s4_matrix.get_slot("Dim")
//         .ok_or_else(|| Error::Other("Failed to get 'Dim' slot".into()))?
//         .as_integer_vector()
//         .ok_or_else(|| Error::Other("Failed to convert 'Dim' slot to integer vector".into()))?;
    
//     if dim.len() < 2 {
//         return Err(Error::Other("'Dim' slot must have at least 2 elements".into()));
//     }
    
//     let nrows = dim[0] as usize;
//     let ncols = dim[1] as usize;
    
//     // Convert R's 0-based indices to Rust vectors
//     let row_indices: Vec<usize> = i.iter().map(|&idx| idx as usize).collect();
//     let col_ptrs: Vec<usize> = p.iter().map(|&idx| idx as usize).collect();
//     let values: Vec<f64> = x.iter().map(|&val| val as f64).collect();
    
//     // Validate dimensions
//     if col_ptrs.len() != ncols + 1 {
//         return Err(Error::Other(format!(
//             "Column pointer array length ({}) should be number of columns + 1 ({})",
//             col_ptrs.len(), ncols + 1
//         ).into()));
//     }
    
//     // Create a sparse matrix in CSC format using sprs
//     // Using new() instead of new_checked() as it might not be available in your version
//     let sparse_mat: sprs::CsMatBase<f64, usize, Vec<usize>, Vec<usize>, Vec<f64>> = CsMat::new(
//         (nrows, ncols),
//         col_ptrs,
//         row_indices,
//         values,
//     );
    
//     Ok(sparse_mat)

// }






// /// Call PCHA from R
// ///
// /// @param input_mat   numeric matrix (variables in rows, samples in columns)
// /// @param noc   integer, number of archetypes/components
// /// @return list of matrices
// /// @export
// #[extendr]
// fn pcha(input_mat: Robj, noc: Robj) -> Robj {
//     // let input_mat = input_mat.as_matrix().unwrap();
//     let mat: RMatrix<f64> = input_mat
//         .try_into()
//         .expect("`x` must be a numeric matrix");
//     let noc = if let Some(val) = noc.as_real_vector() {
//         val[0] as usize                // double vector
//     } else if let Some(val) = noc.as_integer_vector() {
//         val[0] as usize                // integer vector
//     } else {
//         return extendr_api::Error::from("`noc` must be numeric").into();
//     };
//     let nrow = mat.nrows();
//     let ncol = mat.ncols();
//     assert!(noc  <= ncol, "`k` ({}) must be ≤ ncol(x) ({})", noc, ncol);
//     // R is column-major, ndarray default is row-major → use `from_shape_vec`
//     let arr = ArrayView2::from_shape(
//             (nrow, ncol).f(),               // .f() = column-major stride
//             mat.data(),
//         )
//         .unwrap()
//         .to_owned();                        // now an owned Array2<f64>
    
//     // eprintln!("input_mat: {:?}", arr);
//     // eprintln!("noc: {:?}", noc);
//     let result = pcha::pcha(&arr, noc, None, None, None).unwrap();
//     let list = list!(
//         C        = robj_matrix(&result.c),    // |I| × k
//         S        = robj_matrix(&result.s),    // k × |U|
//         XC       = robj_matrix(&result.xc),   // p × k
//         sse      = result.sse,
//         varExpl  = result.var_expl
//     );
//     list.into()
// }

// fn robj_matrix(a: &Array2<f64>) -> Robj {
//     let (nrow, ncol) = a.dim();
//     let mut data:  Vec<f64> = Vec::with_capacity(nrow * ncol);
//     for col in a.columns() { data.extend(col); }      // col-major copy
//     let robj = r!(data);
//     robj.set_attrib("dim", r!([nrow as i32, ncol as i32])).unwrap();
//     robj
// }



// /// @export
// #[extendr]
// fn getconsensus(rstring: Robj, index_add: Robj) -> Robj {
//     // let rstring = "NNNNNNNNATGCGTANNNNNNNTTGACNNNNNNNN".to_string();
//     let rstring = rstring.as_str().unwrap().to_string();
//     let index_add = *index_add.as_real_vector().unwrap().first().unwrap() as usize;
//     let output = getconsensus::getconsensus(rstring, index_add);
//     output.into_iter().collect_robj()
// }

// /// @export
// #[extendr]
// fn align_rust(rstring1: Robj, rstring2: Robj, atype: Robj, verbose: Robj) -> Robj {
//     // eprint!("rstring1: {:?}, rstring2: {:?}", rstring1, rstring2);
//     let atype = atype.as_str_vector().unwrap().first().unwrap().to_string();
//     let x = rstring1.as_str_vector().unwrap().first().unwrap().to_string().into_bytes();
//     let y = rstring2.as_str_vector().unwrap().first().unwrap().to_string().into_bytes();
//     let score = |a: u8, b: u8| if a == b { 1i32 } else { -1i32 };
//     let verbose = verbose.as_logical_vector().unwrap().first().unwrap().to_bool();
//     // gap open score: -5, gap extension score: -1
//     let mut aligner = Aligner::with_capacity(x.len(), y.len(), -5, -1, &score);
//     if atype=="global" {
//         let alignment = aligner.global(&x, &y);
//         if verbose {
//             println!("alignment: {:?}", alignment);
//         }
//         return alignment.score.into_robj();
//     } else if atype=="local" {
//         let alignment = aligner.local(&x, &y);
//         if verbose {
//             println!("alignment: {:?}", alignment);
//         }
//         return alignment.score.into_robj();
//     } else {
//         let alignment = aligner.semiglobal(&x, &y);
//         if verbose {
//             println!("alignment: {:?}", alignment);
//         }
//         return alignment.score.into_robj();
//     }
// }

// /// @export
// #[extendr]
// fn pretty(rstring1: Robj, rstring2: Robj, atype: Robj, verbose: Robj) {
//     // eprint!("rstring1: {:?}, rstring2: {:?}", rstring1, rstring2);
//     let atype = atype.as_str_vector().unwrap().first().unwrap().to_string();
//     let x = rstring1.as_str_vector().unwrap().first().unwrap().to_string().into_bytes();
//     let y = rstring2.as_str_vector().unwrap().first().unwrap().to_string().into_bytes();
//     let score = |a: u8, b: u8| if a == b { 1i32 } else { -1i32 };
//     let verbose = verbose.as_logical_vector().unwrap().first().unwrap().to_bool();
//     // gap open score: -5, gap extension score: -1
//     let mut aligner = Aligner::with_capacity(x.len(), y.len(), -5, -1, &score);
//     if atype=="global" {
//         let alignment = aligner.global(&x, &y);
//         if verbose {
//             println!("alignment: {:?}", alignment.pretty(&x, &y, 10));
//         }
//     } else if atype=="local" {
//         let alignment = aligner.local(&x, &y);
//         if verbose {
//             println!("alignment: {:?}", alignment.pretty(&x, &y, 10));
//         }
//     } else {
//         let alignment = aligner.semiglobal(&x, &y);
//         if verbose {
//             println!("alignment: {:?}", alignment.pretty(&x, &y, 10));
//         }
//     }
// }

// // Macro to generate exports.
// // This ensures exported functions are registered with R.
// // See corresponding C code in `entrypoint.c`.
// extendr_module! {
//     mod rustytools;
//     fn test_conversion;
//     fn diffuse_expr_r;
//     fn getconsensus;
//     fn align_rust;
//     fn pcha;
// }
