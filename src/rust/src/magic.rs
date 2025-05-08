//! magic.rs  –  back‑end diffusion kernel (no R / extendr types)
//! ============================================================
//! Computes **P^t · X** where:
//! * `P` – cells × cells, CSR, row‑stochastic
//! * `X` – cells × genes, CSR
//!
//! Strategy: blocked sparse‑dense multiply.
//! 1. For gene indices `[g_start, g_end)`, copy that slice of `X` into a
//!    row‑major dense scratch `(n_cells × width)`.
//! 2. Accumulate rows: `scratch[i,*] += P[i,j] * scratch[j,*]`.
//! 3. Collect non‑zeros, build COO, compress to CSR.
//!
//! Panics encountered earlier were due to an off‑by‑one in the flat‑buffer
//! index calculation.  Fixed by computing `row * width + col` *after* ensuring
//! `row < n_cells` and `col < width`.
//!
//! © 2025 Scott Furlan – MIT OR Apache‑2.0.

use rayon::prelude::*;
use sprs::{CsMat, TriMat};

/// Top‑level entry: apply diffusion for `n_steps`.
pub fn diffuse_expr(p: &CsMat<f64>, x: &CsMat<f64>, n_steps: usize, chunk: usize) -> CsMat<f64> {
    assert_eq!(p.rows(), p.cols(), "P must be square (cells × cells)");
    assert_eq!(p.cols(), x.rows(), "P cols must equal X rows (cells)");

    let mut cur = x.clone();
    for _ in 0..n_steps {
        cur = csr_spmm_blocked(p, &cur, chunk.max(1));
    }
    cur
}

/// One multiply: **P · X**.
fn csr_spmm_blocked(p: &CsMat<f64>, x: &CsMat<f64>, chunk: usize) -> CsMat<f64> {
    let (n_cells, n_genes) = (x.rows(), x.cols());

    let triplets: Vec<_> = (0..n_genes)
        .into_par_iter()
        .step_by(chunk)
        .flat_map(|g_start| {
            let g_end = (g_start + chunk).min(n_genes);
            let width = g_end - g_start;
            let mut buf = vec![0.0f64; n_cells * width];
            let idx = |row: usize, col: usize| row * width + col;   // row‑major

            // ---- copy X slice → dense scratch
            for (row_i, row) in x.outer_iterator().enumerate() {
                if row_i >= n_cells {        // extra empty row: ignore
                    continue;
                }
                if row_i >= n_cells { panic!("row_i {} >= n_cells {}", row_i, n_cells); }
                for (&col, &val) in row.indices().iter().zip(row.data()) {
                    if col >= g_start && col < g_end {
                        let col_off = col - g_start;
                        if col_off >= width {
                            panic!(
                                "col_off {} (col {}, g_start {}, width {}) >= width",
                                col_off, col, g_start, width
                            );
                        }
                        buf[idx(row_i, col_off)] = val;
                    }
                }
            }

            // 2) dense ← P · dense
            for (i, prow) in p.outer_iterator().enumerate() {
                for (&j, &p_ij) in prow.indices().iter().zip(prow.data()) {
                    if j >= n_cells { continue; } // extra safety against malformed P
                    let base_i = idx(i, 0);
                    let base_j = idx(j, 0);
                    for l in 0..width {
                        buf[base_i + l] += p_ij * buf[base_j + l];
                    }
                }
            }

            // 3) collect non‑zeros in this block
            let mut local = Vec::with_capacity(n_cells * width / 8);
            for col_off in 0..width {
                for row_i in 0..n_cells {
                    let v = buf[idx(row_i, col_off)];
                    if v != 0.0 {
                        local.push((row_i, g_start + col_off, v));
                    }
                }
            }
            local
        })
        .collect();

    // COO → CSR
    let mut tri = TriMat::<f64>::with_capacity((n_cells, n_genes), triplets.len());
    for (r, c, v) in triplets { tri.add_triplet(r, c, v); }
    tri.to_csr()
}

// ------------------- tests -------------------
#[cfg(test)]
mod tests {
    use super::*;
    use sprs::CsMat;

    #[test]
    fn identity_preserves_x() {
        let p = CsMat::<f64>::eye(3);
        let mut tri = TriMat::<f64>::new((3, 4));
        tri.add_triplet(0, 1, 2.0);
        tri.add_triplet(2, 3, 5.0);
        let x = tri.to_csr();
        assert_eq!(diffuse_expr(&p, &x, 3, 2), x);
    }
}
