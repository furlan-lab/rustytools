// pcha.rs — final cleaned version
// -----------------------------------------------------------------------------
// Rust implementation of Principal Convex Hull Analysis (PCHA) / Archetypal
// Analysis translated from MATLAB (Morten Mørup, 2010).
// -----------------------------------------------------------------------------
use ndarray::{Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand::{distributions::Uniform, thread_rng, Rng};
use std::error::Error;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
// use qhull::QhBuilder;   // <-- correct types
// use ndarray::s;
// use ndarray_linalg::Determinant;
// use qhull::Vertex;
// use ndarray::linalg::Dot;

/// Optional parameters controlling the optimisation.
#[derive(Debug, Clone)]
pub struct PchaOptions {
    pub max_iter: usize,   // outer‑loop iterations (default 750)
    pub conv_crit: f64,    // relative ΔSSE convergence (default 1e‑6)
    pub delta: f64,        // ℓ₁ relaxation on C (0 → exact simplex)
    pub c_init: Option<Array2<f64>>, // optional initial C (|I| × k)
    pub s_init: Option<Array2<f64>>, // optional initial S (k × |U|)
    pub num_cores: usize
}
impl Default for PchaOptions {
    fn default() -> Self {
        Self { max_iter: 750, conv_crit: 0.000001, delta: 0.0, c_init: None, s_init: None, num_cores: 1 }
    }
}

/// Result struct.
#[derive(Debug, Clone)]
pub struct PchaResult {
    pub xc: Array2<f64>,   // archetypes X_I * C
    pub s:  Array2<f64>,   // noc × |U|
    pub c:  Array2<f64>,   // |I| × noc
    pub sse: f64,          // final residual SSE
    pub var_expl: f64,     // (SST – SSE)/SST
    // pub t_ratio:  Option<f64>,
}

// -----------------------------------------------------------------------------
// Public entry point
// -----------------------------------------------------------------------------
pub fn pcha(
    x: &Array2<f64>,
    noc: usize,
    i_idx: Option<&[usize]>,
    u_idx: Option<&[usize]>,
    opts: Option<PchaOptions>,
    // calc_t_ratio: bool,
) -> Result<PchaResult, Box<dyn Error>> {
    let mut opts = opts.unwrap_or_default();
    // ---- index helpers ------------------------------------------------------
    let i_vec: Vec<usize> = i_idx.map(|v| v.to_vec()).unwrap_or_else(|| (0..x.ncols()).collect());
    let u_vec: Vec<usize> = u_idx.map(|v| v.to_vec()).unwrap_or_else(|| (0..x.ncols()).collect());

    let ni = i_vec.len();
    let nu = u_vec.len();

    let x_i = select_columns(x, &i_vec);
    let x_u = select_columns(x, &u_vec);

    let sst = x_u.mapv(|v| v * v).sum();

    eprintln!("Using {} core(s)", opts.num_cores);
    let _ = ThreadPoolBuilder::new()
        .num_threads(opts.num_cores)
        .build()
        .expect("Failed to build thread pool");
    // ---- initialise C -------------------------------------------------------
    let mut c: Array2<f64> = if let Some(init) = opts.c_init.take() {
        init
    } else {
        let seed = thread_rng().gen_range(0..ni);
        let seeds = furthest_sum(&x_i, noc, seed, &[]);
        let mut m = Array2::<f64>::zeros((ni, noc));
        for (k, &idx) in seeds.iter().enumerate() { m[[idx, k]] = 1.0; }
        m
    };
    let mut xc = x_i.dot(&c);

    // ---- initialise S -------------------------------------------------------
    let mut s: Array2<f64> = if let Some(init) = opts.s_init.take() { init } else {
        let mut rng = thread_rng();
        let mut tmp = Array2::<f64>::random_using((noc, nu), Uniform::new(0.0, 1.0), &mut rng);
        normalise_columns(&mut tmp);
        tmp
    };

    let mut mu_s     = 1.0;
    let mut mu_c     = 1.0;
    let mut mu_alpha = 1.0;

    // ---- helper products & initial SSE --------------------------------------
    let mut sst_mat      = s.dot(&s.t());
    let mut ct_x_t_xc    = xc.t().dot(&xc);
    let mut sse = {
        let term1 = xc.t().dot(&x_u);
        sst - 2.0 * (&term1 * &s).sum() + (&ct_x_t_xc * &sst_mat).sum()
    };

    // quick S refinement (25 inner steps) -------------------------------------
    {
        let x_ct_x = xc.t().dot(&x_u);
        let (s_new, sse_new, mu_s_new, sst_mat_new) =
            s_update(&s, &x_ct_x, &ct_x_t_xc, mu_s, sst, sse, 25);
        s         = s_new;
        sse       = sse_new;
        mu_s      = mu_s_new;
        sst_mat   = sst_mat_new;
    }

    // ---- main loop ----------------------------------------------------------
    let mut iter     = 0;
    let mut d_sse    = f64::INFINITY;
    let mut var_expl = (sst - sse) / sst;

    while d_sse.abs() >= opts.conv_crit * sse.abs()
        && iter < opts.max_iter
        && var_expl < 0.9999 {
        iter += 1;
        let sse_old = sse;

        // ---- C update ----------------------------------------------------
        let x_s_t = x_u.dot(&s.t());
        let (c_new, sse_c, mu_c_new, mu_alpha_new, ct_x_t_xc_new, xc_new) = c_update(
            &x_i, &x_s_t, &xc, &sst_mat, &c, opts.delta,
            mu_c, mu_alpha, sst, sse, 10,
        );
        c           = c_new;
        sse         = sse_c;
        mu_c        = mu_c_new;
        mu_alpha    = mu_alpha_new;
        ct_x_t_xc   = ct_x_t_xc_new;
        xc          = xc_new;

        // ---- S update ----------------------------------------------------
        let x_c_t_x = xc.t().dot(&x_u);
        let (s_new, sse_s, mu_s_new, sst_mat_new) =
            s_update(&s, &x_c_t_x, &ct_x_t_xc, mu_s, sst, sse, 10);
        s        = s_new;
        sse      = sse_s;
        mu_s     = mu_s_new;
        sst_mat  = sst_mat_new;

        d_sse    = sse_old - sse;
        var_expl = (sst - sse) / sst;
    }

    // ---- order components by usage -----------------------------------------
    let usage = s.sum_axis(Axis(1));
    let mut order: Vec<usize> = (0..noc).collect();
    order.sort_by(|&a, &b| usage[b].partial_cmp(&usage[a]).unwrap());
    c  = reorder_columns(&c, &order);
    s  = reorder_rows(&s, &order);
    xc = reorder_columns(&xc, &order);

    Ok(PchaResult {
        xc, s, c, sse, var_expl
    })
    // let mut t_ratio = None;
    // if calc_t_ratio {
    //     if noc > 1 {
    //         t_ratio = t_ratio_func(&xc, &x_u, noc);
    //     }
    // }

    // Ok(PchaResult {
    //     xc, s, c, sse, var_expl, t_ratio,
    // })
}
// -----------------------------------------------------------------------------
// helpers
// -----------------------------------------------------------------------------
fn select_columns(x: &Array2<f64>, cols: &[usize]) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((x.nrows(), cols.len()));
    for (k, &c) in cols.iter().enumerate() { out.column_mut(k).assign(&x.column(c)); }
    out
}
fn reorder_columns(m: &Array2<f64>, order: &[usize]) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((m.nrows(), order.len()));
    for (k, &idx) in order.iter().enumerate() { out.column_mut(k).assign(&m.column(idx)); }
    out
}
fn reorder_rows(m: &Array2<f64>, order: &[usize]) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((order.len(), m.ncols()));
    for (k, &idx) in order.iter().enumerate() { out.row_mut(k).assign(&m.row(idx)); }
    out
}
// fn normalise_columns(m: &mut Array2<f64>) {
//     for mut col in m.columns_mut() {
//         let s = col.sum();
//         if s > 0.0 { col.mapv_inplace(|v| v / s); }
//     }
// }

fn normalise_columns(m: &mut Array2<f64>) {
    // 1) collect all the mutable column views into a Vec
    let cols: Vec<_> = m.axis_iter_mut(Axis(1)).collect();
    // 2) run the Vec’s parallel iterator
    cols.into_par_iter().for_each(|mut col| {
        let sum = col.sum();
        if sum > 0.0 {
            col.mapv_inplace(|v| v / sum);
        }
    });
}



// -----------------------------------------------------------------------------
// S‑update (projected‑gradient on the simplex)
// -----------------------------------------------------------------------------
fn s_update(
    s: &Array2<f64>,
    xc_t_x: &Array2<f64>,
    ct_x_t_xc: &Array2<f64>,
    mut mu_s: f64,
    sst: f64,
    mut sse: f64,
    niter: usize,
) -> (Array2<f64>, f64, f64, Array2<f64>) {
    let (noc, j) = s.dim();
    let e = Array1::<f64>::ones(noc);
    let mut s_cur = s.to_owned();
    let mut sst_mat = s_cur.dot(&s_cur.t());

    for _ in 0..niter {
        let sse_old = sse;
        let grad = (ct_x_t_xc.dot(&s_cur) - xc_t_x).to_owned() / (sst / j as f64);
        // let grad = &grad - &e.clone().insert_axis(Axis(1))
        // .dot(&Array1::from_vec(vec![(&grad * &s_cur).sum()]));
        let col_sum = (&grad * &s_cur).sum_axis(Axis(0));           // length j
        let grad = &grad
            - &e.clone().insert_axis(Axis(1)) * &col_sum;                   // k × j
        
        let mut stop = false;
        let s_old = s_cur.clone();
        while !stop {
            s_cur = &s_old - &(grad.to_owned() * mu_s);
            project_simplex_columns(&mut s_cur);
            sst_mat = s_cur.dot(&s_cur.t());
            sse = sst - 2.0 * (xc_t_x * &s_cur).sum() + (ct_x_t_xc * &sst_mat).sum();
            if sse <= sse_old * (1.0 + 0.000000001) { mu_s *= 1.2; stop = true; } else { mu_s /= 2.0; }
        }
    }
    (s_cur, sse, mu_s, sst_mat)
}

fn project_simplex_columns(m: &mut Array2<f64>) {
    let cols: Vec<_> = m.axis_iter_mut(Axis(1)).collect();
    cols.into_par_iter().for_each(|mut col| {
        let mut v = col.to_owned();
        project_simplex(&mut v);
        col.assign(&v);
    });
}

fn project_simplex(v: &mut Array1<f64>) {
    let mut u: Vec<f64> = v.to_vec();
    u.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let mut cssv = 0.0;
    let mut rho  = 0;
    for (j, &u_j) in u.iter().enumerate() {
        cssv += u_j;
        let t = (cssv - 1.0) / (j as f64 + 1.0);
        if u_j - t > 0.0 { rho = j + 1; }
    }
    let theta = (u[..rho].iter().sum::<f64>() - 1.0) / rho as f64;
    v.mapv_inplace(|x| (x - theta).max(0.0));
}

// -----------------------------------------------------------------------------
// C‑update (projected‑gradient, optional ℓ₁ relaxation)
// -----------------------------------------------------------------------------
#[allow(clippy::too_many_arguments)]
fn c_update(
    x_i: &Array2<f64>,
    x_s_t: &Array2<f64>,
    xc: &Array2<f64>,
    sst_mat: &Array2<f64>,
    c: &Array2<f64>,
    delta: f64,
    mut mu_c: f64,
    mut mu_alpha: f64,
    sst: f64,
    mut sse: f64,
    niter: usize,
) -> (Array2<f64>, f64, f64, f64, Array2<f64>, Array2<f64>) {
    let (j, noc) = c.dim();
    let mut c_cur = c.to_owned();

    let mut alpha_c = if delta == 0.0 { Array1::<f64>::ones(noc) } else { c_cur.sum_axis(Axis(0)) };
    let x_t_x_s_t = x_i.t().dot(x_s_t);

    for _ in 0..niter {
        let sse_old = sse;
        let mut grad = (x_i.t().dot(&(xc.dot(sst_mat))) - &x_t_x_s_t).to_owned() / sst;
        if delta != 0.0 {
            for k in 0..noc { grad.column_mut(k).mapv_inplace(|v| v * alpha_c[k]); }
        }
        let ones_j = Array1::<f64>::ones(j);
        // let grad = &grad - &ones_j.insert_axis(Axis(1)).dot(&Array1::from_vec(vec![(&grad * &c_cur).sum()]));
        let col_sum = (&grad * &c_cur).sum_axis(Axis(0));           // length k
        let grad = &grad - &ones_j.insert_axis(Axis(1)) * &col_sum;              // j × k
        let mut stop = false;
        let c_old = c_cur.clone();
        while !stop {
            c_cur = &c_old - &(grad.to_owned() * mu_c);
            c_cur.mapv_inplace(|v| v.max(0.0));
            for mut col in c_cur.columns_mut() { let s = col.sum().max(f64::EPSILON); col.mapv_inplace(|v| v / s); }
            if delta != 0.0 {
                for k in 0..noc { c_cur.column_mut(k).mapv_inplace(|v| v * alpha_c[k]); }
            }
            let xc_new = x_i.dot(&c_cur);
            let ct_x_t_xc_new = xc_new.t().dot(&xc_new);
            sse = sst - 2.0 * (&xc_new * x_s_t).sum() + (&ct_x_t_xc_new * sst_mat).sum();
            if sse <= sse_old * (1.0 + 0.000000001) { mu_c *= 1.2; stop = true; } else { mu_c /= 2.0; }
        }

        if delta != 0.0 {
            let sse_old = sse;
            let ct_x_t_xc = xc.t().dot(xc);
            let mut grad = Array1::<f64>::zeros(noc);
            for k in 0..noc {
                let term1 = ct_x_t_xc.row(k).dot(&sst_mat.column(k)) / alpha_c[k];
                let term2 = c_cur.column(k).dot(&x_t_x_s_t.column(k));
                grad[k] = (term1 - term2) / (sst * j as f64);
            }
            let alpha_old = alpha_c.clone();
            let mut stop = false;
            while !stop {
                alpha_c = &alpha_old - &(grad.to_owned() * mu_alpha);
                for a in alpha_c.iter_mut() { *a = a.clamp(1.0 - delta, 1.0 + delta); }
                let scale = &alpha_c / &alpha_old;
                let mut xc_new = xc.clone();
                for k in 0..noc { xc_new.column_mut(k).mapv_inplace(|v| v * scale[k]); }
                let ct_x_t_xc_new = xc_new.t().dot(&xc_new);
                sse = sst - 2.0 * (&xc_new * x_s_t).sum() + (&ct_x_t_xc_new * sst_mat).sum();
                if sse <= sse_old * (1.0 + 0.000000001) { mu_alpha *= 1.2; stop = true; } else { mu_alpha /= 2.0; }
            }
        }
    }

    if delta != 0.0 {
        for k in 0..noc { c_cur.column_mut(k).mapv_inplace(|v| v * alpha_c[k]); }
    }
    let xc_final = x_i.dot(&c_cur);
    let ct_x_t_xc_final = xc_final.t().dot(&xc_final);

    (c_cur, sse, mu_c, mu_alpha, ct_x_t_xc_final, xc_final)
}

// -----------------------------------------------------------------------------
// Furthest‑Sum initialisation --------------------------------------------------
// -----------------------------------------------------------------------------
fn furthest_sum(k: &Array2<f64>, noc: usize, seed: usize, exclude: &[usize]) -> Vec<usize> {
    let j = k.ncols();
    let mut index = vec![true; j];
    for &e in exclude { if e < j { index[e] = false; } }
    let mut arche: Vec<usize> = vec![seed.min(j - 1)];
    index[arche[0]] = false;

    let mut sum_dist = vec![0.0; j];
    let norms: Vec<f64> = k.columns().into_iter().map(|col| col.dot(&col)).collect();
    let mut ind_t = arche[0];

    while arche.len() < noc {
        for t in 0..j {
            if index[t] {
                let dot_kt = k.column(ind_t).dot(&k.column(t));
                let dist = (norms[ind_t] + norms[t] - 2.0 * dot_kt).sqrt();
                sum_dist[t] += dist;
            }
        }
        let (best, _) = sum_dist.iter().enumerate()
            .filter(|(idx, _)| index[*idx])
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        arche.push(best);
        index[best] = false;
        ind_t = best;
    }
    arche
}



// /// Return the volume of the convex hull of `points` (p × n matrix).
// /// `None` → hull is degenerate or has < p + 1 points.
// pub fn _hull_volume(points: &Array2<f64>) -> Option<f64> {
//     let (p, n) = points.dim();
//     if p < 2 || n < p + 1 {
//         return None;
//     }

//     // --- 1. Flat buffer for qhull (column-major like ndarray) ------------
//     let mut buf: Vec<f64> = Vec::with_capacity(p * n);
//     for col in 0..n {
//         buf.extend(points.slice(s![.., col]).iter());
//     }

//     // --- 2. Build the hull ------------------------------------------------
//     let hull = match QhBuilder::default().build(p, &mut buf) {
//         Ok(h) => h,
//         Err(e) => panic!("qhull failed: {e:?}"),
//     };

//     // --- 3. Centroid as interior reference point -------------------------
//     let centroid: Array1<f64> = points.mean_axis(Axis(1)).unwrap();

//     // --- 4. Sum simplex volumes over facets ------------------------------
//     let fact = (1..=p).product::<usize>() as f64;           // p!
//     let mut vol = 0.0_f64;
//     for facet in hull.facets() {
//         // 1. unwrap the Option and turn the Set into a Vec
//     let verts_idx: Vec<Vertex> = facet
//         .vertices()          // -> Option<Set<'_, Vertex<'_>>>
//         .unwrap()            // skip non-simplicial
//         .iter()              // &Set   → iterator over &Vertex
//         // .cloned()            // &Vertex → Vertex (copy the handle)
//         .collect();          // iterator → Vec<Vertex>
//         if verts_idx.len() != p { continue; }          // need exactly p vertices

//         // 2. use the collected Vec for indexing
//         let mut m = Array2::<f64>::zeros((p, p));
//         for (k, vtx) in verts_idx.iter().enumerate() {
//             // vtx.point() -> &[f64]
//             let vk = Array1::from(vtx.point()?.to_vec());
//             m.column_mut(k).assign(&(&vk - &centroid));
//         }
//         let det = m.det().unwrap();
//         vol += det.abs() / fact;
//     }

//     (vol > 0.0).then_some(vol)
// }

// pub fn _t_ratio_func(xc: &Array2<f64>, x: &Array2<f64>, k: usize) -> Option<f64> {
//     // 1) project to the first k-1 dimensions (= rows 0 .. k-2)
//     let dim = k - 1;                    // same as R’s data_dim
//     let data_slice      = x.slice(s![0..dim, ..]).to_owned();
//     let arche_slice     = xc.slice(s![0..dim, ..]).to_owned();

//     // 2) hull volume of the data
//     let hull_v = _hull_volume(&data_slice)?;

//     // 3) hull volume of archetypes + a few jittered samples
//     let mut rng = rand::thread_rng();
//     let jitter  = Array2::<f64>::random_using(
//                      (dim, 20), Uniform::new(-1e-6, 1e-6), &mut rng);
//     let arc_pts = ndarray::concatenate![Axis(1), arche_slice, jitter];
//     let arc_v   = _hull_volume(&arc_pts)?;

//     Some(arc_v / hull_v)
// }
// -----------------------------------------------------------------------------
// end of file
