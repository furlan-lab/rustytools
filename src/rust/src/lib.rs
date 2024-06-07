use extendr_api::prelude::*;

mod getconsensus;
mod align;
use crate::align::Aligner;

/// @export
#[extendr]
fn getconsensus(rstring: Robj, index_add: Robj) -> Robj {
    // let rstring = "NNNNNNNNATGCGTANNNNNNNTTGACNNNNNNNN".to_string();
    let rstring = rstring.as_str().unwrap().to_string();
    let index_add = *index_add.as_real_vector().unwrap().first().unwrap() as usize;
    let output = getconsensus::getconsensus(rstring, index_add);
    output.into_iter().collect_robj()
}

/// @export
#[extendr]
fn align(rstring1: Robj, rstring2: Robj, atype: Robj, verbose: Robj) -> Robj {
    // eprint!("rstring1: {:?}, rstring2: {:?}", rstring1, rstring2);
    let atype = atype.as_str_vector().unwrap().first().unwrap().to_string();
    let x = rstring1.as_str_vector().unwrap().first().unwrap().to_string().into_bytes();
    let y = rstring2.as_str_vector().unwrap().first().unwrap().to_string().into_bytes();
    let score = |a: u8, b: u8| if a == b { 1i32 } else { -1i32 };
    let verbose = verbose.as_logical_vector().unwrap().first().unwrap().to_bool();
    // gap open score: -5, gap extension score: -1
    let mut aligner = Aligner::with_capacity(x.len(), y.len(), -5, -1, &score);
    if atype=="global" {
        let alignment = aligner.global(&x, &y);
        if verbose {
            println!("alignment: {:?}", alignment);
        }
        return alignment.score.into_robj();
    } else if atype=="local" {
        let alignment = aligner.local(&x, &y);
        if verbose {
            println!("alignment: {:?}", alignment);
        }
        return alignment.score.into_robj();
    } else {
        let alignment = aligner.semiglobal(&x, &y);
        if verbose {
            println!("alignment: {:?}", alignment);
        }
        return alignment.score.into_robj();
    }
}

/// @export
#[extendr]
fn pretty(rstring1: Robj, rstring2: Robj, atype: Robj, verbose: Robj) {
    // eprint!("rstring1: {:?}, rstring2: {:?}", rstring1, rstring2);
    let atype = atype.as_str_vector().unwrap().first().unwrap().to_string();
    let x = rstring1.as_str_vector().unwrap().first().unwrap().to_string().into_bytes();
    let y = rstring2.as_str_vector().unwrap().first().unwrap().to_string().into_bytes();
    let score = |a: u8, b: u8| if a == b { 1i32 } else { -1i32 };
    let verbose = verbose.as_logical_vector().unwrap().first().unwrap().to_bool();
    // gap open score: -5, gap extension score: -1
    let mut aligner = Aligner::with_capacity(x.len(), y.len(), -5, -1, &score);
    if atype=="global" {
        let alignment = aligner.global(&x, &y);
        if verbose {
            println!("alignment: {:?}", alignment.pretty(&x, &y, 10));
        }
    } else if atype=="local" {
        let alignment = aligner.local(&x, &y);
        if verbose {
            println!("alignment: {:?}", alignment.pretty(&x, &y, 10));
        }
    } else {
        let alignment = aligner.semiglobal(&x, &y);
        if verbose {
            println!("alignment: {:?}", alignment.pretty(&x, &y, 10));
        }
    }
}

// Macro to generate exports.
// This ensures exported functions are registered with R.
// See corresponding C code in `entrypoint.c`.
extendr_module! {
    mod rustytools;
    fn getconsensus;
    fn align;
}
