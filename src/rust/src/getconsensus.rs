// use std::process::Output;
#![allow(unused_assignments)]

#[derive(Debug)]
struct NonNsubstring {
    start: usize,
    end: usize,
    seq: String,
}

impl NonNsubstring {
    fn new(start: usize) -> Self {
        Self {
            start,
            end: 0,
            seq: String::new(),
        }
    }

    fn close(&mut self, end: usize, seq: &mut String) -> String {
        self.end = end;
        self.seq = seq.clone();
        format!("{}_{}_{}", self.start, self.end - 1, self.seq)
    }
}

pub fn getconsensus(rstring: String, index_add: usize) -> Vec<String> {
    let mut buff = String::new();
    let mut n: Option<NonNsubstring> = None;
    let mut output: Vec<String> = Vec::new();

    for (i, ch) in rstring.chars().enumerate() {
        match ch {
            'A' | 'C' | 'G' | 'T' => {
                if n.is_none() {
                    n = Some(NonNsubstring::new(i + index_add));
                }
                buff.push(ch);
                if i == rstring.len() - 1 {
                    if let Some(mut non_n) = n.take() {
                        output.push(non_n.close(i + index_add + 1, &mut buff));
                    }
                }
            },
            _ => {
                if let Some(mut non_n) = n.take() {
                    output.push(non_n.close(i + index_add, &mut buff));
                    buff.clear();
                }
            },
        }
    }

    output
}