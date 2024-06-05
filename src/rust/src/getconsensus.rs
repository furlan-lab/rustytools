// use std::process::Output;
#![allow(unused_assignments)]

// #[derive(Debug)]
// struct NonNsubstring {
//     start: Option<usize>,
//     end: Option<usize>,
//     seq: Option<String>,
// }


// fn start_string() -> Option<NonNsubstring> {
//     None
// }

// fn new_found(start: usize) -> Option<NonNsubstring> {
//     Some(NonNsubstring {
//         start: Some(start),
//         end: None,
//         seq: None,
//     })
// }

// fn close_found(n: Option<NonNsubstring>, end: usize, buff: & mut String) -> (String, Option<NonNsubstring>) {
//     let mut n = n.unwrap();
//     n.end = Some(end);
//     n.seq = Some(buff.to_string());
//     (format!("{}_{}_{}", n.start.unwrap(), n.end.unwrap() - 1, n.seq.as_ref().unwrap()), None)
// }

// pub fn getconsensus(rstring: String, index_add: usize) -> Vec<String> {
//     // This function will be called from R.
//     // It will take a string and print the coordinates of every repeated character substring made up of either "ACGT".
//     let mut buff = "".to_string();
//     // let mut n = NonNsubstring {
//     //     status: false,
//     //     start: None,
//     //     end: None,
//     //     seq: None,
//     // };
//     let mut n: Option<NonNsubstring> = start_string();
//     // eprintln!("String length {}", rstring.len().to_string());
//     let mut output: Vec<String> = Vec::new();
//     let mut ss = String::new();
//     for i in 0..rstring.len() {
//         let ch = rstring.chars().nth(i).unwrap();
//         if (ch == 'A' || ch == 'C' || ch == 'G' || ch == 'T') && n.is_none() {
//             // eprintln!("Found a new non-N substring at index {}", i);
//             // n = Some(NonNsubstring {
//             //     start: Some(i),
//             //     end: None,
//             //     seq: None,});
//             n = new_found(i+index_add);
//             buff.push(ch);
//             continue;
//         } else if (ch == 'A' || ch == 'C' || ch == 'G' || ch == 'T') && n.is_some() && i < rstring.len() - 1 {
//             // eprintln!("Continuing a new non-N substring at index {}", i);
//             buff.push(ch);
//             continue;
//         } else if n.is_some() && i == rstring.len() - 1 && (ch == 'A' || ch == 'C' || ch == 'G' || ch == 'T'){
//             // eprintln!("Found the end of the string closing with a substring at index {}", i);
//             (ss, n) = close_found(n, i + index_add, &mut buff);
//             output.push(ss);
//             buff.clear();
//             continue;
//         } else if (ch != 'A' || ch != 'C' || ch != 'G' || ch != 'T') && n.is_some() {
//             // eprintln!("Found a closing N at index {}", i);
//             // n.unwrap().end = Some(i);
//             // n.unwrap().seq = Some(buff.to_string());
//             // buff.clear();
//             (ss, n) = close_found(n, i + index_add, &mut buff);
//             output.push(ss);
//             buff.clear();
//             continue;
//         } else {
//             // eprintln!("Found an N at index {}", i);
//             continue;
//         }
//     }
//     output
// }

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