<hr>
<p align="left"><img src="man/figures/rustytools.png" alt="" width="300"></a></p>
<hr>

[![Project Status: Active – The project has reached a stable, usable state and is being activelydeveloped.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Lifecycle:stable](https://img.shields.io/badge/lifecycle-stable-brightgreen.svg)](https://lifecycle.r-lib.org/articles/stages.html)


# Welcome to `rustytools`

Version 0.0.2

`rustytools` is a high-performance bioinformatics toolkit that integrates Rust-based computational kernels into R. It provides fast and memory-efficient implementations of common algorithms used in genomics and single-cell analysis, with a focus on speed, safety, and interoperability.

## Key Features

- 🔍 **Fuzzy motif detection**  
  Identify imperfect tandem repeats in DNA sequences using Rust-backed pattern matching.

- 🧬 **FASTA utilities**  
  Efficient sequence retrieval and scanning from large reference genomes.

- 🧠 **MAGIC diffusion**  
  Fast and scalable implementation of the MAGIC algorithm for imputing single-cell RNA-seq data  
  <sup>[van Dijk et al., *Cell*, 2018](https://www.cell.com/cell/fulltext/S0092-8674(18)30724-4)</sup>

- 🧱 **PCHA archetypal analysis**  
  Project data onto convex combinations of extreme states using Principal Convex Hull Analysis  
  <sup>[Groves et al., *Cell Systems*, 2022](https://www.cell.com/cell-systems/fulltext/S2405-4712(22)00313-1)</sup>

- ⚙️ **R-friendly with CLI speed**  
  All core computations are written in Rust, with clean R interfaces for seamless integration.

## Installation

Install the development version from GitHub:

```r
# install.packages("remotes")
remotes::install_github("furlan-lab/rustytools")
````

## Example: Motif Finding

```r
library(rustytools)

seq <- charToRaw("ATGATGCTGATGATG")
res <- find_runs(seq, motif = charToRaw("ATG"), min_repeats = 3, max_mismatches = 1)
print(res)
```

## Vignettes

* 📘 [Getting Started](articles/introduction.html)
* 🔁 [Pairwise Alignment](articles/seqAlign.html)
* 🧬 [FASTA Files](articles/getConsensus.html)
* 🧠 [MAGIC Imputation](articles/magic.html)
* 🧱 [PCHA for Archetype Analysis](articles/pcha.html)

## Reference

See the [function reference](reference/index.html) for full documentation.

Developed by the [Furlan Lab](https://furlan-lab.github.io/) at Fred Hutchinson Cancer Center
MIT Licensed

<hr>
<p align="center"><img src="man/figures/furlan_lab_logo.png" alt="" width="300"></a></p>
</hr>

---
