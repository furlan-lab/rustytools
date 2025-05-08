#' Run MAGIC diffusion on a Seurat object (Rust backend)
#'
#' This function wraps the Rust-based MAGIC diffusion kernel (`diffuse_expr_r`)
#' and seamlessly integrates it into a Seurat workflow. It will extract a sparse
#' expression matrix from your chosen assay/slot, build or reuse a k-NN graph,
#' row-normalize it to form a transition matrix, run diffusion for \code{t} steps,
#' and store the imputed values back into your Seurat object.
#'
#' @param obj A \code{\link[Seurat]{Seurat}} object.
#' @param assay Character(1). Which assay to use (default \code{"RNA"}).
#' @param slot Character(1). Which data slot within the assay:
#'   \code{"counts"}, \code{"data"}, or \code{"scale.data"} (default \code{"data"}).
#' @param t Integer(1). Diffusion time (number of steps, default \code{3}).
#' @param alpha Numeric(1) interpolation weight in [0,1]
#' @param k Integer(1). Number of neighbors to use if no graph exists (default \code{30}).
#' @param chunk Integer(1). Number of gene-columns per block in the Rust kernel
#'   (trades memory vs speed; default \code{2048}).
#' @param out_slot Character(1). Where to store the result:
#'   \code{"magic"} (adds to \code{@misc$magic} in the same assay) or
#'   \code{"new_assay"} (creates a new assay named \code{"MAGIC"}). Default is \code{"magic"}.
#' @param verbose Logical(1). Whether to print progress messages (default \code{TRUE}).
#'
#' @return The input Seurat object, with a new imputed matrix stored according to \code{out_slot}.
#'
#' @examples
#' \dontrun{
#' library(Seurat)
#' so <- CreateSeuratObject(counts = my_counts)
#' so <- NormalizeData(so)
#' so <- seurat_magic(so, assay = "RNA", slot = "data", t = 3, k = 30)
#' }
#'
#' @importFrom Matrix rowSums
#' @importFrom Seurat GetAssayData FindNeighbors CreateAssayObject DefaultAssay
#' @export
seurat_magic <- function(obj,
                         assay    = "RNA",
                         slot     = "data",
                         t        = 3L,
                         alpha    = 1L,
                         k        = 30L,
                         chunk    = 2048L,
                         out_slot = c("new_assay", "magic"),
                         verbose  = TRUE) {

  stopifnot(inherits(obj, "Seurat"))
  out_slot <- match.arg(out_slot)
  start_time <- Sys.time()

  ## 1) Extract sparse expression matrix (genes × cells)
  Seurat::DefaultAssay(obj) <- assay
  X <- Seurat::GetAssayData(obj[[assay]], slot = slot)
  if (!inherits(X, "dgCMatrix")) {
    stop("Slot '", slot, "' is not a dgCMatrix; please convert to sparse Matrix first")
  }

  ## 2) Build or reuse k-NN graph → row-stochastic transition matrix P
  graph_name <- paste0(assay, "_nn")
  if (!graph_name %in% names(obj@graphs)) {
    if (verbose) message("Finding ", k, "-nearest neighbors…")
    obj <- Seurat::FindNeighbors(obj,
                                 assay    = assay,
                                 features = NULL,
                                 k.param  = k,
                                 verbose  = verbose)
  }
  P <- obj@graphs[[graph_name]]
  # P <- P / Matrix::rowSums(P)  # ensure each row sums to 1

  row_sums <- Matrix::rowSums(P)
  P <- Matrix::Diagonal(x = 1 / pmax(row_sums, 1e-12)) %*% P

  ## 3) Call Rust diffusion (cells × genes orientation)
  rn <- rownames(X)
  cn <- colnames(X)
  pre_density <- length(X@x) / prod(dim(X))
  if (verbose) message("Running Rust MAGIC diffusion (t = ", t, ", alpha = ", alpha, ")…")
  comps <- diffuse_expr_r(P, Matrix::t(X), t = as.integer(t), chunk = as.integer(chunk))
  if (alpha < 1) {
    X_imp <- Matrix::t(dgC_from_components(comps))  # back to genes × cells
    X_imp <- (1 - alpha) * X + alpha * X_imp
    rm(X, comps)
    gc()
  } else {
    rm(X)
    gc()
    X_imp <- Matrix::t(dgC_from_components(comps))  # back to genes × cells
    rm(comps)
    gc()
  }
  rownames(X_imp) <- rn
  colnames(X_imp) <- cn


  ## 4) Store result
  if (out_slot == "magic") {
    obj[[assay]]@misc$magic <- X_imp
  } else {
    obj[["MAGIC"]] <- Seurat::CreateAssayObject(counts = X_imp)
  }

  end_time   <- Sys.time()
  elapsed    <- as.numeric(difftime(end_time, start_time, units = "secs"))
  if (verbose) {
    density <- length(X_imp@x) / prod(dim(X_imp))
    message("MAGIC complete:\n\ttotal runtime: ", signif(elapsed, 3), " seconds", "\n\tdensity before: ", signif(pre_density, 3), "\n\tdensity after: ", signif(density, 3))
  }
  obj
}


#' Build dgCMatrix from MAGIC Rust components
#'
#' @param comps list with elements `p`, `i`, `x`, `Dim`
#' @return a \link[Matrix]{dgCMatrix}
#' @export
dgC_from_components <- function(comps) {
  if (!requireNamespace("Matrix", quietly = TRUE))
    stop("Need the Matrix package")

  p   <- comps$p
  i   <- comps$i
  x   <- comps$x
  dim <- comps$Dim

  ## indptr length always = ncols + 1 in a CSC
  if (length(p) != dim[2] + 1) {
    if (length(p) == dim[1] + 1) {
      # swap the dimensions
      dim <- rev(dim)
    } else {
      stop("length(p) does not equal Dim[2]+1 or Dim[1]+1 – inconsistent list")
    }
  }

  methods::new("dgCMatrix",
               p        = as.integer(p),
               i        = as.integer(i),
               x        = as.numeric(x),
               Dim      = as.integer(dim),
               Dimnames = list(NULL, NULL)
  )
}
