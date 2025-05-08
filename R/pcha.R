#' Ordinary–least-squares line through a set of points
#'
#' Quick wrapper around \code{\link[stats]{lm}} that returns the slope
#' and intercept of the straight line \eqn{y = m x + b} fitted to
#' \code{(x, y)} via ordinary–least-squares.
#'
#' @param x Numeric vector of explanatory values.
#' @param y Numeric vector of response values (same length as \code{x}).
#'
#' @return A named list with components
#'   \item{m}{Numeric scalar, slope \eqn{m}.}
#'   \item{b}{Numeric scalar, intercept \eqn{b}.}
#'
#' @keywords internal
.linerr <- function(x, y) {
  ## slope and intercept of the OLS line through (x,y)
  fit <- lm(y ~ x)
  list(m = coef(fit)[2], b = coef(fit)[1])
}

#' Piece-wise (two-segment) 1-norm regression error
#'
#' Computes the sum of absolute residuals obtained by fitting two
#' independent OLS lines to the portions of the curve lying to the left
#' and right of a proposed breakpoint.
#'
#' @param x Numeric vector of \eqn{x}-coordinates.
#' @param y Numeric vector of \eqn{y}-coordinates (same length as \code{x}).
#' @param brk Integer index designating the breakpoint (1 ≤ \code{brk} ≤ \code{length(x)}).
#'
#' @details
#' The points \code{x[1:brk]} and \code{y[1:brk]} form the *left* segment;
#' the points \code{x[brk:length(x)]} and \code{y[brk:length(x)]} form the
#' *right* segment.  Each segment is fitted with an OLS line via
#' \code{.linerr()}, and the 1-norms (sum of absolute residuals) of the
#' two fits are added.
#'
#' @return Numeric scalar giving the total 1-norm error for that
#'   breakpoint.
#'
#' @keywords internal
seg_err <- function(x, y, brk) {
  ## 1-norm of residuals for two-segment fit with breakpoint 'brk'
  left  <- .linerr(x[1:brk],           y[1:brk])
  right <- .linerr(x[brk:length(x)],   y[brk:length(x)])
  eL <- abs(left$m  * x[1:brk]          + left$b  - y[1:brk])
  eR <- abs(right$m * x[brk:length(x)]  + right$b - y[brk:length(x)])
  sum(eL) + sum(eR)
}


#' Locate the “elbow” of a monotone curve
#'
#' Implements a two–segment piece-wise linear method to identify the knee
#' (a.k.a. elbow) in a curve such as the reconstruction-error profile
#' obtained when varying the number of archetypes.
#'
#' @param y Numeric vector of \eqn{y}-values (must have ≥ 3 points).
#' @param x Numeric vector of \eqn{x}-values (defaults to
#'   \code{seq_along(y)}). Need not be sorted.
#' @param make_plot Logical; if \code{TRUE}, a \pkg{ggplot2} graphic is
#'   produced showing the curve and the detected knee.
#'
#' @return A list with components
#' \describe{
#'   \item{knee_x}{The \eqn{x}-coordinate of the knee.}
#'   \item{idx}{Integer index (1-based) of that knee in the supplied
#'     vectors.}
#'   \item{error_curve}{Numeric vector of summed absolute-error values
#'     for every tested breakpoint (length = \code{length(y) - 2}).}
#' }
#'
#' @examples
#' ks  <- 2:25
#' err <- exp(-ks) + 0.05 * rnorm(length(ks))
#' knee_pt(err, ks, make_plot = TRUE)
#'
#' @importFrom ggplot2 ggplot geom_line geom_point geom_vline labs theme_bw aes
#' @export
find_knee_pt <- function(y, x = seq_along(y), make_plot = FALSE) {
  stopifnot(length(y) >= 3L,
            length(x) == length(y))

  ord <- order(x)                      # sort by x just in case
  x <- x[ord]; y <- y[ord]

  errs <- vapply(2:(length(y) - 1), seg_err,
                 FUN.VALUE = numeric(1),
                 x = x, y = y)

  brk <- which.min(errs) + 1L          # +1 because we started at index 2

  if (make_plot) {
    g <- ggplot2::ggplot(data.frame(k = x, rel = y)) +
      ggplot2::geom_line(ggplot2::aes(k, rel)) +
      ggplot2::geom_point(ggplot2::aes(k, rel)) +
      ggplot2::geom_vline(xintercept = x[brk],
                          colour = "red", linetype = "dashed") +
      ggplot2::labs(x = "Number of archetypes (k)",
                    y = "Relative SSE",
                    title = "Elbow detected by knee_pt()") +
      ggplot2::theme_bw()
    print(g)
  }

  list(knee_x      = x[brk],
       idx         = brk,
       error_curve = errs)
}
