

# find_cuts<-function(len, n){
#   cut<-ceiling(len/n)
#   c(seq(1, len, by = cut), len)
# }
#
#
# split_string<-function(string, n){
#   cuts<-find_cuts(nchar(string), n)
#   string<-sapply(1:(length(cuts)-1), function(n){
#     substring(string, cuts[n], cuts[n+1])
#   })
#   return(list(cuts, string))
# }

#' Get Consensus Sequences from a FASTA File
#'
#' This function reads a DNA sequence from a given FASTA file, splits the sequences for parallel processing,
#' and retrieves consensus sequences for each split segment.
#'
#' @param fasta A character string specifying the path to the FASTA file.
#' @param splits An integer specifying the number of segments to split the sequence into for processing. Default is 10,000.
#' @param cores An integer specifying the number of cores to use for parallel processing. Default is 1.
#' @param genome A character string specifying the genome version for sequence information. Default is "hg38".
#'
#' @return A \code{GRanges} object containing the consensus sequences with their genomic ranges and associated sequence information.
#'
#' @details The function reads the input FASTA file, splits each sequence into smaller segments for parallel processing,
#'          and retrieves consensus sequences for each segment. The results are combined into a \code{GRanges} object.
#'
#' @examples
#' \dontrun{
#'   # Example usage:
#'   fasta_file <- "path/to/your/file.fasta"
#'   consensus <- get_consensus(fasta_file, splits = 5000, cores = 2, genome = "hg38")
#'   print(consensus)
#' }
#'
#' @importFrom seqinr read.fasta
#' @importFrom GenomeInfoDb Seqinfo
#' @importFrom pbmcapply pbmclapply
#' @importFrom GenomicRanges GRanges
#' @importFrom IRanges IRanges
#' @importFrom S4Vectors DataFrame
#' @references This documentation was written by ChatGPT v4o - OpenAI, conversation with the author, 6-5-2024.
#' @export

#get_consensus<-function(fasta, splits = 10000, cores=1, genome="hg38", test_with_n = NULL){
get_consensus<-function(fasta, cores=1, genome="hg38", test_with_n = NULL){
  if(!file.exists(fasta)) {stop("Cannot find fasta")}
  message(paste0("Reading fasta: ", fasta, "..."))
  fa<-seqinr::read.fasta(file = fasta,
                 seqtype = "DNA", as.string = TRUE, forceDNAtolower = FALSE,
                 set.attributes = TRUE)
  message("Fasta ingestion complete")
  message("Building sequence info")
  seqinfo<-GenomeInfoDb::Seqinfo(names(fa), seqlengths=sapply(1:length(fa), function(n) nchar(fa[[n]][1])), isCircular=NA, genome=genome)
  # x<-fa[[1]][1]

  if(is.null(test_with_n)){
    final_n = length(fa)
  } else {
    if(length(test_with_n)>1){stop("test_with_n must be an integer vector of length 1")}
    final_n = as.integer(test_with_n)
  }
  allres<-pbmclapply(1:final_n, function(i){
    #message(paste0("Processing ", nchar(fa[[i]][1]), " bases from ", names(fa)[i]))
    # xs<-split_string(fa[[i]][1], splits)
    # res<-pbmclapply(1:length(xs[[2]]), function(j){
    #   getconsensus( xs[[2]][j], xs[[1]][j])
    # }, mc.cores = cores)
    res<-getconsensus(fa[[i]][1], 1:nchar(fa[[i]][1]))
    res<-unlist(res)
    if(length(res)>0){
      df<-t(data.frame(strsplit(res, "_")))
      GRanges(seqnames = rep(names(fa)[i], nrow(df)),
                             ranges = IRanges(start = as.numeric(df[,1]), end = as.numeric(df[,2])),
                             mcols = DataFrame(sequence=df[,3]), seqinfo = seqinfo)
    }
  },mc.cores = cores)
  allres <- do.call(c, allres)
  GenomicRanges::GRangesList(allres)
}

#
#
#
#
#
#
#
#
#
#
#
#
#
# substring(x, 14447574, 14447592)
# substring(x, 2244091, 2244116)
# #
# # Start: 14447574; End: 14447584; Seq: TGCTTCTATCA
# # Start: 14447586; End: 14447592; Seq: GGTCTGT
#
# substring(x, 832501, 832522)
# # "832502_832521_AGACCATGAAGGTCCATGCT"
