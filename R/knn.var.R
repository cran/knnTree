knn.var <- function (train, test, k.vec = seq(1, 
    31, by = 2), theyre.the.same = FALSE, return.all.rates = FALSE, scaling = 1, 
    backward = FALSE, save.call = FALSE, verbose = 0, use.big = TRUE) 
{
#
# knn.var: Call C code to produce a knn object with variable selection.
#
# Arguments: train: training set with classificationd in column 1
#             test: test set with classifications in column 1
#            k.vec: Set of values of k to consider
#  theyre.the.same: TRUE is training and test sets are the same
# return.all.rates: if TRUE, return all error rates (one for each element
#                   in k.vec), not just the smallest
#          scaling: 0 = don't scale; 1 = try SDs; 2 = try MADs.
#         backward: TRUE = backward selection; FALSE = forward
#        save.call: if TRUE, save a copy of the call
#          verbose: Level of verbosity (for debugging)
#          use.big: try a bigger-but-faster approach
#
# Start by sorting the vector of ks. It needs to be in order.
    k.vec <- sort(k.vec)
#
# If column 1 is a factor, turn it into integers starting at 0. If not,
# assume it's already in that form. Is that a mistake?
#
    if (is.factor(train[, 1])) {
        number.of.classes <- length(levels(train[, 1]))
        train[, 1] <- as.integer(unclass(train[, 1])) - 1
    }
    else number.of.classes <- max(train[, 1]) + 1
    train <- as.matrix(train)
#
# Likewise test. If it's missing, presumably test and train are the same.
#
    if (missing(test)) 
        test <- matrix(0, 1, 1)
    else {
        test[, 1] <- as.integer(unclass(test[, 1])) - 1
        test <- as.matrix(test)
    }
#
# If there's only one class represented, we're done. The error rate is 0
# on the training set and easy to compute on the test set. Construct a
# pure object and return.
#
    if (length(unique(train[, 1])) == 1) {
        if (theyre.the.same) {
            rate <- 0
        }
        else {
            rate <- mean(test[, 1] == train[1, 1])
        }
        out <- list(which = rep(FALSE, ncol(train) - 1), rate = rate, 
            best.k = k.vec[1], scaled = FALSE, pure = TRUE, n = nrow(train))
        class(out) <- "knn"
        return(out)
    }
#
# Otherwise set up some stuff for the call.
#
    k.length <- length(k.vec)
    if (return.all.rates) 
        best.error.rate <- rep(1.1, length(k.vec))
    else best.error.rate <- 1.1
    best.k.index <- -1
    which <- numeric(ncol(train))
    col.sds <- numeric(ncol(train))     # might be MADs, too.
    return.classifications <- 0 	# don't do that here
    classifications <- 0
    if (use.big) 
        status <- 2
    else status <- 1
#
# All this is for setting up the status file. The backslash stuff is
# for S-Plus.
#
    pr <- Sys.getenv("HOME")
    pr.chars <- substring(pr, 1:nchar(pr), 1:nchar(pr))
    backslash <- pr.chars == "\\"
    pr.chars[backslash] <- "/"
    pr <- paste(pr.chars, collapse = "")
    filename <- paste(pr, "/status.txt", sep = "")
    thang <- .C("knnvar", as.double(train), as.integer(c(nrow(train), 
        ncol(train))), as.double(test), as.integer(c(nrow(test), 
        ncol(train))), as.integer(number.of.classes), as.integer(k.vec), 
        as.integer(k.length), as.integer(theyre.the.same), rate = as.double(best.error.rate), 
        as.integer(return.all.rates), best.k.index = as.integer(best.k.index), 
        which = as.double(which), scaling = as.integer(scaling), 
        col.sds = as.double(col.sds), as.integer(return.classifications), 
        as.integer(classifications), as.integer(backward), as.integer(verbose), 
        filename, status = as.integer(status))
#
# Okay. We're back. First thing: is status = 0?
#
    status <- thang$status
    if (status != 0) 
        warning(paste("Uh-oh; bad status", status, "returned"))
    col.sds <- thang$col.sds
    names(col.sds) <- dimnames(train)[[2]]
    scaled <- as.logical(thang$scaling)
    out <- list(which = as.logical(thang$which)[-1], rate = thang$rate, 
        best.k = k.vec[thang$best.k.index + 1], scaled = as.logical(thang$scaling), 
        n = nrow(train))
    if (scaled) 
        out$col.sds <- thang$col.sds[-1]
    if (all(out$which == FALSE)) 
        out$pure <- TRUE
    else out$pure <- FALSE
    if (save.call) 
        out$call <- match.call()
    class(out) <- "knn"
    out
}
