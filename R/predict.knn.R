predict.knn <- function(object, test, train, theyre.the.same = FALSE, return.classifications = FALSE, 
	verbose = 0, ...)
{
#
# predict.knn: predict from an object with class "knn."
#
# Arguments:  object: object of class knn.
#                   test: Items to be predicted
#                  train: Things to use to do prediction
#        theyre.the.same: Are test and train the same? Then use leave-one-out cv
#  return.classifcations: If true, return the classifications of the test set
#                verbose: Level of verbosity
#
# First check to see whether this training set is pure. If so this is easy: if
# theyre the same, the error rate is zero; if not it's the proportion of test set
# items whose classes are different from the training set's unique class. In either
# class every item's classification is the training set one.
#
	if(all(object$which == FALSE) || (any(names(object) == "pure") && object$
		pure == TRUE) || length(unique(train[, 1])) == 1) {
		train.11 <- as.character(as.vector(train[1, 1]))
		if(theyre.the.same) {
			if(return.classifications)
				return(list(rate = 0, classifications = rep(
				  train.11, nrow(train))))
			else return(list(rate = 0))
		}
		else if(return.classifications) {
			return(list(rate = mean(test[, 1] != train[1, 1]), 
				classifications = rep(train.11, nrow(test))))
		}
		else return(list(rate = mean(test[, 1] != train[1, 1])))
	}
#
# Grab the true classes of the training set from column 1. If that column is a factor, 
# save the levels and then convert that column to numeric.
#
	if(is.factor(train[, 1])) {
		class.is.factor <- TRUE
		labels <- levels(train[, 1])
		train[, 1] <- as.integer(unclass(train[, 1])) - 1
	}
	else {
		class.is.factor <- FALSE
		labels <- unique(train[, 1])
	}
	train <- as.matrix(train)
	if(theyre.the.same && !missing(train)) {
		warning("They're the same, so I'm ignoring the 'test' argument"
			)
		test <- 0
	}
	else {
		if(is.factor(test[, 1])) {
			class.is.factor <- TRUE
			labels <- levels(test[, 1])
			test[, 1] <- as.integer(unclass(test[, 1])) - 1
		}
		else {
			class.is.factor <- FALSE
			labels <- unique(test[, 1])
		}
		test <- as.matrix(test)
	}
	number.of.classes <- length(labels)
	k.vec <- object$best.k
	k.length <- 1
	return.all.rates <- 0
	best.error.rate <- 1.1
	best.k.index <- -1	# Will come back zero-based, so add 1
	which <- c(TRUE, object$which)
	scaled <- object$scaled
	col.sds <- object$col.sds
	if(return.classifications == TRUE)
		classifications <- numeric(nrow(test))
	else classifications <- 0
	backward <- 0
	status <- 1	#
#
# Get "filename" (which will only be used if verbose > 0)
#
pr <- Sys.getenv("HOME")
	pr.chars <- substring(pr, 1:nchar(pr), 1:nchar(pr))
	backslash <- pr.chars == "\\"
	pr.chars[backslash] <- "/"
	pr <- paste(pr.chars, collapse = "")
	filename <- paste(pr, "/status.txt", sep = "")	#
#
# Call the DLL, and save the result.
#
	thang <- .C("myknn",
		as.double(train),
		as.integer(c(nrow(train), ncol(train))),
		as.double(test),
		as.integer(c(nrow(test), ncol(test))),
		as.integer(number.of.classes),
		as.integer(k.vec),
		as.integer(k.length),
		as.integer(theyre.the.same),
		rate = as.double(best.error.rate),
		as.integer(return.all.rates),
		best.k.index = as.integer(best.k.index),
		which = as.double(which),
		scaled = as.integer(scaled),
		col.sds = as.double(col.sds),
		as.integer(return.classifications),
		classifications = as.integer(classifications),
		backward = as.integer(backward),
		as.integer(verbose),
		filename,
		status = as.integer(status))	#
# Produce the output list, which contains the rate plus, if return.classifications
# is true, a vector of predictions. Factorize if necessary.
#
	status <- thang$status
	if(status != 0)
		warning(paste("Uh-oh; bad status", status, "returned"))
	if(return.classifications) {
		if(class.is.factor)
			classes <- factor(thang$classifications, labels = 
				labels, levels = 0:(length(labels) - 1))
		else classes <- thang$classifications
		out <- list(rate = thang$rate, classifications = classes)
	}
	else out <- list(rate = thang$rate)	#
	return(out)
}
