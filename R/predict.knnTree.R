predict.knnTree <- function(object, test, train, verbose = FALSE, ...)
{
#
# predict.knnTree: get prediction from "test" on model in "object"
#
# Arguments: object: object of class "knnTree"
#                  test: Data on which to make predictions
#                 train: Data from which model was built (required)
#               verbose: Level of verbosity.
#
# Extract the tree (which is the first entry in "object") and deduce its size.
#
	my.tree <- object[[1]]
	size <- sum(my.tree$frame[, 1] == "<leaf>")	#
#
# If the tree has size one, call predict on the second object, re-classifiy the
# resulting classifications, and return them.
#
	if(size == 1) {
		thing <- object[[2]]
		class <- predict(thing, test, train, theyre.the.same = FALSE,
			return.classifications = TRUE)$class
		if(is.factor(train[, 1]))
			class <- factor(class, levels = levels(train[, 1]), 
				labels = levels(train[, 1]))
		return(class)
	}
#
#
# Create a vector of classifications. Then go through the leaves, calling
# predict on each one.
#
	if(is.factor(train[, 1]))
		class <- factor(rep("", nrow(test)), levels = levels(
			train[, 1]), labels = levels(train[, 1]))
	else class <- character(nrow(test))
	leaf.locations <- my.tree$frame[, 1] == "<leaf>"
	where <- (1:nrow(my.tree$frame))[leaf.locations]
	leaf.number <- dimnames(my.tree$frame)[[1]][leaf.locations]
	new.leaves <- predict(my.tree, test, type = "where")
	for(i in 1:length(where)) {
		new.ind <- new.leaves == where[i]
		if(sum(new.ind) == 0)
			next
		old.ind <- my.tree$where == where[i]
		thing <- object[[leaf.number[i]]]
		predict.out <- predict(thing, test[new.ind,  ], train[
			old.ind,  ], theyre.the.same = FALSE, 
			return.classifications = TRUE)
		class[new.ind] <- predict.out$classifications
		if(verbose)
			cat(i, ": Leaf", leaf.number[i], "(where =", where[i], 
				") has size", sum(new.ind), ", rate", signif(
				predict.out$rate, 4), "\n")	#
	}
	if(is.factor(train[, 1]))
		class <- factor(class, levels = levels(train[, 1]), labels
			 = levels(train[, 1]))
	return(class)
}
