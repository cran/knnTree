print.knnTree <- function(x, ...)
{
#
# print.knnTree: print function for knnTree objects.
#
# Grab "n". Then figure out how many non-leaf items there are. It's length(x) - 1
# if "call" isn't present, and length(x) - 2 if it is.
	n <- length(x[[1]]$y)
	if(any(names(x) == "call"))
		not.leaves <- c(1, length(x))
	else not.leaves <- 1
	leaves <- length(x) - length(not.leaves)	#
#
# Print the sample size and number of leaves. Then to each leaf apply a function that
# extracts n, best.k, rate and where. This forms a matrix. Set the row names.
#
	cat("Tree: size", n, "with", leaves, "leaves;")
	leaf.stuff <- sapply(x[ - not.leaves], function(x)
	c(x$n, x$best.k, x$rate, x$where))
	dimnames(leaf.stuff)[[1]] <- c("n", "k", "rate", "where")	#
#
# Compute overall error rate and print it.
#
	errors <- sum(round(leaf.stuff["rate",  ] * leaf.stuff["n",  ]))
	leaf.numbers <- names(x)[-1]
	cat("training error rate:", signif(errors/n, 4), "\n")	#
#
# Now print one line of leaf-specific info for each leaf.
#
	for(i in 1:ncol(leaf.stuff)) {
		cat(paste("Leaf", leaf.numbers[i], " (where =", leaf.stuff[
			"where", i], ") has n =", leaf.stuff["n", i], ", k =", 
			leaf.stuff["k", i], ", rate", signif(leaf.stuff["rate", 
			i], 4), "\n", sep = ""))
	}
}
