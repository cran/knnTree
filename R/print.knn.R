print.knn <- function(x, ...)
{
#
# print.knn: Print an object of class knn.
#
# 1.) Extract n, print a line.
#
	if(any(names(x) == "n")) n <- x$n else n <- "???"
	cat("This knn classifier is based on", n, "observations.\n")	#
#
# Print a line if the node is pure; print one telling how many variables were used;
# and one telling the training rate and best k.
#
	if(x$scaled == TRUE)
		scaled <- "with"
	else scaled <- "without"
	if(x$pure)
		cat("This node is pure!\n")
	cat("It uses", sum(x$which), "out of", length(x$which), "variables",
		scaled, "scaling.\n\n")
	cat("Training rate is", signif(x$rate, 4), ", achieved at k =", x$
		best.k, "\n")	#
#
# If this knn object is contained in a knnTree object, it has an element called
# "leaf," giving the leaf number and "where" value. Print that, too.
#
	if(any(names(x) == "leaf"))
		cat("This is leaf number", x$leaf, "with where value", x$
			where, "\n")
}
