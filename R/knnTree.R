knnTree <- 
function(trg.set, trg.classes, v = 10, k.vec = seq(1, 31, by = 2), seed = 0, 
	opt.tree = "ignore", opt.tree.size = 4, scaling = 1, prune.function = 
	prune.misclass, one.SE = TRUE, backward = FALSE, v.start = 1, leaf.start = 1, 
	verbose = FALSE, debug = 0, fname = "", use.big = FALSE, save.output = "")
{
#
# knnTree: Build knn classifiers in the leaves of a classification
#          tree with cross-validation
#
# Arguments: 	trg.set : full trg set of data (without classes)
#           trg.classes : vector of classes of the trg.set of data
# 		     v  : Number of pieces for cross-validation
#                k.vec  : Vector of k's to consider
#                 seed  : seed to set if doing CRN work
#               opt.tree: whether or not to calculate optimal tree. Values are:
#                         "ignore": use full tree from first iteration; "find": use
#                         pruned tree from 1st iter.; "fix": use opt.tree.size only;
#                         "max": consider only trees with sizes <= opt.tree.size
#          opt.tree.size: tree size used if opt.tree == "fix" or "max"
#                scaling: Scaling for knn. 0 = none; 1 = by SD; 2 = by MAD.
#         prune.function: Function to be used for pruning operations
#              backward : If TRUE, do backwards selection, else forward
#               v.start : Chunk to start at (for debugging only)
#            leaf.start : Leaf to start at (debugging only)
# 		verbose : whether optimal tree size and other information is to be
#			  dumped to screen	
#                 debug : Handy for debugging
#
#	   return value : Object of class "knnTree".
#				
# The overall Plan for this function:
#
# A. divide data into n pieces.
# B. For each piece "v" (v going from 1 to n)
#     1. Build the over-fit tree for data excluding piece p
#     2. Prune to some size (default: "optimal"); call that "best.size"
#     3. For each tree size (tr going from 1 to best.size)
# 	   a. For each leaf, find optimal k for that leaf using leave one out cv
# C. Compute number of misclasses when data piece p is classified by this scheme
# D. Choose tree size with smallest total number of misclasses
# E. Return that tree plus its leaf info.
#
# If trg.classes isn't a factor, forget it. Otherwise store some stuff in frame 1.
#
	if(!is.factor(trg.classes)) stop("Response is not a factor")
	assign("verbose", verbose, frame = 1)
	assign("fname", fname, frame = 1)	#
#
# "Diagnose" sends an error message out to a file, if the "verbose" level
# is high enough.
#
	diagnose <- function(..., level = 0)
	{
		if(verbose > level)
			cat(..., file = fname, append = TRUE)
	}
#
#
# Create a big ol' data set, and store that in frame 1.
#
	data <- data.frame(class = trg.classes, trg.set)
	assign("data", data, frame = 1)	#
# Build the big unpruned tree.
#
	overall.big.tree <- tree(class ~ ., data = data)
	diagnose("Overall tree, unpruned, has size", sum(overall.big.tree$frame[
		, 1] == "<leaf>"), "\n", level = 1)
	nrow.data <- nrow(trg.set)
	if(!missing(seed)) set.seed(seed)	#
# A: Divide data into v pieces. Here we sample the data and compute starting
# points for each of the chunks. So chunk 1 consists of the elements of "samp"
# from 1 to (chunk.start[2]-1); chunk 2 is the elements from chunk.start[2] to
# (chunk.start[3]-1), and so on.
#
	samp <- sample(nrow.data)
	chunk.start <- round(seq(1, nrow.data, len = v + 1))[ - (v + 1)]	#
#
# If opt.tree is "fix," set the optimal tree size now.
#
	if(opt.tree == "fix") {
		optimal.size <- opt.tree.size
		diagnose("Tree size fixed at", opt.tree.size, "!\n", level = 0)
	}
	else {
#
# Otherwise, if the optimal tree is already size 1, we'll use that, and it not, 
# call cv.tree to cross-validate. Use the one.SE rule if requested, but not with
# anything other than prune.misclass().
#
		if(nrow(overall.big.tree$frame) == 1) overall.best.size <- 1
			 else {
			overall.cv <- cv.tree(overall.big.tree, FUN = 
				prune.function)
			if(!all.equal(prune.function, prune.misclass))
				if(one.SE) {
				  warning(
				    "Can't do one SE without prune.misclass")
				  one.SE <- FALSE
				}
			if(one.SE) {
#
# Here's the one SE rule in action. Find the smallest rate r; then compute r-star =
# sqrt(r (1-r)/n), and find the smallest size whose rate <= r-star.
#
				overall.rates <- overall.cv$dev/nrow.data
				min.rate <- min(overall.rates)
				one.se.rate <- min.rate + sqrt((min.rate * (1 - 
				  min.rate))/length(overall.big.tree$y))
				overall.best.size <- min(overall.cv$size[
				  overall.rates <= one.se.rate])
			}
			else {
#
# If one.SE isn't supplied, pick the size with the smallest rate.
#
				overall.best.size <- min(overall.cv$size[
				  overall.cv$dev == min(overall.cv$dev)])
			}
		}
		if(opt.tree == "max") {
			if(overall.best.size > opt.tree.size)
				diagnose("Using specified max", opt.tree.size, 
				  "\n", level = 0)
			overall.best.size <- opt.tree.size
		}
		results <- numeric(overall.best.size)	#
#
# The big loop, over the chunks. V.start = 1 except when debugging.
#
		for(i in v.start:v) {
			if(i == v)
				chunk <- samp[(chunk.start[i]):nrow.data]
			else chunk <- samp[(chunk.start[i]):(chunk.start[i + 1] -
				  1)]
			assign("chunk", chunk, frame = 1)	#
			diagnose("Top of loop for i =", i, "\n", level = 0)	#
#
# B.1: build the oversize tree excluding this subset
#
			big.tree <- tree(class ~ ., data = data, subset =  - 
				chunk)	#
#
# B.2: prune to some size. If opt.tree is "find," use the best.size we've already
# constructed. If it's "ignore," we would use the unpruned size, I guess -- but let's
# not do this for now. So all we do is prune to best.size.
#
			if(i == v.start)
				best.size <- overall.best.size
			if(nrow(big.tree$frame) == 1)
				best.tree <- big.tree
			else best.tree <- prune.function(big.tree, best = 
				  best.size)
			diagnose("Found best tree; its size is", best.size, 
				"\n", level = 0)	#
#
# B.3: loop over tree sizes.
#
			if(opt.tree == "fix")
				smallest <- best.size
			else smallest <- 1
			for(tr in smallest:best.size) {
				diagnose("...Top of loop for tree of size", tr, 
				  "\n", level = 0)	#
#
# Handle the annoying "singlenode" case. Also get the node numbers, which are
# the row names of the "frame" component of the tree.
#
				if(class(best.tree) == "singlenode")
				  current.tree <- best.tree
				else current.tree <- prune.function(best.tree, 
				    best = tr)
				node.names <- dimnames(current.tree$frame)[[1]]
				leaf.numbers <- node.names[current.tree$frame[, 
				  1] == "<leaf>"]	#
#
# At this stage we may as well figure out which "chunk" items fall in which leaf.
# We need special handling for the one-leaf tree.
#
				if(tr == 1)
				  where <- rep(1, length(chunk))
				else {
				  where <- predict(current.tree, data[chunk,  ],
				    type = "tree")$where
				  where <- node.names[where]
				}
				list.of.outs <- vector("list", length(
				  leaf.numbers))
				list.of.item.vectors <- vector("list", length(
				  leaf.numbers))
				names(list.of.item.vectors) <- leaf.numbers
				names(list.of.outs) <- leaf.numbers	#
#
# Loop over leaves. Leaf.start = 1 except when debugging.
#
				for(leaf in leaf.start:length(leaf.numbers)) {
				  this.leaf <- leaf.numbers[leaf]
				  items.in.leaf <- node.names[current.tree$
				    where] == this.leaf
				  diagnose("...   ...Leaf", this.leaf, "has", 
				    sum(items.in.leaf), "entries;", level = 1)	
	#
#
# Here we see whether we computed this leaf last time. If so we can just pull
# this leaf's results out of the last.list.of.outs.
#
				  if(tr != smallest && any(names(
				    last.list.of.item.vectors) == this.leaf) && 
				    all(items.in.leaf == 
				    last.list.of.item.vectors[[this.leaf]])) {
				    out <- last.list.of.outs[[this.leaf]]
				    diagnose("...(already)...found", out$
				      test.error, "misses\n", level = 1)
				    results[tr] <- results[tr] + out$test.error
				  }
				  else {
				    if(verbose > 1) diagnose("...(new)...", 
				        level = 1)	#
#
# Otherwise, figure out what subset of the data to use, and only use those k values
# which are smaller than the number of items in the leaf. If none of the k-vec entries
# satisfy that condition, set k = 1. Then call knn.var.select and save the output.
#
				    sub.data <- data[ - chunk,  ][items.in.leaf,
				      ]
				    new.k.vec <- k.vec[k.vec < sum(
				      items.in.leaf)]	#
				    if(length(new.k.vec) < 1)
				      new.k.vec <- 1
				    out <- knn.var.select(sub.data, k = 
				      new.k.vec, scaling = scaling, backward = 
				      backward, verbose = 0, theyre.the.same = 
				      TRUE, use.big = use.big)	#
#
# Find the test set items that fall in this leaf and, if there are any, call the
# predict function to get the number of misclassifications.
#
				    sub.data.test <- data[chunk,  ][where == 
				      this.leaf,  , drop = FALSE]
				    if(nrow(sub.data.test) == 0) {
				      diagnose(" found no entries\n", level = 1
				        )
				      out$test.error <- 0
				    }
				    else {
				      misclass.rate <- predict(out, 
				        sub.data.test, sub.data)$rate
				      misclass.count <- round(misclass.rate * 
				        nrow(sub.data.test))	#
#
# Save the number of errors in case we need it next time round.
#
				      out$test.errors <- misclass.count
				      diagnose(" found", misclass.count, 
				        "misses\n", level = 1)
				      results[tr] <- results[tr] + 
				        misclass.count
				    }
				  }
				  list.of.outs[[this.leaf]] <- out
				  list.of.item.vectors[[this.leaf]] <- 
				    items.in.leaf	#
				}
#
#
# end loop over leaves. Save the whole list of outs and item vectors for the next loop.
#
				last.tree <- current.tree
				last.list.of.outs <- list.of.outs
				last.list.of.item.vectors <- 
				  list.of.item.vectors
			}
# end loop over all tree sizes
			diagnose("...End chunk loop: results now", results, 
				"\n", level = 0)
		}
#
#
# D: find optimal sized tree. If we're using the one SE rule, the "n" is the size of the
# training set x the number of cross-vals.
#
		if(one.SE) {
			cv.n <- nrow.data * length(v.start:v)
			overall.rates <- results/cv.n
			best.rate <- min(overall.rates)
			one.se.rate <- best.rate + sqrt((best.rate * (1 - 
				best.rate))/cv.n)
			optimal.size <- min((smallest:best.size)[overall.rates <= 
				one.se.rate])
		}
		else {
			optimal.size <- (smallest:best.size)[results == min(
				results)][1]
		}
		diagnose("optimal.size is", optimal.size, "\n", level = 1)	#
	}
	if(1 > 0) {
#
# This "if" is here to anchor the comments. Sometimes, by the way, you ask to prune
# to some size but the resulting tree is one or two nodes bigger. Maybe this doesn't
# happen with prune.misclass(), but it did with prune.tree().
#
		if(nrow(overall.big.tree$frame) == 1) {
			overall.best.tree <- overall.big.tree
			optimal.size <- 1
		}
		else {
			overall.best.tree <- prune.function(overall.big.tree, 
				best = optimal.size)	#
			optimal.size <- sum(overall.best.tree$frame[, 1] == 
				"<leaf>")
		}
	}
	else {
		diagnose("Warning! Debug version (tree, verbose) used!\n", 
			level = 0)
		overall.best.tree <- letter.tree
	}
	results <- vector("list", optimal.size + 1)
	results[[1]] <- overall.best.tree
	names(results) <- letters[1:length(results)]	#placeholder
	names(results)[1] <- "tree"	#
#
# Compute leaf numbers and "where"s.
#
	leaf.locations <- overall.best.tree$frame[, 1] == "<leaf>"
	where <- (1:nrow(overall.best.tree$frame))[leaf.locations]
	leaf.number <- dimnames(overall.best.tree$frame)[[1]][leaf.locations]	#
#
# Now go through the leaves and re-fit, this time, of course, using all the data.
# Store the results and set the name on the result.list
#
	for(leaf in leaf.start:length(where)) {
		items.in.leaf <- overall.best.tree$where == where[leaf]
		new.k.vec <- k.vec[k.vec < sum(items.in.leaf)]
		if(length(new.k.vec) == 0)
			new.k.vec <- 1
		diagnose("...   ...Leaf", leaf, "(where", where[leaf], ") has", 
			sum(items.in.leaf), "entries;", level = 0)
		knn.out <- knn.var.select(data[items.in.leaf,  ], k = new.k.vec,
			theyre.the.same = TRUE, scaling = scaling, backward = 
			backward, verbose = 0, use.big = use.big)
		knn.out$leaf <- leaf
		knn.out$where <- where[leaf]
		results[[leaf + 1]] <- knn.out
		names(results)[leaf + 1] <- leaf.number[leaf]
		diagnose("Just filled results[[", leaf + 1, "]]\n", level = 1)
	}
#
#
# Save the call, set the class, dump the output if requested, and return.
#
	results$call <- match.call()
	class(results) <- "knnTree"
	if(save.output != "") {
		assign("results", results, frame = 1)
		data.dump("results", save.output)
	}
	return(results)
}
