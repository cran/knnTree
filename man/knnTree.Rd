\name{knnTree}
\alias{knnTree}
\title{K-NEAREST NEIGHBOR CLASSIFIERS WITHIN LEAVES OF A TREE}
\description{Construct or predict with a knnTree object, which is
a set of k-nearest neighbor classifiers, one for each leaf of a tree.}
\usage{knnTree (trg.set, trg.classes, v = 10, 
k.vec = seq(1, 31, by = 2), seed = 0, opt.tree = "ignore", 
opt.tree.size = 4, scaling = 1, prune.function = prune.misclass, 
one.SE = TRUE, backward = FALSE, v.start = 1, leaf.start = 1, 
verbose = FALSE, debug = 0, fname = "", use.big = FALSE, save.output = FALSE)
}
\arguments{
\item{trg.set} {data frame or matrix of training data without classifications}
\item{trg.classes} {categorical vector of training set classifications}
\item{v} {numeric, number of blocks for cross-validation}
\item{k.vec} {numeric vector of numbers of k to consider}
\item{seed} {if present, passed to \code{set.seed()} to initialize the random
number generator}
\item{opt.tree} {character, giving method by which to choose the size
of the tree.  Choices are ignore (consider all sizes up to size of first
unpruned tree); find (consider all sizes up to size of first pruned tree);
fix (use tree size passed in opt.tree.size); max (consider all sizes
<= opt.tree.size)}
\item{opt.tree.size} {tree size used if opt.tree = fix or max}
\item{scaling} {numeric describing scaling technique: 0 means do no scaling;
1 means choose between no scaling and scaling each column by its SD; 2 means
choose between no scaling and scaling each column by its MAD.}
\item{prune.function} {function to do pruning, normally prune.tree or
prune.misclass}
\item{one.SE} {logical; if TRUE, prune and then use one-SE rule}
\item{backward} {logical describing variable selection technique.  TRUE
means start with all variables and delete them one at a time until there is
no improvement; FALSE means start with no variables and add them one at a 
time.}
\item{v.start} {number of cross-validation block to start at; for 
debugging only}
\item{leaf.start} {number of leaf to start at; for debugging only}
\item{verbose} {numeric for debugging purposes.  If verbose is 0, no
diagnostic output is produced.  If verbose > 0, diagnostic output (more
as the value increases) is sent to file fname, which is the screen if
fname is the empty string.}
\item{debug} {currently not used}
\item{fname} {string naming the file to which diagnostic output is sent
if verbose > 0}
\item{use.big} {logical, TRUE if the C code should try to use a
technique that uses more memory but runs faster.}
\item{save.output} {character; if not empty, the resulting object is
assigned to results in frame 1 and also dumped to disk
in the file named in save.output. This can be useful for parallel
processing.}
}
\value{Object of class knnTree. If the tree has n leaves, this will
be a list with n+2 elements. The first is the global tree. The next n 
elements are the n individual knn.var objects, one per leaf. Each of these
objects has two additional pieces: leaf (giving the leaf number) and
where (giving the row number of the global tree's frame for this leaf).
The n+2-th element of the list is named call and is the call used to 
create the object.}
\keyword{models}

