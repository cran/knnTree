\name{knn.var}
\alias{knn.var}
\title{K-Nearest Neighbor Classification With Variable Selection}
\description{Construct or predict with k-nearest-neighbor classifiers,
using cross-validation to select variables by forward or backward selection,
to choose the best k and to choose scaling methods.}
\details{\R{knn.var} constructs a k-nearest-neighbor classifier using
Euclidean metric.  Leave-one-out cross-validation together with stepwise (forward
or backward, but not both) selection is used to find the best set of variables to
include, the best choice of k, and whether the data should be scaled.}
\usage{knn.var (train, test, k.vec = seq (1, 31, by=2), 
theyre.the.same=FALSE,
return.all.rates=FALSE, scaling = 1, backward = FALSE, save.call = FALSE, 
verbose = 0, use.big=TRUE)}
\arguments{
\item{train} {data frame or matrix of training data with classifications 
in column 1}
\item{test} {data frame or matrix of training data with classifications 
in column 1. Optional if theyre.the.same = TRUE}
\item{k.vec} {numeric vector of possible values of k, the number of nearest neighbors.}
\item{theyre.the.same} {logical describing whether train and test are the same 
data set.  If so, test is ignored and leave-one-out cross-validation is used.  
This will normally be TRUE when building the classifier and FALSE when 
predicting.}
\item{return.all.rates} {logical, TRUE if all error rates (that is, one for
every element of k.vec) should be returned.  If FALSE, only the smallest is 
returned.}
\item{scaling} {numeric describing scaling technique: 0 means do no scaling;
1 means choose between no scaling and scaling each column by its SD; 2 means
choose between no scaling and scaling each column by its MAD.}
\item{backward} {logical describing variable selection technique.  TRUE
means start with all variables and delete them one at a time until there is
no improvement; FALSE means start with no variables and add them one at a 
time.}
\item{save.call} {logical, TRUE if a copy of the call should be saved in the
resulting object}
\item{verbose} {numeric for debugging purposes.  If verbose is 0, no
diagnostic output is produced.  If verbose > 0, diagnostic output (more
as the value increases) is placed in a file called "status.txt" in 
the \code{HOME} directory.  When verbose is 2 or (especially) 3 this file
may become very large.}
\item{use.big} {logical, TRUE if the C code should try to use a
technique that uses more memory but runs faster.}
}
\value{Object of class knn.  This is a list with between six and eight
of the following components:\cr
which: logical vector, one per input variable; the i-th element of which is TRUE if the i-th input variable is in the classifier\cr
rate: Smallest misclassification rate acheived by algorithm.  If 
return.all.rates is TRUE this is a vector of error rates, one for each element
of k.vec\cr
best.k: Number giving the optimal value of k, chosen from among the elements
of k.vec.\cr
scaled: indicator of best scaling.  FALSE means no scaling was used; TRUE 
means scaling was used. \cr
n: the number of observations in the training set\cr
col.sds: Numeric vector of scaling factors, present only if scaled != 0.  
If scaled = 1 these are column SD's; if scaled = 2 they are MAD's.\cr
pure: logical, TRUE if every item in the training set had the same class.  If
a training set is pure then all the elements of which are FALSE, best.k is
taken to be the first element of k.vec and scaled is set to 0.\cr
call: a copy of the call used to create the object, if save.call was TRUE
}

\examples{
data(iris)
set.seed (3)
samp <- sample (nrow(iris), size = 75, replace=FALSE)
knn.var (iris[samp,c(5, 1:4)]) # Build classifier
# Output produced by \code{print.knn}
\dontrun{This knn classifier is based on 75 observations.
It uses 1 out of 4 variables without scaling.

Training rate is 0.01333 , achieved at k = 1}
iris.knn <- knn.var (iris[samp,c(5, 1:4)]) # Build and save, then predict
predict (iris.knn, iris[-samp,c(5, 1:4)], iris[samp, c(5, 1:4)])
\dontrun{$rate
[1] 0.08}
}

\keyword{models}
\references{Buttrey and Karo, 2002}
\author{Sam Buttrey \email{buttrey@nps.navy.mil}}
\seealso{\code{\link{knnTree}}}
