\name{predict.knn}
\alias{predict.knn}
\title{Predict using a knn object}
\description{Predict class labels from a knn object and some data}
\details{This uses a knn object created by \R{knn.var.select} to make
predictions on some data.}
\usage{predict.knn (object, test, train, theyre.the.same = FALSE,
return.classifications = FALSE, verbose=0, ...)}
\arguments{
\item{object} {knn model created by \R{knn.var.select}}
\item{test} {data frame or matrix of test data with classifications 
in column 1.}
\item{train} {data frame or matrix of training data with classifications 
in column 1}
\item{theyre.the.same} {logical describing whether train and test are the same 
data set.  If so, train is ignored.}
\item{return.classifications} {logical: if TRUE, classifications for each
element in the test set are returned.}
\item{verbose} {numeric for debugging purposes.  If verbose is 0, no
diagnostic output is produced.  If verbose > 0, diagnostic output (more
as the value increases) is placed in a file called "status.txt" in 
the \code{HOME} directory.  When verbose is 2 or (especially) 3 this file
may become very large.}
\item{...} {Other arguments, currently ignored}
}
\value{List with a component named "rate" giving test set error rate. If
return.classifications = TRUE, a second item named classifications is a vector
with the classification of each item in the test set.}

\keyword{models}
\references{Buttrey and Karo, 2002}
\author{Sam Buttrey \email{buttrey@nps.navy.mil}}
\seealso{\code{link{knn.var}}, \code{\link{knnTree}}}
