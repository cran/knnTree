\name{predict.knnTree}
\alias{predict.knnTree}
\title{Predict using a knnTree object}
\description{Predict class labels from a knnTree object and some data}
\details{This uses a knnTree object created by \R{knnTree} to make
predictions on some data.}
\usage{predict.knnTree (object, test, train, verbose=FALSE, ...)}
\arguments{
\item{object} {knnTree model created by \R{knnTree}}
\item{test} {data frame or matrix of test data with classifications 
in column 1.}
\item{train} {data frame or matrix of training data with classifications 
in column 1}
\item{verbose} {logical; if TRUE the error rate for each leaf is printed}
\item{...} {Other arguments, currently ignored}
}
\value{A vector of classifications of the test data.}

\keyword{models}
\references{Buttrey and Karo, 2002}
\author{Sam Buttrey \email{buttrey@nps.navy.mil}}
\seealso{\code{link{knn.var}}, \code{\link{knnTree}}}
