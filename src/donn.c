/*
**      do_nn (): S-Plus-to-C version of nearest neighbor
**
**    NOTE : EVERYTHING IS TRANSPOSED IN THIS VERSION!
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "matrix.h"
#include "utils.h"

#include "donn.h"

#define poll pollx

#ifdef USE_S_ALLOC
extern char *S_alloc();
extern void do_nothing();
#define calloc S_alloc
#define free do_nothing;
#endif
/*
** If big_and_fast is defined, we will compute distances from every test set item to every
** training set item, for each component, one time when we're first called. We determine
** this by looking at "action" the first time through. If "theyre_the_same," we'll also
** try to save time by storing all the distances from item i to item j, i < j. We can then
** use the same (already-calculated) distance when it comes time to compare items j and i.
** This part we need to do separately each time. For the moment we'll try to do it whenever
** theyre_the_same is TRUE, regardless of big_and_fast.
*/
static MATRIX **dist_holder;
MATRIX *within_sample_distances;
static int big_and_fast;

double hold_this; /* testing only */


int poll(long *classes, double *distances, long *k, long how_many_ks, int number_of_classes,
		 long largest_k, long slots, MATRIX *prior, long *outcome, int quit, 
                 FILE *status_file, int poll_debug);
double c_euclidean (double *, double *, double *, long, double);
double c_transposed_euclidean (MATRIX *m_1, long m1_col, MATRIX *m_2, long m2_col,
							   double *c, double threshold, FILE *status_file);
double c_transposed_euclidean_big_and_fast (MATRIX **dist_holder, long m1_col, long m2_col,
		    double *c, unsigned long c_col, double threshold, FILE *status_file);
double f_euclidean (double *vec_1, double *vec_2, double *phi, double *c,
                    long n, double threshold, long *cats_in_var, long *cum_cats,
                    MATRIX *prior, double **knots);
double c_absolute (double *, double *, double *, long, double);

/*=========================== do_nn =================================*/

int do_nn (long action, long first_time, MATRIX *training, MATRIX *test,
           MATRIX *c, long *k, long how_many_ks,
           long theyre_the_same, int number_of_classes,
           MATRIX *cost, MATRIX *prior, double *error_rates,
           MATRIX *misclass_mat, long return_classifications,
           long *classifications, int verbose, FILE *status_file)
{
/*
** This is the simple version of donn. The metric is euclidean, with variable weights
** given in the "c" matrix. Theyre_the_same if training and test are the same: in that
** case item i of the training set can't be a neighbor of item i of the test set.
*/
long j, another_j; unsigned long train_ctr, test_ctr;
long test_item_count;
long dist_ctr, move_ctr, k_ctr,
       number_of_nearest,
       largest_k = -1,
       test_class, train_class;
double train_val, test_val;

int all_training_items_are_in_same_class;    /* If node is pure, this problem's easy  */
int all_cs_are_0;                            /* If all c's are 0, this problem's easy */
long class_of_training_item_0 = 0L;          /* Helps determine whether node is pure  */
long number_of_vars;                         /* Number of variables in the problem    */
double dist;                                 /* Dist: test item to current trg item   */
long max_class_count;                        /* Largest count in any class (for naive)*/
int class_with_most;                         /* Number of class with most entries (") */

static int initialized = 0;                  /* Is this initialized?                  */
static double *nearest_distance;             /* Distance to closest neighbor          */
static long *nearest_class;                  /* Class of nearest neighbor             */
static long *nearest_neighbor;               /* Number of NN                          */
static long *poll_result;                    /* Winner of poll (est'd. class)         */
static long *misclass_with_distance_zero;    /* ?                                     */
static long *class_ctrs;                     /* Ctrs for computing naive model        */
static long slots;                           /* Number of neighbors to keep track of. */
static int use_within = 0;                   /* Try to use within_sample dists?       */

double *test_ptr;   /* test only */
int poll_debug = 0; /* test only */


/*
** Note that we use nrow, because everything is transposed.
*/

number_of_vars = training->nrow - 1L;

/*
** 0th thing: If "action" is DO_NN_QUIT, free up all that stuff we allocated earlier.
*/

if (action == DO_NN_QUIT)
{
    if (initialized)
    {
        free (nearest_distance);
        free (nearest_neighbor);
        free (nearest_class);
        free (poll_result);
        free (misclass_with_distance_zero);
        if (big_and_fast == TRUE)
        {
            for (j = 1; j < number_of_vars + 1; j ++)
            {
                free (dist_holder[j]->data);
		free (dist_holder[j]);
            }
	    big_and_fast = FALSE;
	}
        initialized = FALSE;
    }
    return (0);
}

/*
** Set all the error rates to 0.
*/
for (k_ctr = 0; k_ctr < how_many_ks; k_ctr++) {
	error_rates[k_ctr] = 0.0;
}

/*
** Disable within_sample matrix
*/
use_within = FALSE;

/*
** ============================ NAIVE MODEL CHECK ======================================
** First thing: if all the entries of "c" (starting with #1, not with #0, which we don't
** care about) are 0, then we don't need to compute any distances. We just find the most
** common class and assign every test item to that.
*/
all_cs_are_0 = TRUE;
for (j = 1; j < (long) test->nrow; j++)
{
    if (*SUB (c, 0L, j) != 0)
    {
        all_cs_are_0 = FALSE;
        break;
    }
}

if (all_cs_are_0)
{
    if (alloc_some_longs (&class_ctrs, number_of_classes) != 0)
    {
        if (verbose > 0)
	    fprintf (status_file, "Unable to get %i longs for class ctrs; abort\n", number_of_classes);
	    return (-1);
	}
	for (j = 0; j < number_of_classes; j++)
            class_ctrs[j] = 0L;
	for (train_ctr = 0; train_ctr < training->ncol; train_ctr++)
        {
            train_class = (long) *SUB (training, 0L, train_ctr);
            class_ctrs[train_class]++;
        }
	class_with_most = 0;
	max_class_count = class_ctrs[0];
	if (verbose > 1)
            fprintf (status_file, "Check: found %li in class %i (naive)\n", class_ctrs[0], 0);
	for (j = 1; j < number_of_classes; j++)
	{
        if (verbose > 1)
            fprintf (status_file, "Check: found %li in class %li (naive)\n", class_ctrs[j], j);
		if (class_ctrs[j] > max_class_count)
		{
			class_with_most = j;
			max_class_count = class_ctrs[j];
		}
	}
	if (verbose > 0) {
            fprintf (status_file, "Winner: found %li in class %i\n", max_class_count, class_with_most);
	    fflush (status_file);
        }

	for (test_ctr = 0; test_ctr < test->ncol; test_ctr++)
	{
            test_class = (long) *SUB (test, 0L, test_ctr);      /* This is the true classification. */
/*
** If a misclass matrix was specified, use it. This has the same problem here as down below;
** it's not correct unless k_ctr == 1.
*/
            for (k_ctr = 0; k_ctr < how_many_ks; k_ctr++)
            {
                if (misclass_mat != (MATRIX *) NULL)
                    *SUB (misclass_mat, test_class, class_with_most)
		                = *SUB (misclass_mat, test_class, class_with_most) + 1;
                if (return_classifications == TRUE)
		   	        classifications[test_ctr] = class_with_most;

                if (class_with_most != test_class)
                {
/*
** If a cost matrix was passed in, use it. The relevant entry is the one
** with the true class as the row index and the prediction as the column.
** If there's no cost matrix, just add one to the relevant error counter.
*/
                    if (cost == (MATRIX *) NULL)
                        error_rates[k_ctr]++;
                    else
                        error_rates[k_ctr] += *SUB (cost, test_class, class_with_most);
/* This code comes from below. I'm going to ignore it for now.
**                if (nearest_distance[0] == 0.0)
**                  misclass_with_distance_zero[k_ctr]++;
*/
                } /* end "if the prediction doesn't match the true class */

                if (verbose >= 2 && k_ctr == 0)
                    fprintf (status_file,
                        "k = %ld: Classified test rec. %li (a %li) as %i (naive model)\n",
                         k[k_ctr], test_ctr, test_class, class_with_most);
                } /* end "for k_ctr" loop to fill misclasses, errors, and classifications. */
        } /* end "for test_ctr" loop over test set */

        if (verbose > 0)
            fflush (status_file);

        for (k_ctr = 0; k_ctr < how_many_ks; k_ctr++)
        {
            if (verbose >= 2)
            {
                fprintf (status_file,
                    "k = %ld: misclassed %f records out of %li, fraction %f (naive)\n",
                    k[k_ctr], error_rates[k_ctr], test->ncol,
                    ((double) error_rates[k_ctr]) / ( (double) test->ncol));
            }
            error_rates[k_ctr] /= (double) test->ncol;
	} /* end "for k_ctr" loop to compute error rates. */

	free (class_ctrs);
	return 0;
}

/* ======================== END NAIVE MODEL ACTION ======================================*/


/*
** If this hasn't been initialized, do so. First find the largest k; then set up the number
** of slots (let's use, oh, 15 more than we have neighbors); then allocate some memory for
** that many nearest_distances, nearest_neighbors, nearest_classes, poll_results, and
** misclasses_with_distance_zero. Set these latter, plus the error rates, to be 0.
*/

/*
if (first_time == FALSE)
    initialized = TRUE;
*/

/*
** This stuff 
*/

if (initialized == FALSE)
{
    for (j = 0; j < how_many_ks; j++)
        if (k[j] > largest_k)
            largest_k = k[j];

    slots = largest_k + 15;

    if (alloc_some_doubles (&nearest_distance, (unsigned long) slots) != 0)
    {
        if (verbose > 0)
		fprintf (status_file, "Unable to get %li doubles 4 dists; abort\n", slots);
        return (-1);
    }

    if (alloc_some_longs (&nearest_neighbor, (unsigned long) slots) != 0)
    {
	if (verbose > 0)
            fprintf (status_file, "Unable to get %li longs 4 neighbors; abort\n", slots);
	    free (nearest_distance);
        return (-1);
    }

    if (alloc_some_longs (&nearest_class, (unsigned long) slots) != 0)
    {
        if (verbose > 0)
            fprintf (status_file, "Unable to get %li longs for classes; abort\n", slots);
        free (nearest_distance);
	free (nearest_neighbor);
        return (-1);
    }

    if (alloc_some_longs (&poll_result, (unsigned long) how_many_ks) != 0)
    {
	if (verbose > 0)
            fprintf (status_file, "Unable to get %li longs for results; abort\n",
                how_many_ks);
        free (nearest_distance);
        free (nearest_neighbor);
        free (nearest_class);
        return (-1);
    }

    if (alloc_some_longs (&misclass_with_distance_zero,
              (unsigned long) how_many_ks) != 0)
    {
	if (verbose > 0)
	    fprintf (status_file, "Unable to get %li longs for miss II; abort\n",
                 how_many_ks);
	free (nearest_distance);
	free (nearest_neighbor);
	free (nearest_class);
	free (poll_result);
        return (-1);
    }
/*
** Now the big one. If "action" is DO_NN_INITIALIZE_BIG, we're going to want to try
** to allocate (number of vars) matrices, each one training->ncol (remember this is
** transposed) * test->ncol. The (j,k)th entry of the ith matrix measures the squared
** distance between training set item j and test set item k on the ith variable. If we
** can't allocate all this mmemory,don't worry. We'll do it the old way. But we do want
** to back our way out nicely. Remember that J starts at 1, since the 0th row is the
** class membership. Number of vars doesn't count that class membership row. So we go
** up to "< number_of_vars + 1". Ex: number_of_vars = 10, we start at 1, go up to "< 11,"
** we end up with 11 pointers and ten matrices, of which we use 1, 2, ..., 10 but not 0.
*/
	big_and_fast = FALSE;
	if (action == DO_NN_INITIALIZE_BIG)
	{
	    big_and_fast = TRUE;
            if ( (dist_holder = (MATRIX **) calloc ((number_of_vars + 1), sizeof (MATRIX *))) == NULL) {
                if (verbose > 0)
                    fprintf (status_file, "Warning: unable to get matrices for big and fast allocation!\n");
            }
            else 
	    {
                for (j = 1; j < number_of_vars + 1; j++)
                {
                    dist_holder[j] = make_matrix (training->ncol, test->ncol, "Dist", REGULAR, TRUE);
                    if (dist_holder[j] == (MATRIX *) NULL)
                    {
                        big_and_fast = FALSE;
                        if (verbose > 0)
                            fprintf (status_file, "Dist alloc failed, j = %li\n", j);
                        for (another_j = 0; another_j < j; another_j++){
                            if (verbose > 0) {
                                fprintf (status_file, "free, another_j = %li, j = %li\n", another_j, j);
                                fflush (status_file);
                            }
                            free (dist_holder[another_j]->data);
                            free (dist_holder[another_j]);
                        }
                        break;
                    }
                } /* end "for" loop over vars */

                if (big_and_fast == FALSE) {
                    if (verbose > 0)
                        fprintf (status_file, "Warning: unable to do big and fast allocation!\n");
		}
		else
		{
/*
** Get all the distances into the proper places. J starts at one since the zeroth
** row is the class membership, and goes up to "< number_of_vars + 1" since "number
** of vars" is the number of data columns, and number_of_vars + 1 is the actual
** number of rows in the data.
*/
                      for (j = 1; j < number_of_vars + 1; j++)
                            for (test_ctr = 0; test_ctr < test->ncol; test_ctr++)
                                for (train_ctr = 0; train_ctr < training->ncol; train_ctr++) {
                                    train_val = *SUB (training, j, train_ctr);
                                    test_val  = *SUB (test,  j, test_ctr);
                                    *SUB (dist_holder[j], train_ctr, test_ctr) =
                                        (train_val - test_val) * (train_val - test_val);
                                }
                } /* end "if BIG_AND_FAST is TRUE after the allocations */
            } /* end "else", i.e. if allocation of dist_holder succeeded */
	} /* end "if action is DO_NN_INITIALIZE_BIG" */

        for (j = 0; j < how_many_ks; j++)
        {
            error_rates[j] = 0.0;
            misclass_with_distance_zero[j] = 0L;
        }
/*
** Set up the within_sample_distance matrix, if theyre_the_same is true. This should
** really be a symmetric matrix, but I'm a little concerned about those right now.
*/
    if(use_within && theyre_the_same)
    {
       	if (verbose > 0)
            fprintf (status_file, "About to allocate %li by %li\n", training->ncol, test->ncol);
        within_sample_distances = make_matrix (training->ncol, test->ncol, "WithinDist", REGULAR, TRUE);

        if (within_sample_distances == (MATRIX *) NULL && verbose > 0) {
            fprintf (status_file, "Warning: unable to do within-sample allocation!\n");
            within_sample_distances = (MATRIX *) NULL;
        }
    }
    else
        within_sample_distances = (MATRIX *) NULL;

    initialized = TRUE;

} /* end "if initialized */

/*
** Zero out the misclass_matrix, if there is one.
*/
if (misclass_mat != (MATRIX *) NULL)
{
    zero_matrix (misclass_mat);
}

if (verbose > 0)
    fflush (status_file);

test_item_count = 0L;
all_training_items_are_in_same_class = TRUE;


/*=========================== BEGIN TEST SET LOOP =============================*/
/*
** Now we go through the columns (because of the transpose) of the test set...
*/
/*==============================================================================*/
for (test_ctr = 0; test_ctr < test->ncol; test_ctr++)
{

    test_item_count++;
    if (verbose >= 3)
        fprintf (status_file, "Computing dists for test record %li\n", test_ctr);
/*
** ..zero out the nearest neighbor information...
*/
    for (j = 0; j < slots; j++)
    {
        nearest_neighbor[j] = (long) -1;
        nearest_distance[j] = -1.0;
        nearest_class[j]    = (long) -1;
    }
    number_of_nearest = 0;
/*
** ... and compute the distance from this record to each of the training
** set records, in turn. Again, use columns because of the transpose.
*/
    for (train_ctr = 0; train_ctr < training->ncol; train_ctr++)
    {
/*
** For the first test record, find out whether the training set is pure, that is,
** if every item has the same class as the first one. We save the class of the first
** item, then compare it to all other classes. By the time test_ctr = 1, we know.
*/
        if (test_ctr == 0) {
            if (train_ctr == 0) 
                class_of_training_item_0 = (long) *SUB (training, 0L, 0L);
            else  {
                if ( (long) *SUB (training, 0, train_ctr) != class_of_training_item_0)
                    all_training_items_are_in_same_class = FALSE;
            }
        }
        else
            if (all_training_items_are_in_same_class)
                break;

/*
** Skip this record if we're classifying from ourself...
*/
        if (theyre_the_same && test_ctr == train_ctr)
            continue;
/*
** Here's a time-saving step. If theyre_the_same is TRUE, then we save the "dist" if
** train_ctr > test_ctr, and we use the already-calculated one if train_ctr < test_ctr.
** Problem: the stored dist may be -1 (which happens when the distance-computing function
** interrupts itself in the middle); in that case we will need to do the calculation.
**
*/
	dist = -1.0;
	if (theyre_the_same && within_sample_distances != (MATRIX *) NULL &&  train_ctr < test_ctr)        {
	    dist = *SUB (within_sample_distances, train_ctr, test_ctr);
	}

	if (dist < 0) {
	    if (big_and_fast == TRUE) {
		    dist = c_transposed_euclidean_big_and_fast (dist_holder, train_ctr, test_ctr,
                        SUB (c, 0, 0), c->ncol, nearest_distance[slots - 1], status_file);
            }
	    else {
                test_ptr = SUB (c, 0, 0);
                dist = c_transposed_euclidean (training, train_ctr, test, test_ctr,
                    test_ptr, nearest_distance[slots - 1], status_file);
            }
        }

            if (theyre_the_same && within_sample_distances != (MATRIX *) NULL &&  train_ctr > test_ctr) {
                *SUB (within_sample_distances, test_ctr, train_ctr) = dist;
            }
/*
** The distance function is given a threshold, which is the largest of the
** "nearest-neighbor" distances. As soon as a distance gets above that, we
** know we can stop this comparison. The function returns -1, and we continue.
*/

        if (verbose >= 3) {
            fprintf (status_file, "Dist from (train) %li to (test) %li was %f!\n", train_ctr, test_ctr, dist);
	    fflush (status_file);
        }
        if (dist < 0)
            continue;
/*
** If this distance is smaller than the largest on our list, or if we haven't
** encountered "slots" records yet, begin the nearest neighbor processing.
*/
        if (dist < nearest_distance[slots-1] || number_of_nearest < slots)
        {
            if (verbose >= 3 && dist <= nearest_distance[0]) {
                fprintf (status_file, "Smallest so far for %li is %li, distance %f\n",
                         test_ctr, train_ctr, dist);
		fflush (status_file);
            }
/*
** Find the spot for this new neighbor. When we find it (and by "the spot"
** I mean the smallest current neighbor bigger than this distance) we move
** everybody from the spot forward one in the list and insert the new entry
** in that spot. Make sure to save the largest distance.
*/
            for (dist_ctr = 0; dist_ctr < slots; dist_ctr ++)
            {
/*
** If the current "nearest_distance" isn't a -1, and if it's smaller
** than "dist," move up to the next "nearest_distance.
*/
                if (nearest_distance[dist_ctr] >= 0.0 &&
                    dist >= nearest_distance[dist_ctr])
                    continue;
/*
** Otherwise, move everything from spot (i-1) to spot i,  and put
** information for this record into the appropriate spot.
*/
                for (move_ctr = slots - 1; move_ctr > dist_ctr; move_ctr --)
                {
                    nearest_distance[move_ctr] =
                        nearest_distance[move_ctr - 1];
                    nearest_class[move_ctr] = nearest_class[move_ctr - 1];
                    nearest_neighbor[move_ctr] =
                        nearest_neighbor[move_ctr - 1];
                }
                nearest_distance[dist_ctr] = dist;
                nearest_neighbor[dist_ctr] = train_ctr;
                nearest_class[dist_ctr] = (long) *SUB (training, 0, train_ctr);
                break;
            } /* end "for" loop on nearest neighbor arrays. */
            if (number_of_nearest < slots)
                number_of_nearest ++;
        } /* end "if this is a nearest neighbor" */
    } /* end "for" for looping over the training set. */

/*
** We've finished for this item. If the training set isn't pure, poll the nearest neighbors
** and get their predicted values, one for each value in the k vector. "Prior" will be used
** if it's supplied. If the training set is pure, then the test item must be assigned the
** class of all the training items.
*/
	if (all_training_items_are_in_same_class)
	{
		for (k_ctr = 0; k_ctr < how_many_ks; k_ctr++)
			poll_result[k_ctr] = class_of_training_item_0;
	}
	else {
                poll_debug = 0;
                if (poll_debug)  {
printf ("Poll on test %li, nearest are %li %li %li\n", test_ctr, nearest_class[0],
nearest_class[1], nearest_class[2]);
                }
		poll (nearest_class, nearest_distance, k, how_many_ks, number_of_classes,
			  largest_k, slots, prior, poll_result, FALSE, status_file, poll_debug);
        }

    test_class = (long) *SUB (test, 0L, test_ctr);      /* This is the true classification. */

/*
** Go through the k's. If a misclass matrix was supplied, fill it up. This only works if
** there's exactly one k: otherwise we're incrementing the misclass matrix once for each k.
*/

    for (k_ctr = 0; k_ctr < how_many_ks; k_ctr++)
    {
        if (misclass_mat != (MATRIX *) NULL)
            *SUB (misclass_mat, test_class, poll_result[k_ctr])
            = *SUB (misclass_mat, test_class, poll_result[k_ctr]) + 1;
		if (return_classifications == TRUE && k_ctr == 0)
			classifications[test_ctr] = poll_result[0];
        if (poll_result[k_ctr] != test_class)
        {
/*
** If a cost matrix was passed in, use it. The relevant entry is the one
** with the true class as the row index and the prediction as the column.
** If there's no cost matrix, just add one to the relevant error counter.
*/
            if (cost == (MATRIX *) NULL)
                error_rates[k_ctr]++;
            else
                error_rates[k_ctr]
                    += *SUB (cost, test_class, poll_result[k_ctr]);
            if (nearest_distance[0] == 0.0)
                misclass_with_distance_zero[k_ctr]++;
        }
		if (verbose >= 2 && k_ctr == 0)
			fprintf (status_file,
			"k = %ld: Classified test rec. %li (a %li) as %li (nearest: %li, dist. %f)\n",
			k[k_ctr], test_ctr, test_class, (long) poll_result[k_ctr],
			(long) nearest_neighbor[0], nearest_distance[0]);
    }

} /* end "for test_ctr" loop for test set. */

/*
** Call poll() with quit = TRUE to clean up a little. Other arguments are ignored.
*/
poll (nearest_class, nearest_distance, k, how_many_ks, number_of_classes,
          largest_k, slots, prior, poll_result, TRUE, status_file, poll_debug);

for (k_ctr = 0; k_ctr < how_many_ks; k_ctr++)
{
    if (verbose >= 2)
    {
        fprintf (status_file,
			"k = %ld: misclassed %f records out of %li, fraction %f\n",
            k[k_ctr], error_rates[k_ctr], test_item_count,
                ((double) error_rates[k_ctr]) / ( (double) test_item_count));
    }
    if (verbose >= 2)
    {
        fprintf (status_file,
            "Of those, %li (fraction %f) has nearest distance zero\n",
            misclass_with_distance_zero[k_ctr],
                 ((double) misclass_with_distance_zero[k_ctr])
                                          / ( (double) test_item_count));
    }

    error_rates[k_ctr] /= (double) test_item_count;

}

if (within_sample_distances != (MATRIX *) NULL)
    free (within_sample_distances->data);


free (nearest_distance);
free (nearest_neighbor);
free (nearest_class);
free (poll_result);
free (misclass_with_distance_zero);
/*
initialized = FALSE;
*/

return 0;

} /* end "do_nn." */

/*============================  poll  =====================================*/
int poll (long *classes, double *distances, long *k, long how_many_ks, int number_of_classes,
          long largest_k, long slots, MATRIX * prior, long *outcome, int quit, 
          FILE *status_file, int poll_debug)
{
static double *class_results;
static double *class_results_with_ties;
static int *tie_marker;
int i, k_ctr;
int tie;
double max_count;
int max_class = -1;

static long initialized = 0;


/*
** If "quit" is TRUE, de-allocate space and leave.
*/

if (quit == TRUE)
{
	free (class_results);
	free (class_results_with_ties);
	free (tie_marker);
	initialized = FALSE;
	return (0);
}

/*
** If "initialized" is FALSE, allocate some space we'll need.
*/


if (initialized == FALSE)
{
if (poll_debug)
printf ("Poll: not initized!\n");
    if (alloc_some_doubles (&class_results, number_of_classes) != 0)
    {
        if (status_file != (FILE *) 0)
			fprintf (status_file, "Couldn't get %i doubles for poll results; abort\n",
            number_of_classes);
		return (-1);
    }
    if (alloc_some_doubles (&class_results_with_ties, number_of_classes) != 0)
    {
        if (status_file != (FILE *) 0)
	    fprintf (status_file, "Couldn't get %i ints for poll results; abort\n",
                number_of_classes);
	free (class_results);
	return (-1);
    }
    if (alloc_some_ints (&tie_marker, number_of_classes) != 0)
    {
		if (status_file != (FILE *) 0)
			fprintf (status_file, "Couldn't get %i ints for ties in poll; abort\n",
            number_of_classes);
		free (class_results);
		free (class_results_with_ties);
		return (-1);
    }
    initialized = TRUE;
}
else
    if (poll_debug)
        printf ("Poll: initized!\n");

/* Zero out the results and tie markers arrays. */
for (i = 0; i < number_of_classes; i++)
{
    class_results[i] = 0.0;
    class_results_with_ties[i] = 0.0;
    tie_marker[i] = 0;
}

/* Okay. Now we go through the list of k's. First of all, if any
** k is 1 or 2, return the class of the nearest neighbor. (This is
** clear for k = 1. For k = 2, ties are broken by the first nearest
** neighbor anyway.)
*/
for (k_ctr = 0; k_ctr < how_many_ks; k_ctr++)
{
/*** This is wrong. These things can be ties.
    if (k[k_ctr] == 1 || k[k_ctr] == 2)
    {
        outcome[k_ctr] = classes[0];
        continue;
    }
****/

/* Zero out the "class_results" array ... */
    for (i = 0; i < number_of_classes; i++)
        class_results[i] = 0.0;

/* ...and go through the neighbors to fill it up again. When classes[i]
** = j, add one to the j-th entry of class_results. Well, not one,
** exactly; if priors isn't NULL, add 1/(that class' prior). That
** way, classes with large priors contribute less. Which is as it should be.
*/
    for (i = 0; i < k[k_ctr]; i++)
    {
if (poll_debug)
printf ("k is %li; Classes %i is %li\n", k[k_ctr], i, classes[i]);
        if (prior == (MATRIX *) NULL)
            class_results[classes[i]] ++;
        else
            class_results[classes[i]] +=
                 (1.0 / *SUB (prior, classes[i], classes[i]));
    }
/*
** Okay. It could happen that some other neighbors are tied
** with the kth one. We would know that if their distances equalled the
** kth distance. First copy the "class_results" array to the handy
** "class_results_with_ties"; then look through the remaining neighbors
** (there are "slots" neighbors) to see if any of them should be counted.
*/

    for (i = 0; i < number_of_classes; i++)
        class_results_with_ties[i] = class_results[i];

    i = k[k_ctr];

    while (i < slots && distances[i] == distances[k[k_ctr]-1])
    {
        if (prior == (MATRIX *) NULL)
            class_results_with_ties[classes[i]] ++;
        else
            class_results_with_ties[classes[i]] +=
                 (1.0 / *SUB (prior, classes[i], classes[i]));
        i++;
    }

/* Now we're effectively using i-nn, not just k-nn. So reset k. */

/* Now we want to find the maximum. In case of a tie, we use...*/
    tie = 0;
    max_class = -1;
    max_count = -1;
/*
** For each class, see how many entries in the "classes" array have that number.
*/
    for (i = 0; i < number_of_classes; i++)
    {
/*
** "Results[i]," then, is the number of times "i" appears in "classes." If
** this is smaller than max_count, move on. If it's equal, note that a tie
** exists.  Otherwise, save the count and the number of this class.
*/
        if (class_results_with_ties[i] < max_count)
            continue;
        if (class_results_with_ties[i] == max_count)
        {
            tie = 1;
            continue;
        }
        tie = 0;
        max_class = i;
        max_count = class_results_with_ties[i];
    }

    if (tie == 0)
    {
        outcome[k_ctr] = max_class;
        continue;
    }

/* Make a note of all tied classes.... */
    for (i = 0; i < number_of_classes; i ++)
    {
        if (class_results_with_ties[i] == max_count)
            tie_marker[i] = 1;
    }
/* ...and return the first class that belongs to one of the tied ones. */
    for (i = 0; i < k_ctr; i++)
        if (tie_marker[classes[i]] == 1)
            outcome[k_ctr] = classes[i];

/* We should never get here. */
    outcome[k_ctr] = max_class;

} /* end "for k_ctr" counting through the k's. */

return (0);

} /* end "poll" */

/*=========================  c_euclidean  ==================================*/

double c_euclidean (double *vec_1, double *vec_2, double *c,
                    long n, double threshold)
{
long i;
double sum;

sum = 0.0;
for (i = 0; i < n; i++)
{
    if (c[i] == 0)
        continue;
    sum += c[i] * (vec_1[i] - vec_2[i]) * (vec_1[i] - vec_2[i]);
    if (threshold > 0 && sum > threshold)
        return (-1.0);
}
return (sum);

} /* end "euclidean" */

/*===================  c_tranposed_euclidean  ============================*/

/*
** This computes distances between the m1_col'th column of matrix m_1 and the
** m2_col'th column of m_2. The first element in each column is skipped, since
** that's just the class specification. c holds one weight for each row (the first
** entry is not used). We compute the sum of squared differences, except that if
** that sum exceeds "threshold" we can stop.
*/
double c_transposed_euclidean (MATRIX *m_1, long m1_col, MATRIX *m_2, long m2_col,
                               double *c, double threshold, FILE *status_file)
{
unsigned long i;
double sum, m1_entry, m2_entry;

sum = 0.0;
for (i = 1L; i < m_1->nrow; i++) /* Remember not to start at 0 */
{
    if (c[i] == 0)
        continue;
	m1_entry = *SUB (m_1, i, m1_col);
	m2_entry = *SUB (m_2, i, m2_col);
    sum += c[i] * (m1_entry - m2_entry) * (m1_entry - m2_entry);
    if (threshold > 0 && sum > threshold)
        return (-1.0);
}
return (sum);

} /* end "c_transposed_euclidean" */

/*===================  c_tranposed_euclidean_big_and_fast  ============================*/

/*
** This computes distances between entries m1_col and m2_col for each non-zero element
** of "c" (except the first) by looking the component-wise distances up in dist_holder
** and adding them up. Once the sum exceeds "threshold" we can stop.
*/
double c_transposed_euclidean_big_and_fast (MATRIX **dist_holder, long m1_col, long m2_col,
					double *c, unsigned long c_col, double threshold, FILE *status_file)
{
unsigned long i;
double sum;

sum = 0.0;
for (i = 1L; i < c_col; i++) /* Remember not to start at 0 */
{
    if (c[i] == 0) {
        continue;
	}
    sum += c[i] * *SUB (dist_holder[i], m1_col, m2_col);
    if (threshold > 0 && sum > threshold)
        return (-1.0);
}
return (sum);

} /* end "c_transposed_euclidean_big_and_fast" */

/*=========================  c_absolute  ==================================*/
double c_absolute (double *vec_1, double *vec_2, double *c,
                   long n, double threshold)
{
long i;
double sum;

sum = 0.0;
for (i = 0; i < n; i++)
{
    sum += c[i] * fabs (vec_1[i] - vec_2[i]);
    if (threshold > 0 && sum > threshold)
        return (-1.0);
}
return (sum);

} /* end "c_absolute" */
