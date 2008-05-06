#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "matrix.h"
#include "utils.h"
#include "donn.h"

#ifdef USE_S_ALLOC
extern char *S_alloc();
extern void do_nothing();
#define malloc S_alloc
#define free do_nothing
#endif

#ifdef BUILDING_FOR_R
#define INT_OR_LONG int
#else
#define INT_OR_LONG long
#endif

#define DONT_SCALE 0
#define USE_SDS    1
#define USE_MAD    2

FILE *status_file;   /* Place to put status messages if incoming_verbose > 0 */

/*
** This function has two purposes. If theyre_the_same is true, then we're 
** trying to find the best k-NN classifier based on leave-one-out cross-
** validation. "k-NN" here includes finding the best k, the best subset 
** of variables, and whether or not the data is scaled.  When theyre_the_same 
** is true, we examine "scaled" to see whether we should attempt scaling
** as part of finding the best classifier. If "scaled" is DONT_SCALE, don't 
** try scaling; if it's USE_SDS, we return the SD of each column in the 
** relevant spot of col_sds (then on return we set scaled to TRUE if the 
** classifier based on the scaled data did better than the one based on 
** the unscaled, FALSE if unscaled did best); if scaled is equal to USE_MAD,
** we compute the median absolute deviation instead of the SD (with the same 
** return value). c_dat holds 1 for the columns used by the classifier and 
** 0 for the ones not used.
**
** If theyre_the_same is not true, we're doing regular k-NN classification 
** of the test set by the training set, returning one error rate for each 
** value of the k_vec. We use the weights in the c_dat, and if "scaled" is 
** TRUE we scale by the values in "col_sds."
*/

void *knnvar (double *train_dat,           /* Training data               */
         INT_OR_LONG *train_dim,           /* Training data (nrow, ncol)  */
	 double *test_dat,                 /* Test data                   */
         INT_OR_LONG *test_dim,            /* Test data (nrow, ncol)      */
	 INT_OR_LONG *number_of_classes,   /* Number of classes           */
	 INT_OR_LONG *k_vec,               /* Vector of choices for k     */
	 INT_OR_LONG *number_of_ks,        /* Length of k_vec             */
	 INT_OR_LONG *theyre_the_same,     /* Are train and test the same?*/
	 double *best_error_rate,          /* Vector of error rates       */
	 INT_OR_LONG *return_all_rates,    /* Return all rates or just best? */
	 INT_OR_LONG *best_k_index,        /* Number of best element of k_vec*/
	 double *c_dat,                    /* Vector of weights for columns  */
	 INT_OR_LONG *scaled,              /* "Best" classifier's scaltype ? */
	 double *col_sds,                  /* Sds of each column          */
	 INT_OR_LONG *return_classifications, /* Return 'em if true...    */
	 INT_OR_LONG *classifications,     /* ...putting 'em here         */
	 INT_OR_LONG *backward,            /* Step backward or forward?   */
         INT_OR_LONG *max_steps,           /* Max. number of steps to take*/
	 INT_OR_LONG *incoming_verbose,    /* Level of verbosity          */
	 char **status_file_name,          /* Place to dump messages      */
	 INT_OR_LONG *status)              /* Info. about any failures    */
{


MATRIX *train, *test;                 /* Matrices holding train and test data*/
MATRIX *c, *c_unscaled;               /* Holds coefficients of each variable */
unsigned long i; long k_ctr;          /* Counters                            */
long action = FALSE;                  /* Item needed by do_nn                */
/* FILE *status_file = (FILE *) 0;       ** File into which messages are put */
double best_error_rate_this_set;      /* Best err rate when this one deleted..*/
int best_k_index_this_set = 0;        /* ...and the corresponding k index    */
double best_error_rate_after_deletion;/* Best rate when any var...           */
int best_k_index_after_deletion = 0;  /* ...the corresponding k index...     */
long best_variable_to_delete = 0L;    /* ...and the variable that did it     */
double best_rate_unscaled;            /* Best rate after unscaled proc'ng... */
int best_k_index_unscaled;              /* ...and the corresponding k index  */
int dont_return_classifications = FALSE;/* Says "just compute misclass rates"*/
char *action_ed, *action_ing;           /* String for printing debug msgs.   */
int first_time;                         /* First time calling do_nn?         */
long number_of_steps;                   /* Number of steps taken             */
int verbose;                            /* "int" version of verbose          */

double *error_rates,
       *best_error_rates_after_deletion,
       *best_error_rates_unscaled,
	   *best_error_rates_scaled;
MATRIX *cost = (MATRIX *) 0, *prior = (MATRIX *) 0,  
       *misclass_mat = (MATRIX *) 0;
verbose = (int) *incoming_verbose;
verbose = 0;

if (*backward) {
    action_ed = "deleted";
	action_ing = "deleting";
}
else {
    action_ed = "added";
	action_ing = "adding";
}

/*
** Start by making a matrix of training data. "Allocate data" is of course 
** FALSE; we then point the "data" element of the matrix to the train_dat. 
** Likewise for test.  ** Remember these matrices come in in column order. 
** That means we need to switch the dimensions, and to be aware that they're 
** transposed compared to S-Plus.
*/


/* Set up the status file if verbose is > 0. */
if (verbose > 0) {
	status_file = fopen (*status_file_name, "a");
	fprintf (status_file, "File name is %s\n", *status_file_name);
        fflush (status_file);
}
train = make_matrix ((unsigned long) train_dim[1], (unsigned long) train_dim[0],
					 "Training set", REGULAR, FALSE);
train->data = train_dat;
test = make_matrix ((unsigned long) test_dim[1], (unsigned long) test_dim[0],
					"Test set", REGULAR, FALSE);
test->data = test_dat;

/* C has one entry for each row (after transpose) of test and train. */

c = make_matrix (1L, train->nrow, "Weights", REGULAR, FALSE);
c->data = c_dat;

if (*theyre_the_same == FALSE)
{
/*========================= Thread 1 =========================================*/
/*
** One thread of the function comes here. If "theyre_the_same" is FALSE, we call
** do_nn one time. We need to scale the data if asked.
*/

	if (alloc_some_doubles (&error_rates, (unsigned long) *number_of_ks))
	{
		*status = -1L;
		if (verbose > 0) fclose (status_file);
		return (void *) 0;
	}


/* Right now, USE_MAD is not implemented. */
	if (*scaled == USE_SDS || *scaled == USE_MAD)
	{
/*
** Scale the train and test matrices. No need to center.
** Set the first element of "c" to be 0 so the classes don't get re-scaled.
*/
		*SUB (c, 0L, 0L) = 0.0;

		scale_matrix_rows (train, FALSE, TRUE, c, USE_THESE_SCALINGS, (double *) 0, col_sds);
		scale_matrix_rows (test, FALSE, TRUE, c, USE_THESE_SCALINGS, (double *) 0, col_sds);

	}


	if (*status == 2)
            action = DO_NN_INITIALIZE; /* Don't do "big" on thread 1; it won't help. */
	else
            action = DO_NN_INITIALIZE;
        first_time = TRUE;
	do_nn (action, first_time, train, test, c, k_vec, *number_of_ks,  
               *theyre_the_same, *number_of_classes, cost, prior, error_rates, 
               misclass_mat, *return_classifications,
			   classifications, verbose, status_file);

	*best_error_rate = 1.1;
	for (k_ctr = 0; k_ctr < *number_of_ks; k_ctr++)
	{
		if (*return_all_rates == TRUE) {
			best_error_rate[k_ctr] = error_rates[k_ctr];
		}
		if (error_rates[k_ctr] < *best_error_rate)
		{
			if (*return_all_rates == FALSE) {
				*best_error_rate = error_rates[k_ctr];
			}
			*best_k_index    = k_ctr;
		}
	}

	free (error_rates);

        first_time = FALSE;
	do_nn (DO_NN_QUIT, first_time, train, test, c, k_vec, *number_of_ks,  *theyre_the_same, 
             *number_of_classes, cost, prior, error_rates, misclass_mat, *return_classifications,
              classifications, verbose, status_file);

	if (verbose > 0) fclose (status_file);
	*status = 0L;
	return ( (void *) 0);
}

/*========================= End of Thread 1 ==================================*/


/*
** Set all the c's to have 1's in them (if we're going backward) 
** or 0's (if forward) 
*/


if (verbose > 0) {
    fprintf (status_file, "Hey There! We're in thread 2!\n");
    fflush (status_file);
}

if (*backward)
    for (i = 0; i < c->ncol; i++)
	    *SUB (c, 0, i) = 1.0;
else
    for (i = 0; i < c->ncol; i++)
	    *SUB (c, 0, i) = 0.0;

/*
** Allocate space for error rates
*/
if (verbose > 0) {
fprintf (status_file, "Okay, we're this far, with %li k's\n", 
                       (long) *number_of_ks);
    fflush (status_file);
}

if (alloc_some_doubles (&error_rates,           (unsigned long) *number_of_ks)
    || alloc_some_doubles (&best_error_rates_after_deletion, 
                                                (unsigned long) *number_of_ks)
    || alloc_some_doubles (&best_error_rates_unscaled,       
                                                (unsigned long) *number_of_ks)
    || alloc_some_doubles (&best_error_rates_scaled,
                                                (unsigned long) *number_of_ks) )
{
    *status = -1L;
    if (verbose > 0) fclose (status_file);
        return (void *) 0;
}

/*
** Step 1: get k-NN error rates for each k, all variables included (if backward)
** or none included (if forward). The variables action_ing and action_ed contain
** the strings "adding"/"added" or "deleting"/"deleted" as appropriate.
*/

*theyre_the_same = TRUE;

if (*status == 2)
    action = DO_NN_INITIALIZE_BIG;
else
    action = DO_NN_INITIALIZE;

if (verbose > 0) {
    fprintf (status_file, "About to call with verbose = %i\n", verbose);
    fflush (status_file);
}

first_time = TRUE;
if (verbose > 0) {
fprintf (status_file, "Calling for the first time!\n");
    fflush (status_file);
}
do_nn (action, first_time, train, train, c, k_vec, *number_of_ks,  
           *theyre_the_same, *number_of_classes,
           cost, prior, error_rates, misclass_mat, dont_return_classifications,
		   (INT_OR_LONG *) 0, verbose, status_file);
first_time = FALSE;

action = DO_NN_COMPUTE;

/*
** Find the smallest error rate and the k it belongs to.
*/
*best_error_rate = 1.1;
for (k_ctr = 0; k_ctr < *number_of_ks; k_ctr++)
{
    best_error_rates_unscaled[k_ctr] = error_rates[k_ctr];
    if (error_rates[k_ctr] < *best_error_rate) {
    	*best_error_rate  = error_rates[k_ctr];
    	*best_k_index     = k_ctr;
    }
}

if (verbose > 0)
{
    fprintf (status_file, "U: Starting point: rate %f, index %li\n", 
             *best_error_rate, (long) *best_k_index);
    fflush (status_file);
}

/*
** Now go through each variable (row) in turn, finding the error rates when
** that variable is "deleted." Take as many steps as there are in "*max_steps,"
** or go forever if that's negative.
*/
number_of_steps = 0L;
while (*max_steps < 0 || number_of_steps < *max_steps){

	best_error_rate_after_deletion = .1;

	for (i = 0; i < train->nrow; i++) 
	{
/* Skip if already processed */
            if ((*backward && *SUB (c, 0, i) == 0.0) || (!(*backward) && *SUB (c, 0, i) == 1.0))
            {
                if (verbose > 1) fprintf (status_file, "S: Skipping %li; already %s\n", i, action_ed);
                    continue;
            }
            if (*backward)
                *SUB (c, 0, i) = 0.0;                  /* Toggle it           */
            else
                *SUB (c, 0, i) = 1.0;
/* Get error rates */
            do_nn (action, first_time, train, train, c, k_vec, *number_of_ks,
                 *theyre_the_same, *number_of_classes, cost, prior,
                    error_rates, misclass_mat, dont_return_classifications,
                    (INT_OR_LONG *) 0, verbose, status_file);
            if (*backward)
                *SUB (c, 0, i) = 1.0;                  /* Put it back         */
            else
                *SUB (c, 0, i) = 0.0;
    
            best_error_rate_this_set = 1.1;
            for (k_ctr = 0; k_ctr < *number_of_ks; k_ctr++)
            {
                if (error_rates[k_ctr] <= best_error_rate_this_set) { /* STRICT */
                    best_error_rate_this_set  = error_rates[k_ctr];
                    best_k_index_this_set     = k_ctr;
                 } /* end "if this is the best rate of this set so far */
            } /* end "for" loop through error rates */

            if (verbose > 1) {
                fprintf (status_file, 
"U: About to compare best this set %f to best after %s %f\n",
best_error_rate_this_set, action_ing, best_error_rate_after_deletion);
                fflush (status_file);
            }
            if (best_error_rate_this_set <= best_error_rate_after_deletion) /* STRICT */
            {
                if (verbose > 1) {
                    fprintf (status_file, "U: After %s %li, best this set is now %f\n", 
                        action_ing, i, best_error_rate_this_set);
                    fflush (status_file);
                }
                for (k_ctr = 0; k_ctr < *number_of_ks; k_ctr++) {
                    best_error_rates_after_deletion[k_ctr] = error_rates[k_ctr];
                }
	 
                best_error_rate_after_deletion = best_error_rate_this_set;
                best_k_index_after_deletion    = best_k_index_this_set;
                best_variable_to_delete        = i;
            }

        } /* end "for" loop through variables */

	if (verbose > 1)
		fprintf (status_file, "U: Big Loop done, winner is %lu with rate %f\n",
		best_variable_to_delete, best_error_rate_after_deletion);

	if (best_error_rate_after_deletion <= *best_error_rate) /* STRICT */
	{
		if (*backward)
		    *SUB (c, 0, best_variable_to_delete) = 0.0;
        else
		    *SUB (c, 0, best_variable_to_delete) = 1.0;
		*best_error_rate = best_error_rate_after_deletion;
		*best_k_index    = best_k_index_after_deletion;
		for (k_ctr = 0; k_ctr < *number_of_ks; k_ctr++)
				best_error_rates_unscaled[k_ctr] = best_error_rates_after_deletion[k_ctr];
		if (verbose > 0)
			fprintf (status_file, "U: We do best (rate %f) after %s variable %li\n",
			best_error_rate_after_deletion, action_ing, best_variable_to_delete);
	}
	else
	{
		if (verbose > 0)
			fprintf (status_file, "U: No %s helps; best_error_rate is %f\n", action_ing, *best_error_rate);
		break;
	}

    number_of_steps++;
} /* end "while (forever)" loop */


/*
****************************************************************************
*/
/*
** At this stage we've done everything we need to with the unscaled variables.
** If we haven't been asked to scale, we're done. Best error_rate, best_k_index,
** and c are properly set. So we can go home. Do the cleanup first (whether
** we're quitting or not, since if we're scaling we'll need to re-compute all 
** the distances anyway.)
*/
do_nn (DO_NN_QUIT, first_time, train, train, c, k_vec, *number_of_ks,
			   *theyre_the_same, *number_of_classes, cost, prior,
                error_rates, misclass_mat, dont_return_classifications,
				(INT_OR_LONG *) 0, verbose, status_file);

if (*scaled == DONT_SCALE) {
	if (verbose > 0) fclose (status_file);
	*status = 0L;
    return ( (void *) 0);
}


/*
** Otherwise, save the winning error rate, k_index, and c matrix for later comparison.
*/

best_rate_unscaled    = *best_error_rate;
best_k_index_unscaled = *best_k_index;
c_unscaled = make_matrix (1L, train->nrow, "Unscaled Weights", REGULAR, TRUE);
matrix_copy (c_unscaled, c);

/* Reset the values in C to be 1's, regardless of forward/backward, for scaling purposes. */

for (i = 0; i < c->ncol; i++)
	*SUB (c, 0L, i) = 1.0;

/*
** Scale the train and test matrices. No need to center.
** Set the first element of "c" to be 0 so the classes don't get re-scaled.
*/
*SUB (c, 0L, 0L) = 0.0;
if (verbose > 0)
	fprintf (status_file, "About to scale (thread 2); *scaled is %li!\n", 
                              (long) *scaled);

if (*scaled == USE_SDS)
	scale_matrix_rows (train, FALSE, TRUE, c, COMPUTE_SCALINGS, (double *) 0, col_sds);
else {
	if (verbose > 0) fprintf (status_file, "Calling MAD thingy!\n");
	scale_matrix_rows_with_mad (train, c, col_sds);
}

/* Reset the first element. If we're going "forward," set everything else to 0. */
*SUB (c, 0L, 0L) = 1.0;
if (!(*backward))
    for (i = 1; i < c->ncol; i++)
        *SUB (c, 0L, i) = 0.0;

if (*status == 2)
	action = DO_NN_INITIALIZE_BIG;
else
	action = DO_NN_INITIALIZE;

do_nn (action, first_time, train, train, c, k_vec, *number_of_ks,  *theyre_the_same, *number_of_classes,
           cost, prior, error_rates, misclass_mat, dont_return_classifications,
		   (INT_OR_LONG *) 0, verbose, status_file);

action = DO_NN_COMPUTE;
/*
** Find the smallest error rate and the k it belongs to.
*/
*best_error_rate = 1.1;
for (k_ctr = 0; k_ctr < *number_of_ks; k_ctr++)
{
	*best_error_rates_scaled = error_rates[k_ctr];
	if (error_rates[k_ctr] < *best_error_rate) {
		*best_error_rate  = error_rates[k_ctr];
		*best_k_index     = k_ctr;
	}
}

if (verbose > 0)
	fprintf (status_file, "S: Starting point: rate %f, index %li\n", 
                 *best_error_rate, (long) *best_k_index);

/*
** Now go through each variable (row) in turn, finding the error rates when
** that variable is 'deleted.'
*/

number_of_steps = 0L;
while (*max_steps < 0 || number_of_steps < *max_steps) {

	best_error_rate_after_deletion = 1.1;

	for (i = 0; i < train->nrow; i++) {

/* Skip if deleted already */
		if ((*backward && *SUB (c, 0, i) == 0.0) || (!(*backward) && *SUB (c, 0, i) == 1.0))	{
			if (verbose > 1) fprintf (status_file, "S: Skipping %li; already deleted\n", i);
			    continue;
			}

		if (*backward)
		    *SUB (c, 0, i) = 0.0;                                  /* Toggle it           */
		else
		    *SUB (c, 0, i) = 1.0;
		do_nn (action, first_time, train, train, c, k_vec, *number_of_ks,   /* Get error rates     */
			   *theyre_the_same, *number_of_classes, cost, prior,
                error_rates, misclass_mat, dont_return_classifications,
				(INT_OR_LONG *) 0, verbose, status_file);
		if (*backward)
		    *SUB (c, 0, i) = 1.0;                                  /* Put it back         */
		else
			*SUB (c, 0, i) = 0.0;

		best_error_rate_this_set = 1.1;
		for (k_ctr = 0; k_ctr < *number_of_ks; k_ctr++)
		{
			if (error_rates[k_ctr] <= best_error_rate_this_set) { /* STRICT */
				best_error_rate_this_set  = error_rates[k_ctr];
				best_k_index_this_set     = k_ctr;
			} /* end "if this is the best rate of this set so far */
		} /* end "for" loop through error rates */

		if (verbose > 2)
			fprintf (status_file, "S: About to compare best this set %f to best after %s %f\n",
				best_error_rate_this_set, action_ing, best_error_rate_after_deletion);
		if (best_error_rate_this_set <= best_error_rate_after_deletion) /* STRICT */
		{
			if (verbose > 1)
				fprintf (status_file, "S: After %s %li, best this set is now %f\n", action_ing, i, best_error_rate_this_set);
			for (k_ctr = 0; k_ctr < *number_of_ks; k_ctr++) {
				best_error_rates_after_deletion[k_ctr] = error_rates[k_ctr];
			}
			best_error_rate_after_deletion = best_error_rate_this_set;
			best_k_index_after_deletion    = best_k_index_this_set;
			best_variable_to_delete        = i;
		}

	} /* end "for" loop through variables */

	if (verbose > 1)
		fprintf (status_file, "S: Big Loop done, winner is %lu with rate %f\n",
			best_variable_to_delete, best_error_rate_after_deletion);

	if (best_error_rate_after_deletion <= *best_error_rate) /* STRICT */
	{
		if (*backward)
		    *SUB (c, 0, best_variable_to_delete) = 0.0;
		else
		    *SUB (c, 0, best_variable_to_delete) = 1.0;
		*best_error_rate = best_error_rate_after_deletion;
		*best_k_index    = best_k_index_after_deletion;
		for (k_ctr = 0; k_ctr < *number_of_ks; k_ctr++)
			best_error_rates_scaled[k_ctr] = best_error_rates_after_deletion[k_ctr];
		if (verbose > 0)
			fprintf (status_file, "S: We do best (rate %f) after %s variable %li\n",
			best_error_rate_after_deletion, action_ing, best_variable_to_delete);
	}
	else
	{
		if (verbose > 0)
			fprintf (status_file, "S: No %s helps; best_error_rate is %f\n", action_ing, *best_error_rate);
		break;
	}

    number_of_steps ++;
} /* end "while (forever)" loop */


/*
** if scaled is the winner, the error rate, k_index and "c" are already set. Otherwise
** we need to re-set them back to the unscaled values.
*/
if (best_rate_unscaled <= *best_error_rate) /* PREFER UNSCALED */
{
	if (verbose > 0)
		fprintf (status_file, "\n***\nUnscaled wins!\n***\n");
	if (*return_all_rates)
		for (k_ctr = 0; k_ctr < *number_of_ks; k_ctr++)
			best_error_rate[k_ctr] = best_error_rates_unscaled[k_ctr];
	else
		*best_error_rate = best_rate_unscaled;
	*best_k_index    = best_k_index_unscaled;
	*scaled          = 0;
	matrix_copy (c, c_unscaled);
}
else
{
	*scaled          = 1;
	if (*return_all_rates) {
		for (k_ctr = 0; k_ctr < *number_of_ks; k_ctr++) {
			fprintf (status_file, "Copying error rate %li; it's %f\n", k_ctr,
				best_error_rates_scaled[k_ctr]);
			best_error_rate[k_ctr] = best_error_rates_scaled[k_ctr];
		}
	}
	if (verbose > 0)
		fprintf (status_file, "\n***\nScaled wins!\n***\n");
}

do_nn (DO_NN_QUIT, first_time, train, train, c, k_vec, *number_of_ks,
			   *theyre_the_same, *number_of_classes, cost, prior,
                error_rates, misclass_mat, dont_return_classifications,
				(INT_OR_LONG *) 0, verbose, status_file);

free (error_rates);
free (best_error_rates_scaled);
free (best_error_rates_unscaled);
free (c_unscaled->data);

*status = 0L;
if (verbose > 0) fclose (status_file);
return (void *) 0;
}
