/*
**			     utils.c
**
**	 This holds some utilities to allocate memory, and other stuff.
**
*/
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <malloc.h>
#include <errno.h>
#include "utils.h"
/* Required for Microsoft? */
#include <stddef.h>

extern void dsort_ (double *vector, long *length, long *flag);

#ifdef USE_S_ALLOC
extern char *S_alloc();
void do_nothing();
#define calloc S_alloc
#define free do_nothing
#endif

#define TRUE  1
#define FALSE 0

/*============================ count_lines ===========================*/
int count_lines (FILE *in_file, long starting_point, unsigned long *line_count)
{
/*
** This counts how many lines there are in a file, starting at "starting_
** point, by reading all the way through it and counting new-line characters.
** How handy.
*/
char in_buffer[100];	/* We read up to 100 characters at a time. */
int i;			/* Indexes in_buffer			   */
long char_count;	/* How many did we get?			   */
long pointer_at_input;	/* Where were we at the start?		   */

*line_count = 0L;
char_count = 1;	       /* set this so as to enter the loop the first time */

/* Save the point where the file pointer is now. */
pointer_at_input = ftell (in_file);

/* Now position to "starting_point." */
fseek (in_file, starting_point, 0);

/* Loop: read up to 100 characters, count new lines, resume if not EOF */
while (char_count > 0)
{
    char_count = fread (in_buffer, 1, 100, in_file);
    for (i = 0; i < char_count; i++)
	if (in_buffer[i] == '\n')
	    (*line_count)++;
}

/* Reset the file, as a courtesy. */
fseek (in_file, pointer_at_input, 0);

return (0);

}

/*======================  alloc_some_double_pointers  =====================*/
int alloc_some_double_pointers (double ***my_array, unsigned long how_many)
{
/* Allocate "how_many" double pointers in memory. */
if ( (*my_array = (double **) calloc (how_many, sizeof (double *))) == NULL)
{
    fprintf (stderr, "Unable to alloc %lu pointers-to-double\n", how_many);
    return (-1);
}
return (0);
}

/*=========================  alloc_some_doubles	 =====================*/
int alloc_some_doubles (double **my_array, unsigned long how_many)
{
/* Allocate "how_many" doubles in memory. */
if ( (*my_array = (double *) calloc (how_many,  sizeof (double))) == NULL)
{
    fprintf (stderr, "Unable to alloc %lu doubles, error %i\n", how_many, errno);
    return (-1);
}
return (0);
}

/*=========================  alloc_some_floats	=====================*/
int alloc_some_floats (float **my_array, unsigned long how_many)
{
/* Allocate "how_many" floats in memory. */
if ( (*my_array = (float *) calloc (how_many, sizeof (float))) == NULL)
{
    fprintf (stderr, "Unable to alloc %lu floats\n", how_many);
    return (-1);
}
return (0);
}

/*=========================  alloc_some_long_ptrs  =====================*/
int alloc_some_long_ptrs (long ***my_array, unsigned long how_many)
{
/* Allocate "how_many" long ptrs in memory. */
if ( (*my_array = (long **) calloc
	  (how_many, sizeof (long *))) == NULL)
{
    fprintf (stderr, "Unable to alloc %li long ptrs\n", how_many);
    return (-1);
}
return (0);
}

/*=========================  alloc_some_longs  =====================*/
int alloc_some_longs (long **my_array, unsigned long how_many)

{
/* Microsoft */
/* size_t my_size = (size_t) how_many * sizeof (long); */
/* Allocate "how_many" longs in memory. */
if ( (*my_array = (long *) calloc (how_many, sizeof (long))) == NULL)

/* 	  (how_many,  (unsigned) sizeof (long))) == NULL) */
{
    fprintf (stderr, "Unable to alloc %lu longs\n", how_many);
    return (-1);
}
return (0);
}
/*=========================  alloc_some_u_longs	 =====================*/
int alloc_some_u_longs (unsigned long **my_array, unsigned long how_many)
{
/* Allocate "how_many" unsigned longs in memory. */
if ( (*my_array = (unsigned long *) calloc
	  (how_many, sizeof (unsigned long))) == NULL)
{
    fprintf (stderr, "Unable to alloc %lu u_longs\n", how_many);
    return (-1);
}
return (0);
}
/*=========================  alloc_some_ints  ========================*/
int alloc_some_ints (int **my_array, unsigned long how_many)
{
/* Allocate "how_many" ints in memory. */
if ( (*my_array = (int *) calloc (how_many, sizeof (int))) == NULL)
{
    fprintf (stderr, "Unable to alloc %lu ints\n", how_many);
    return (-1);
}
return (0);
}
/*=========================  alloc_some_char_ptrs  ========================*/
int alloc_some_char_ptrs (char ***my_array, unsigned long how_many)
{
/* Allocate "how_many" char ptrs in memory. */
if ( (*my_array = (char **) calloc (how_many,  sizeof (char *))) == NULL)
{
    fprintf (stderr, "Unable to alloc %lu char_ptrs\n", how_many);
    return (-1);
}
return (0);
}
/*=========================  alloc_some_chars  ========================*/
int alloc_some_chars (char **my_array, unsigned long how_many)
{
/* Allocate "how_many" char in memory. */
if ( (*my_array = (char *) calloc (how_many, sizeof (char))) == NULL)
{
    fprintf (stderr, "Unable to alloc %lu chars\n", how_many);
    return (-1);
}
return (0);
}

/*=========================  provisional_means	===========================*/
int provisional_means (double *new_vector, long vector_length, long status,
		       double *mean, double *var)
{
/*
** In this function, we compute component-wise means and variances of sets
** of vectors by the method of provisional means.  The first time through,
** and indeed every time, "vector_length" should hold the number of elements
** in the vector. The calling function also supplies pointers to vectors
** of results. This may be keeping track of several "groups" of vectors at
** once (since the only information it needs to keep is the number of vectors
** supplied).  "Status" can be QUIT (meaning both quit and be prepared to
** initialize next time through), END_GROUP (meaning compute the results for
** this group), INCREMENT (meaning all groups are though, so increment the
** count) or none of these (process one vector for one group).
*/
static int initialized = FALSE;
static long number_count = 0L;	/* How many vectors are there?		    */

double previous_prov_mean;	/* Don't you love long variable names?	    */
long i;

/*
** If it's quitting time, return the provisional mean. Since prov_var
** holds the numerator, divide by (n-1). Print. If there are no numbers,
** set mean =0, var = 0.
*/

if (!initialized)
{
    initialized = TRUE;
    if (vector_length < 1)
		return (ILLEGAL_LENGTH);
    number_count = 1L;
    for (i = 0L; i < vector_length; i ++)
    {
		mean[i] = 0.0;
		if (var != (double *) NULL)
			var[i] = 0.0;
    }
    if (status != INCREMENT)
		status = DONT_QUIT;
}
if (status == QUIT || status == END_GROUP)
{

    if (var != (double *) NULL)
		for (i = 0L; i < vector_length; i++)
		{
/*
** Each time through, number_count is incremented at the bottom of the loop. That means
** when it's finally time to quit it's one too big. We subtract one to get the correct
** number_count, and one more to get the inbiased estimate.
*/
			var[i] /= (number_count - 2);
		}
    if (status == QUIT)
    {
		number_count = 1L;
		initialized = FALSE;
    }
    return (0);
}
/*
** Add 1 to the count if necessary; compute new provisional mean and variance.
*/
for (i = 0L; i < vector_length; i++)
{
    previous_prov_mean = mean[i];
    mean[i] += (new_vector[i] - mean[i]) / number_count;
    if (var != (double *) NULL)
	var[i] += (new_vector[i] - previous_prov_mean)
			   * (new_vector[i] - mean[i]);
}
if (status == INCREMENT)
    number_count++;

return (0);

} /* end "provisional_means" */

/*=========================  compute_mad  ===========================*/

double compute_mad (double *in_vector, long vector_length, long *status)
{
/*
** Compute the median absolute deviation of in_vector in several steps. First sort
** and compute the median; then compute the absolute deviations from the median, sort
** again, and compute the median of those deviations. We don't know in advance whether
** vector_length will be odd or even, so we take the mean of the two central elements
** (which will be the same if vector_length is odd).
*/
double *sorted_data, median, mad;
long i;
long sort_ascending = 1L;
int center_index_1, center_index_2;

*status = 0L;

center_index_1 = (int) floor ( ((double) vector_length + 1.0)/ 2.0) - 1;
center_index_2 = (int) ceil  ( ((double) vector_length + 1.0)/ 2.0) - 1;


if (alloc_some_doubles (&sorted_data, (unsigned) vector_length))
{
	*status = -1L;
	return (0.0);
}

/*
** Load the data into "sorted_data"
*/
for (i = 0; i < vector_length; i++)
    sorted_data[i] = in_vector[i];


/* Sort once  in preparation for finding the median. */

dsort_ (sorted_data, &vector_length, &sort_ascending);
median = (sorted_data[center_index_1] + sorted_data[center_index_2]) / 2.0;

for (i = 0; i < vector_length; i++)
    sorted_data[i] = fabs (sorted_data[i] - median);

/* Sort a second time. The median of these is the MAD */

dsort_ (sorted_data, &vector_length, &sort_ascending);
mad = (sorted_data[center_index_1] + sorted_data[center_index_2]) / 2.0;

free (sorted_data);

return (mad);

} /* end "compute_mad" */

#ifdef USE_S_ALLOC
void do_nothing (void *ptr)
{
}
#endif
