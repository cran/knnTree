/*
**		     File matrix.c:
**
**   Fun things to do with matrices.
**
*/
#include <stdio.h>
#include <math.h>
/* #include <unistd.h> */
#include <malloc.h>
#include <string.h>
#include <ctype.h>
#include "utils.h"
#include "matrix.h"

#ifdef USE_S_ALLOC
extern char *S_alloc();
extern void do_nothing();
#define calloc S_alloc
#define free do_nothing
#endif

#define TRUE  1
#define FALSE 0
#define CHUNK 1300

char error_text[100];
int bose = 0;

double gimme (MATRIX *in_mat, long i, long j)
{
    return (in_mat->data[(i) * (in_mat->ncol) + (j)]);

}
int g2 (MATRIX *in_mat, long i, long j)
{
    return (i * (in_mat->ncol) + (j));
}

/*===========================	dot  ===============================*/
double dot (double *first, double *second,
	    unsigned long first_stride, unsigned long second_stride,
	    unsigned long length)
/*
**  Perform a dot product for length "length" with the given starting
**  points and strides.
*/
{
unsigned long i;    /* Loop counter	     */
double result;	    /* Result of dot product */

/*
** Set result to zero; then go through the loop "length" times, each
** time moving "first" and "second" up by their respective strides.
*/
result = 0.0;
for (i = 0; i < length; i++, first += first_stride, second += second_stride)
{
    result += *first * *second;
}
return (result);

} /* end "dot" */

/*=========================  matdot  ===============================*/
double matdot (MATRIX *first,  int dim_1, unsigned long item_1,
	       MATRIX *second, int dim_2, unsigned long item_2)
{
/*
** This performs a general dot product on rows or columns of matrices.
** For example, matdot (a, COLUMN, 5, b, ROW, 2) produces the dot product
** of the fifth column of a with the second row of B. Only matrices
** with REGULAR storage are eligible (use set_up_vector otherwise).
*/

unsigned long first_stride, second_stride;	 /* Strides	   */
unsigned long first_start, second_start;	 /* Starting point */
unsigned long how_many;				 /* Length of "dot"*/

/* Check for "REGULAR" storage" */
if (first->sym_storage != REGULAR
||  second->sym_storage != REGULAR)
{
    matrix_error ("Matdot: matrices don't have regular storage\n");
    return (NOT_YET_INSTALLED);
}

/* Set the starting point and stride for matrix one, and the length. */
if (dim_1 == ROW)
{
    first_start	 = first->ncol * item_1;
    first_stride = 1;
    how_many	 = first->ncol;
}
else
    if (dim_1 == COLUMN)
    {
	first_start  = item_1;
	first_stride = first->ncol;
	how_many     = first->nrow;
    }
    else
    {
        sprintf (error_text, "Madtot: weird dim_1 (%i) supplied\n", dim_1);
        matrix_error (error_text);
	return (CASE_ERROR);
    }

if (dim_2 != COLUMN && dim_2 != ROW)
{
    sprintf (error_text, "Madtot: weird dim_2 (%i) supplied\n", dim_1);
    matrix_error (error_text);
    return (CASE_ERROR);
}

/* Set starting point and stride for matrix two. */
second_stride = (dim_2 == ROW)?	   1   : second->ncol;
second_start  = (dim_2 == ROW)? second->ncol * item_2 : item_2;

/* Return the appropriate dot product. */
return (dot (first->data + first_start, second->data + second_start,
	     first_stride, second_stride, how_many));
} /* end "matdot" */

/*======================   matrix_multiply ============================*/
int matrix_multiply (MATRIX *a, MATRIX *b, MATRIX *result,
		     char transposes)
{
/*
** This routine multiplies matrix a by matrix b, possibly transposing one
** or both, and puts the result in "result." It is the responsibility
** of the caller to ensure that result has the correct size and that
** its data pointer is initialized.
**
** If all three matrices use REGULAR storage, this routine calls
** "matrix_multiply_all_regular", which takes advantage of the fact
** that dot products are easy to construct and presumably runs quickly.
** Furthermore, "result" must be REGULAR anyway. If either or both of
** a and b is not REGULAR, then the job is a little trickier, and it
** gets done here. Each element of the result matrix will still be a dot
** product, but of two vectors that need to be set up in advance, and
** padded with zeros if the matrix is triangular, or with its own
** values if the matrix is symmetric.
*/
unsigned long i, j;	       /* Loop counters		      */
double *a_vector, *b_vector;   /* Two vectors to be "dotted." */

/* Check that the result has data allocated. */
if (result == (MATRIX *) NULL
||  result->data == (double *) NULL)
{
    matrix_error ("Matrix_multiply; result empty or has no data\n");
    return (NO_DATA);
}

/* Call "matrix_multiply_all_regular" if all matrices are regular. */
if (a->sym_storage == REGULAR
&&  b->sym_storage == REGULAR
&&  result->sym_storage == REGULAR)
    return (matrix_multiply_all_regular (a, b, result, transposes));

/* If result isn't REGULAR, forget it. */
if (result->sym_storage != REGULAR)
{
    matrix_error ("Matrix_multiply; result should have regular storage\n");
    return (NOT_YET_INSTALLED);
}

/*
** For each value of tranposes, first ensure that the two matrices are
** conformable. Then allocate space for the two vectors. For each entry
** in result, set up appropriate vectors (for example, the fifth column
** in a lower-triangular matrix will need to have four zeros prepended),
** using the "set_up_vector" function; and dot them.
*/
switch (transposes)
{
    case NO_TRANSPOSES:
	if (a->ncol != b->nrow
        || a->nrow != result->nrow
        || b->ncol != result->ncol)
        {
            matrix_error ("Mult, No tranposes, non-conformable matrices\n");
	    return (NON_CONFORMABLE);
        }
	if (alloc_some_doubles (&a_vector, a->ncol))
        {
            matrix_error ("Mult, no tranposes, couldn't allocate a\n");
	    return (ALLOCATION_ERROR);
        }
	if (alloc_some_doubles (&b_vector, b->nrow))
        {
            matrix_error ("Mult, no tranposes, couldn't allocate b\n");
            free (a_vector);
	    return (ALLOCATION_ERROR);
        }
	for (i = 0; i < a->nrow; i++)
	    for (j = 0; j < b->ncol; j++)
	    {
		set_up_vector (a, ROW, i, a_vector);
		set_up_vector (b, COLUMN, j, b_vector);
		*SUB (result, i, j) = dot (a_vector, b_vector, 1, 1, a->ncol);
	    }
	break;
    case TRANSPOSE_FIRST:
	if (a->nrow != b->nrow
        ||  a->ncol != result->nrow
        ||  b->ncol != result->ncol)
        {
            matrix_error ("Mult, tranpose first, non-conformable matrices\n");
	    return (NON_CONFORMABLE);
        }
	if (alloc_some_doubles (&a_vector, a->nrow))
        {
            matrix_error ("Mult, tranpose first, couldn't allocate a\n");
	    return (ALLOCATION_ERROR);
        }
	if (alloc_some_doubles (&b_vector, b->nrow))
        {
            free (a_vector);
            matrix_error ("Mult, tranpose first, couldn't allocate b\n");
	    return (ALLOCATION_ERROR);
        }
	for (i = 0; i < a->ncol; i++)
	    for (j = 0; j < b->ncol; j++)
	    {
		set_up_vector (a, COLUMN, i, a_vector);
		set_up_vector (b, COLUMN, j, b_vector);
		*SUB (result, i, j) = dot (a_vector, b_vector, 1, 1, a->nrow);
	    }
	break;
    case TRANSPOSE_SECOND:
	if (a->ncol != b->ncol
        ||  a->nrow != result->nrow
        ||  b->nrow != result->ncol)
        {
            matrix_error ("Mult, tranpose second, non-conformable matrices\n");
	    return (NON_CONFORMABLE);
        }
	if (alloc_some_doubles (&a_vector, a->ncol))
        {
            matrix_error ("Mult, tranpose second, couldn't allocate a\n");
	    return (ALLOCATION_ERROR);
        }
	if (alloc_some_doubles (&b_vector, b->ncol))
        {
            free (a_vector);
            matrix_error ("Mult, tranpose second, couldn't allocate b\n");
	    return (ALLOCATION_ERROR);
        }
	for (i = 0; i < a->nrow; i++)
	    for (j = 0; j < b->nrow; j++)
	    {
		set_up_vector (a, ROW, i, a_vector);
		set_up_vector (b, ROW, j, b_vector);
		*SUB (result, i, j) = dot (a_vector, b_vector, 1, 1, a->ncol);
	    }
	break;
    case TRANSPOSE_BOTH:
	if (a->nrow != b->ncol
        ||  a->ncol != result->nrow
        ||  b->nrow != result->ncol)
        {
            matrix_error ("Mult, tranpose both, non-conformable matrices\n");
	    return (NON_CONFORMABLE);
        }
	if (alloc_some_doubles (&a_vector, a->nrow))
        {
            matrix_error ("Mult, tranpose both, couldn't allocate a\n");
	    return (ALLOCATION_ERROR);
        }
	if (alloc_some_doubles (&b_vector, b->ncol))
        {
            free (a_vector);
            matrix_error ("Mult, tranpose both, couldn't allocate b\n");
	    return (ALLOCATION_ERROR);
        }
	for (i = 0; i < a->ncol; i++)
	    for (j = 0; j < b->nrow; j++)
	    {
		set_up_vector (a, COLUMN, i, a_vector);
		set_up_vector (b, ROW, j, b_vector);
		*SUB (result, i, j) = dot (a_vector, b_vector, 1, 1, a->nrow);
	    }
	break;
}

/* Get rid of the vectors and split. */
free (a_vector);
free (b_vector);
return (0);

} /* end "matrix_multiply" */

/*============================= set_up_vector ==========================*/
int set_up_vector (MATRIX *in, int dim, unsigned long item, double *result)
{
/*
** This matrix produces a particular vector from a matrix; for example,
** set_up_vector (a, ROW, 5, result) would copy the fifth row of a into
** result. The trick is that the matrix might be symmetric or triangular,
** in which case we have to recognize that and take action.
*/
unsigned long i;		  /* Loop counter.  */
unsigned long how_many;		  /* Length to copy */
unsigned long starting_point;	  /* Starting point */
/*
** Figure out how many items to place in result.
** Point data at first appropriate item.
*/
how_many = (dim == ROW? in->ncol: in->nrow);
starting_point = (dim == ROW? item * in->ncol: item);

/* For a REGULAR matrix, just copy elements one by one. */
if (in->sym_storage == REGULAR)
{
    for (i = 0; i < how_many;  i++)
	if (dim == ROW)
	    *(result + i) = *SUB (in, item, i);
	else
	    *(result + i) = *SUB (in, i, item);
    return (0);
}
/*
** For triangular matrices, copy zeros too. If they want a row of a lower
** triangular, put zeros at the end; for a column, put 'em up front.
*/
if (in->sym_storage == LOWER_TRIANGULAR)
{
    for (i = 0; i < how_many; i ++)
	if (dim == ROW)
	    if (i <= item)
		*(result + i) = *SYM_SUB (in, item, i);
	    else
		*(result + i) = 0.0;
	else
	    if (i >= item)
		*(result + i) = *SYM_SUB (in, i, item);
	    else
		*(result + i) = 0.0;
    return (0);
}
if (in->sym_storage == UPPER_TRIANGULAR)
{
    for (i = 0; i < how_many; i ++)
	if (dim == ROW)
	    if (i >= item)
		*(result + i) = *UPPER_SUB (in, item, i);
	    else
		*(result + i) = 0.0;
	else
	    if (i <= item)
		*(result + i) = *UPPER_SUB (in, i, item);
	    else
		*(result + i) = 0.0;
    return (0);
}
/* For symmetric matrices, macro "SYM_SUB" takes case of this. */
if (in->sym_storage == SYMMETRIC)
{
    for (i = 0; i < how_many; i ++)
	*(result + i) = *SYM_SUB (in, item, i);
    return (0);
}

sprintf (error_text, "Setup: weird storage (%i) supplied\n", in->sym_storage);
matrix_error (error_text);
return (CASE_ERROR);

} /* end "set_up_vector" */

/*=====================	 matrix_multiply_all_regular  ================*/
int matrix_multiply_all_regular (MATRIX *a, MATRIX *b, MATRIX *result,
				  char transposes)
{
/*
** This routines multiplies regular matrices. For each element in "result"
** it performs the appropriate dot product using "matdot."
*/
unsigned long i,j; /* Loop counters */
switch (transposes)
{
    case NO_TRANSPOSES:
	if (a->ncol != b->nrow
        ||  a->nrow != result->nrow
        ||  b->ncol != result->ncol)
        {
            matrix_error ("Mult/Reg: No transposes, non-conformable\n");
	    return (NON_CONFORMABLE);
        }
	for (i = 0; i < a->nrow; i++)
	    for (j = 0; j < b->ncol; j++)
            {
                if (bose == 4)
                printf ("Filling result row %ld col %ld\n", i, j);
		*SUB (result, i, j) = matdot (a, ROW, i, b, COLUMN, j);
            }
	break;
    case TRANSPOSE_FIRST:
	if (a->nrow != b->nrow
        ||  a->ncol != result->nrow
        ||  b->ncol != result->ncol)
        {
            matrix_error ("Mult/Reg: Transpose first, non-conformable\n");
	    return (NON_CONFORMABLE);
        }
	for (i = 0; i < a->ncol; i++)
	    for (j = 0; j < b->ncol; j++)
		*SUB (result, i, j) = matdot (a, COLUMN, i, b, COLUMN, j);
	break;
    case TRANSPOSE_SECOND:
	if (a->ncol != b->ncol
        ||  a->nrow != result->nrow
        ||  b->nrow != result->ncol)
        {
            matrix_error ("Mult/Reg: Transpose second, non-conformable\n");
	    return (NON_CONFORMABLE);
        }
	for (i = 0; i < a->nrow; i++)
	    for (j = 0; j < b->nrow; j++)
		*SUB (result, i, j) = matdot (a, ROW, i, b, ROW, j);
	break;
    case TRANSPOSE_BOTH:
	if (a->nrow != b->ncol
        ||  a->ncol != result->nrow
        ||  b->nrow != result->ncol)
        {
            matrix_error ("Mult/Reg: Transpose both, non-conformable\n");
	    return (NON_CONFORMABLE);
        }
	for (i = 0; i < a->ncol; i++)
	    for (j = 0; j < b->nrow; j++)
		*SUB (result, i, j) = matdot (a, COLUMN, i, b, ROW, j);
	break;
    default:
        sprintf (error_text, "Mult/reg, weird transpose (%i) found\n",
                 transposes);
        matrix_error (error_text);
	return (TRANPOSE_ERROR);
} /* end switch */

return (0);

} /* end "matrix_multiply_all_regular" */

/*============================	scalar_multiply	 =====================*/
int scalar_multiply (MATRIX *a, MATRIX *result, double scalar)
{
/*
**  Multiply a matrix by a scalar; put result in "result" or back into
**  the original matrix.
*/
unsigned long how_many, i;   /* The usual */

/* If no result is supplied, operate on the input matrix. */
if (result == (MATRIX *) NULL)
    result = a;

/* Figure out how many you need. Triangles, syymetrics are welcome here. */
how_many = a->sym_storage == REGULAR?
	       a->nrow * a->ncol : a->nrow * (a->nrow + 1) / 2;

/* Multiply them all by the scalar. */
for (i = 0; i < how_many; i++)
    *(result->data + i) = *(a->data + i) * scalar;

/* Have a nice day. */
return (0);

} /* end "scalar_multiply" */

/*============================	matrix_add  =====================*/
int matrix_add (MATRIX *first, MATRIX *second, MATRIX *result, int which)
{
unsigned long i, how_many;

if (first->nrow != second->nrow
||  first->ncol != second->ncol)
{
    matrix_error ("Matrix add, non-conformable matrices\n");
    return (NON_CONFORMABLE);
}
if (first->sym_storage != second->sym_storage)
{
    matrix_error ("Matrix add, unequal storage\n");
    return (NON_CONFORMABLE);
}

if (result == (MATRIX *) NULL)
    result = first;

how_many = first->sym_storage == REGULAR?
	       first->nrow * first->ncol : first->nrow * (first->nrow + 1) / 2;

for (i = 0; i < how_many; i++)
{
    if (which == SUBTRACT)
	result->data[i] = first->data[i] - second->data[i];
    else
	result->data[i] = first->data[i] + second->data[i];
}

return (0);

}


/*============================	matrices_equal	=====================*/
unsigned long  matrices_equal (MATRIX *a, MATRIX *b,
			       double tolerance, double *max_diff)
{
/*
** This function compares two conformable matrices to see if they're
** equal to within "tolerance." This works only on matrices with
** the same storage mode. If "max_diff" is NULL, then the function
** will return the number of the first element where the difference
** is bigger than "tolerance"; if "max_diff" is not NULL, the function
** will go through the whole comparison, and return the number of the
** element where the difference is largest (while putting that absolute
** difference into "max_diff").
*/
double *a_data, *b_data; /* Point to elements being compared */
unsigned long i;	 /* Loop counter		     */
unsigned long how_many;	 /* How many to compare?	     */
unsigned long diff_i;	 /* Which element is different?	     */
double diff;		 /* How big is the difference?	     */

/* If sizes are wrong, that's an error. */
if (a->nrow != b->nrow
||  a->ncol != b->ncol)
{
    matrix_error ("Matrices equal, non-conformable matrices\n");
    return (NON_CONFORMABLE);
}

/* We don't yet handle unequal storage modes. */
if (a->sym_storage != b->sym_storage)
{
    matrix_error ("Matrices equal, unequal storage\n");
    return (NOT_YET_INSTALLED);
}

/* Set pointers to beginning of data. */
a_data = a->data;
b_data = b->data;

/* Count how mnany elements we expect. */
how_many = a->sym_storage == REGULAR?
	       a->nrow * a->ncol : a->nrow * (a->nrow + 1) / 2;

diff_i = 0L;
/* Go into the loop, each time moving up a_data and b_data. */
for (i = 0; i < how_many; i++, a_data ++, b_data ++)
{
/*
** Compute the abolute difference. If it's too big, and max_diff is NULL,
** return the number of this element.
*/
    diff = fabs (*a_data - *b_data);
    if (diff > tolerance)
    {
	if (max_diff == (double *) NULL)
	    return (i);
/* Otherwise, save this number and difference, if the latter is biggest. */
	if (diff > *max_diff)
	{
	    diff_i = i;
	    *max_diff = diff;
	}
    }
}
/*
** Return either the number of the largest difference (which sould be 0),
** or MATRICES_EQUAL.
*/
return (diff_i == 0? MATRICES_EQUAL : diff_i);

} /* end "matrices_equal" */

/*============================	print_matrix  =====================*/
int print_matrix (MATRIX *a, char flags)
{
/* This function prints a matrix, possibly transposed, of any storage type. */

unsigned long i, j;		     /* Loop counters */
unsigned long vertical_count,
	      horizontal_count;	     /* Number of entries vert.'ly & horiz.'ly*/
char output_line[80];		     /* For S format column heads & lines     */
char bunch_o_spaces[30] = "                              ";
char bit[30];
int transpose;
int print_title;
int S_format;
int precision;

if (a == (MATRIX *) NULL)
{
    matrix_error ("Can't print null matrix\n");
    return (FALSE);
}
if (a->sym_storage == CHARACTER)
     return (print_char_matrix ((CHAR_MATRIX *) a, flags));

transpose = 0;
if (TEST_PATTERN (flags, TRANSPOSE_FIRST))
    transpose |= TRANSPOSE_FIRST;
S_format = 0;
if (TEST_PATTERN (flags, S_FORMAT))
    S_format |= S_FORMAT;
print_title = 1;
if (TEST_PATTERN (flags, NO_TITLE))
    print_title = 0;
precision = PRECISION;
if (TEST_PATTERN (flags, HIGH_PRECISION))
    precision = 20;

/* Print name, if desired and if any. */
if (print_title && a->name[0] != '\0')
    printf ("\n%s:\n", a->name);

if (a->data == (double *) 0)
{
    printf ("-- Matrix has no data\n");
    return (0);
}

/* Figure out how many to expect in each direction. */
vertical_count	 = (transpose == TRANSPOSE_FIRST? a->ncol : a->nrow);
horizontal_count = (transpose == TRANSPOSE_FIRST? a->nrow : a->ncol);

if (S_format)
{
    horizontal_count = 1;
    while (horizontal_count <= (transpose == TRANSPOSE_FIRST? a->ncol: a->nrow))
    {
	vertical_count = 1;
	while (vertical_count <= (transpose== TRANSPOSE_FIRST? a->nrow:a->ncol))
	{
	    strcpy (output_line, "     ");
	    for (j = 0; j < PER_LINE
    && (vertical_count + j) <= (transpose == TRANSPOSE_FIRST? a->nrow: a->ncol);
		 j++)
	    {
            sprintf (bit, "%.*s[,%lu]", 8, bunch_o_spaces,vertical_count+j);
		strcat (output_line, bit);
	    }
	    printf ("\n%s\n", output_line);
	    for (i = 0; i < ROWS_PER_PANEL
&& (horizontal_count + i) <= (transpose == TRANSPOSE_FIRST? a->ncol: a->nrow);
		 i++)
	    {
		output_line[0] = '\0';
		sprintf (output_line, "[%lu,]", horizontal_count + i);
		for (j = 0; j < PER_LINE
&& (vertical_count+j) <= (transpose == TRANSPOSE_FIRST? a->nrow: a->ncol);
		j++)
		{
		    if (transpose == TRANSPOSE_FIRST)
			sprintf (bit, "%- *.*e ", precision, precision,
			*SUB (a, vertical_count +j-1, horizontal_count +i-1));
		    else
			sprintf (bit, "%- *.*e ", precision, precision,
			*SUB (a, horizontal_count +i-1, vertical_count +j-1));
		    strcat (output_line, bit);
		}
		printf ("%s\n", output_line);
	     }
	     vertical_count += PER_LINE;
	} /* end "while there are more columns for this panel" */
	horizontal_count += ROWS_PER_PANEL;
    } /* end "while there are more rows" */
} /* end "if S_format" */
else
{
/* Do that loop. */
for (i = 0; i < vertical_count; i++)
{
    for (j = 0; j < horizontal_count; j++)
    {
/* Print a new line if this line is full, unless the EASY_FORMAT bit is on. */
	if ( j > 0 && j % PER_LINE == 0 && !TEST_PATTERN (flags, EASY_FORMAT))
	    printf ("\n");
	switch (a->sym_storage)
	{
	    case REGULAR:
		if (transpose == TRANSPOSE_FIRST)
		    printf ("%- *.*e ", precision, precision, *SUB (a, j, i));
		else
		    printf ("%- *.*e ", precision, precision, *SUB (a, i, j));
		break;
	    case SYMMETRIC:
		printf ("%- *.*e ", precision, precision, *SYM_SUB(a, i, j));
		break;
	    case UPPER_TRIANGULAR:
		if (i <= j)
		    printf ("%- *.*e ", precision, precision,
		       transpose == TRANSPOSE_FIRST ? 0.0 : *UPPER_SUB(a, i,j));
		else
		    printf ("%- *.*e ", precision, precision,
		       transpose == TRANSPOSE_FIRST ? *UPPER_SUB(a, i,j) : 0.0);
		break;
	    case LOWER_TRIANGULAR:
		if (i >= j)
		    printf ("%- *.*e ", precision, precision,
		       transpose == TRANSPOSE_FIRST ? 0.0 : *SYM_SUB(a, i,j));
		else
		    printf ("%- *.*e ", precision, precision,
		       transpose == TRANSPOSE_FIRST ? *SYM_SUB(a, i,j) : 0.0);
		break;
	}
    }
    printf ("\n");
} /* end "for i" loop */
} /* end "else", i.e. if not S-format. */

return (0);
} /* end print_matrix */

/*=========================== print_char_matrix	 ====================*/
int print_char_matrix (CHAR_MATRIX *a, char flags)
{
unsigned long i, j;

for (i = 0L; i < a->nrow; i++)
{
    for (j = 0L; j < a->ncol; j++)
    {
	printf ("%*.*s", (a->string_len + 1), (a->string_len + 1),
			      a->data[(i * a->ncol) + j]);
    }
    printf ("\n");
}

return (0);

}

/*=========================== print_columns  ====================*/
int print_columns (MATRIX *a, MATRIX *b)
{
/*
** Print one-column matrices a and b adjacently.
*/
unsigned long i;
printf ("\n\t%s\t\t%s\n", a->name, b->name);
for (i = 0; i < a->nrow; i++)
{
    printf ("\t%- *e\t\t\t%- *e\n", PRECISION, a->data[i],
				    PRECISION, b->data[i]);
}

return (0);

} /* end "print_columns" */

/*=========================== read_matrix  ====================*/
int read_matrix (MATRIX *mat, FILE *in_file)
{
/*
** Read a matrix from "in_file" and put it into "mat". The file pointed
** to by "in_file" should contain optional leading comments, which have
** a leading (first-column) "#" sign; an optional line saying "Add intercept",
** which directs the function to append a first column of 1's; and then
** the data, stored by row. "Mat" should exist: this function will
** allocate its data and "columns_in" vectors and set the number of rows
** and columns. New exciting added feature: read in output from S, that
** has different rows and things like "[12,]" and stuff.
**
** This function will produce a "REGULAR" matrix only.
*/
/*PROBLEMS: Lines longer than "CHUNK"; maybe tell the function the
**dimensions; blank lines (e.g. at end) */
unsigned long i, j;		 /* Loop counters      */
char in_buffer[CHUNK];		 /* Text from file     */
int add_intercept;		 /* Want an intercept? */
int status;			 /* Function results   */
int this_is_S_format;		 /* File is S format   */
char *in_char;			 /* Holds char read in */
long starting_point;		 /* File starting pt.  */
unsigned long first_column, current_row;
unsigned long columns_per_line = 1L;

mat->sym_storage = REGULAR;
/* By default, there's no intercept. */
add_intercept = FALSE;
/*
** Get the first character of the file, then move back one byte. If that
** first is a comment character, enter the loop to strip out all
** comments.
*/
starting_point = ftell (in_file);
fgets (in_buffer, CHUNK, in_file);
if (in_buffer[0] == '#')
{
    while (in_buffer[0] == '#')
    {
	starting_point = ftell (in_file);
	fgets (in_buffer, CHUNK, in_file);
    }
}

/* If that line tells us to add an intercept, do so....*/
if (memcmp (in_buffer, "Add intercept", 13) == 0)
{
    starting_point = ftell (in_file);
    add_intercept = TRUE;
/* ...and move up the "starting point" appropriately. */
    fgets (in_buffer, CHUNK, in_file);
}
/*
** Now if any character of this line is a "[", we know this is
** an S format file. We will then take appropriate action.
*/
if (strchr (in_buffer, (int) '[') != NULL)
{
    this_is_S_format = TRUE;
    status = get_S_format_dimension (in_file, &(mat->nrow), &(mat->ncol));
    if (status != 0)
    {
        matrix_error ("Print matrix: Weird error in get_S_format_dimension\n");
	return (BAD_FORMAT);
    }
    fseek (in_file, starting_point, SEEK_SET);
}
else
{
    this_is_S_format = FALSE;

/* Set the file pointer back to the beginning of this line. */
    fseek (in_file, starting_point, 0);

/* Count the number of lines (rows) in the file, starting here... */
    status = count_lines (in_file, starting_point, &(mat->nrow));

/* Now count the columns by counting strings of non-spaces in this line. */
    mat->ncol = 0L;
    in_char = in_buffer;
    while (in_char < in_buffer + strlen (in_buffer))
    {
	while (*in_char == ' ') in_char++;
/******	 TEST TEST TEST
	if (in_char >= in_buffer + strlen (in_buffer))
*******	 TEST TEST TEST */
	if (in_char >= (in_buffer + strlen (in_buffer) - 1))
	    break;
	mat->ncol++;
	while (*in_char != ' ') in_char++;
	if (in_char >= in_buffer + strlen (in_buffer))
	    break;
    }
}

/* Whether S format or not, add a column for the intercept if necessary. */
if (add_intercept)
    mat->ncol++;

/* Allocate the "columns_in" vector...*/
/********  DELETED FOR NOW *********
if (alloc_some_ints (&(mat->columns_in), mat->ncol))
{
    fprintf (stderr, "Couldn't allocate %lu doubles for %s: abort\n",
		      mat->ncol, mat->name);
    return (-1);
}
********  DELETED FOR NOW *********/

/* ...and the data vector. */
if (alloc_some_doubles (&(mat->data), mat->nrow * mat->ncol))
{
    sprintf (error_text, "Couldn't allocate %lu doubles for %s: abort\n",
		          mat->nrow * mat->ncol, mat->name);
    matrix_error (error_text);
    return (-1);
}
/* Set up "columns_in" to be all TRUE */
if (mat->columns_in != NULL)
    for (j = 0; j < mat->ncol; j++)
	mat->columns_in[j] = TRUE;
/* For S format data, read a row with column headings, count how many
** there are and note the first one. Then read data rows, and insert the
** data into the appropriate place.
*/
current_row = -1;
if (this_is_S_format)
{
    while (1)
    {
	if (fgets (in_buffer, CHUNK, in_file) == '\0')
	{
	    if (add_intercept)
		for (i = 0; i < mat->nrow; i++)
		    *SUB (mat, i, 0) = 1.0;
	    return (0);
	}
	in_char = strchr (in_buffer, (int) '[');
	if (in_char == NULL)
	    continue;
	if ( *(in_char + 1) == ',')
	{
	    sscanf (in_char + 2, "%lu", &first_column);
	    if (add_intercept == TRUE)
		first_column++;
	    columns_per_line = (unsigned long) 1;
	    current_row = -1;
	    while (1)
	    {
		in_char = strchr (in_char + 1, (int) '[');
		if (in_char == NULL)
		    break;
		columns_per_line ++;
	    }
	    continue;
	}
	else
	{
	    sscanf (in_char + 1, "%lu", &current_row);
	    in_char = strchr (in_buffer, (int) ']') + 1;
	    for (i = 0; i < columns_per_line; i++)
	    {
		sscanf (in_char, " %lf",
			SUB(mat, current_row - 1, first_column + i - 1));
		while (*in_char == ' ') in_char++;
		in_char = strchr (in_char, (int) ' ');
	    }
	}
    }
}
else
/*
** For non-S format data, just stick the data into the matrix.
** If this is the first column, and they want an intercept, deal with that.
*/
{
    for (i = 0; i < mat->nrow; i++)
    {
	for (j = 0; j < mat->ncol; j++)
	{
	    if (add_intercept && j == 0)
		*SUB(mat, i, 0) = 1.0;
	    else
		fscanf (in_file, " %lf", SUB (mat, i, j));
	}
    }
}
/* Later. */
return (0);
} /* end "read_matrix" */

/*======================  read_char_matrix  =============*/
int read_char_matrix (CHAR_MATRIX *mat, FILE *in_file)
{
int status;		 /* Results of int-valued functions. */
char *in_char;		 /* Handy char_ptr		     */
char in_buffer[CHUNK];	 /* Text from file		     */
int long_string;	 /* Encountered a too-long string.   */
unsigned long i, j; int k;		 /* Counters			     */
/*
** This function allocates space for, and reads, a "char" matrix,
** that is, a matrix whose data is a series of strings. The "string_len"
** item tells us the length of the string; we add one to allow for the
** NULL terminator. First, count the number of lines in the file.
*/
status = count_lines (in_file, 0L, &(mat->nrow));
/*
** Now count the number of items on this line; an "item" is a sequence
** of characters that isn't blank or tab. So get the first line.
*/
fgets (in_buffer, CHUNK, in_file);
mat->ncol = 0L;
in_char = in_buffer;
while (in_char < in_buffer + strlen (in_buffer))
{
/* Skip spaces...*/
    while (*in_char == ' ') in_char++;
/* If we're past the end of the buffer, quit. */
    if (in_char >= (in_buffer + strlen (in_buffer) - 1))
	break;
/* Otherwise, add one to ncol and skip non-spaces. */
    mat->ncol++;
    while (*in_char != ' ') in_char++;
    if (in_char >= in_buffer + strlen (in_buffer))
	break;
}

/* Okay. Now we know how many string pointers we need; allocate them.... */
if (alloc_some_char_ptrs (&(mat->data), mat->ncol * mat->nrow))
{
    sprintf (error_text, "Couldn't allocate %lu char_ptrs for %s: abort\n",
		         mat->nrow * mat->ncol, mat->name);
    matrix_error (error_text);
    return (-1);
}

/* ...and allocate strings of the appropriate length. */
for (i = 0; i < mat->nrow * mat->ncol; i++)
{
    if (alloc_some_chars (&(mat->data)[i], 
        (unsigned long)(mat->string_len + 1)))
    {
	sprintf (error_text, "Couldn't alloc. %i chars for %s (%li): abort\n",
			      mat->string_len + 1, mat->name, i);
        matrix_error (error_text);
	return (-1);
    }
}

/* Finally, we reset the file and go after the character strings.  */
long_string = FALSE;
fseek (in_file, 0L, 0);
for (i = 0; i < mat->nrow; i++)
{
    fgets (in_buffer, CHUNK, in_file);
    in_char = in_buffer;
    for (j = 0; j < mat->ncol; j++)
    {
	while (*in_char == ' ' || *in_char == '\t') in_char++;
	for (k = 0; k < mat->string_len; k++)
	{
	    mat->data[(i * mat->ncol)+j][k] = *in_char;
	    in_char++;
	    if (*in_char == ' ' || *in_char == '\t' || *in_char == '\n')
		break;
	}
/* If this isn't a space, tab, or newline, the string's too long. */
	if (*in_char != ' ' && *in_char != '\t' && *in_char != '\n')
	    long_string = TRUE;
	mat->data[(i * mat->ncol) + j][k+1] = '\0';
    }
} /* end "for i" */

return (long_string);

} /* end "read_char_matrix" */

/*======================  get_S_format_dimension  =============*/
int get_S_format_dimension (FILE *in_file,
			    unsigned long *rows, unsigned long *columns)
{

int stay_here;
int status;
int actually_read;
char buffer[CHUNK+1];
char *left_bracket, *new_line, *c_ptr;
/*
** Figure out the dimension of a matrix in S format. This format may
** split rows across different lines. We find the final line consisting
** only of one or more "tags" looking like "[,nn]"; then the last of these
** is the number of the final column (one-based). Slightly more easily,
** the "tag" at the front of the last line is the number of the final
** row (one-based). There's no easy way I know of to read backwards,
** so we read the final 300 or so characters, seek the "[" character,
** and the number following is the desired row number. If there's no such
** character, keep going 'til there is.
*/

if (fseek (in_file, (long) -CHUNK, SEEK_END) < 0)
    fseek (in_file, 0L, SEEK_SET);
buffer[CHUNK] = '\0';
while (1)
{
    fread (buffer, 1, CHUNK, in_file);
    left_bracket = strrchr (buffer, (int) '[');
    if (left_bracket != NULL)
	break;
    fseek (in_file, (long) -CHUNK, SEEK_CUR);
}

/* Okay. "left_bracket" points to the left bracket. The next number is
** the final row.
*/

sscanf (left_bracket + 1, "%lu", rows);

/*
** On to columns. To get this, we will read lines backwards, looking for the
** first one where the final character on the line is a "]". Then we back
** up to the corresponding "[" and that there is the number of columns.
*/

fseek (in_file, (long) -CHUNK, SEEK_END);
*columns = 0L;
while (1)
{
    actually_read = fread (buffer, 1, CHUNK, in_file);
    stay_here = TRUE;
    while (stay_here)
    {
	new_line = strrchr (buffer, (int) '\n');
	if (new_line == NULL)
	{
	    stay_here = FALSE;
	    break;
	}
	*new_line = '\0';
	c_ptr = new_line - 1;
	while (c_ptr >= buffer && *c_ptr == ' ')
	    c_ptr --;
	if (*c_ptr == ']')
	{
	    *c_ptr = '\0';
	    c_ptr = strrchr (buffer, (int) '[');
	    if (c_ptr == NULL)
	    {
/*
** If we get here, we found a right bracket but through bad luck missed the
** corresponding left one. So we back up, oh, 10 characters, and read 11, to
** get the right bracket we just had; then back up to the first left preceding
** that right one, and we're positioned correctly. Remember that the "current"
** position is at the end of this chunk, so we put c_ptr back at where the
** right bracket used to be and position ourselves CHUNK + (c_ptr - buffer) - 10
** bytes before the current file position.
*/
		status = -1 * (long) (actually_read  + strlen (buffer) + 10L);
		status = fseek (in_file, (long) status, SEEK_CUR);
		fread (buffer, 1, 15 + strlen(buffer), in_file);
		buffer[15 + strlen (buffer)] = '\0';
		c_ptr = strrchr (buffer, (int) '[');
		while (!isdigit ((int) *c_ptr)) c_ptr++;
		sscanf (c_ptr, "%lu", columns);
	    }
	    else
/* Otherwise, we're in the rigth spot; move past "[," and find the number. */
		sscanf (c_ptr + 2, "%lu", columns);
	    break;
	}

    }
    if (*columns != 0L)
	break;
    if (fseek (in_file, (long) -2 * CHUNK, SEEK_CUR) < 0)
	fseek (in_file, 0L, SEEK_SET);
}

return (0);
} /* end "get_S_format_dimension" */

/*=========================== read_X_and_y  ====================*/
int read_X_and_y (MATRIX *X, MATRIX *y, FILE *in_file,
		  int dimension, unsigned long which)
{
/*
** Read a matrix that consists of an X matrix with a y matrix in column
** (or row, depending on "dimension") number "which."  This function uses
** read_matrix() to get the matrix, then "matrix_extract" to pull out
** the y, and then copies the old X_data to the new X_data, omitting
** that column (row). Pretty heavy-handed, but hey.
**
** This presumes that the result of read_matrix() is REGULAR.
*/

double *new_X_data;
unsigned long i, j;
unsigned long how_many_new;
unsigned long counter;
int status;

status = read_matrix (X, in_file);
if (status != 0)
    return (status);
y->ncol = 1L; y->nrow = X->nrow;

if (alloc_some_doubles (&(y->data), dimension == COLUMN? X->nrow : X->ncol))
{
    sprintf (error_text, "Allocation for %lu data bytes failed on %s\n",
		         (dimension == COLUMN? X->nrow: X->ncol), X->name);
    matrix_error (error_text);
    return (-1);
}

status = matrix_extract (X, dimension, which, y, FALSE);
if (status != 0)
    return (status);
if (dimension == COLUMN)
    how_many_new = (X->ncol - 1) * X->nrow;
else
    how_many_new = X->ncol * (X->nrow - 1);

if (alloc_some_doubles (&new_X_data, how_many_new))
{
    sprintf (error_text, "Allocation for %lu data bytes failed on %s\n",
		          how_many_new, X->name);
    matrix_error (error_text);
    return (-1);
}

counter = 0L;
for (i = 0; i < X->nrow; i++)
{
    for (j = 0; j < X->ncol; j++)
    {
	if ( (dimension == ROW && i == which)
	||   (dimension == COLUMN && j == which))
	    continue;
	new_X_data[counter] = *SUB (X, i, j);
	counter++;
    }
}
if (dimension == COLUMN)
    X->ncol--;
else
    X->nrow--;

free (X->data);
X->data = new_X_data;
return (0);

} /* end "read_X_and_Y" */

/*=========================== make_matrix  ====================*/
MATRIX *make_matrix (unsigned long rows, unsigned long columns,
		     char *name, int type, int allocate_data)
{
/*
** This function returns a pointer to a MATRIX structure, with the
** given number of rows and columns, and the given name and storage type.
** If "allocate_data" is TRUE, it also allocates a data and "columns_in"
** vector.
*/
MATRIX *mat;		     /* Matrix to be returned  */
unsigned long i,	     /* Loop counter	       */
	 how_many_entries;   /* Number of data entries */

/* Allocate space for the matrix. If it fails, quit. */
mat = (MATRIX *) calloc (sizeof (MATRIX), 1);
if (mat == (MATRIX *) NULL)
{
    sprintf (error_text, "Alloc. for matrix body failed on %s\n", mat->name);
    matrix_error (error_text);
    return (NULL);
}

/* Fill entries with given information. */
mat->nrow = rows;
mat->ncol = columns;
strcpy (mat->name, name);
mat->sym_storage = type;

/*
** If the user asks nicely, allocate some "columns_in" ints, and
** make 'em all TRUE...
*/
if (allocate_data)
{
    mat->columns_in = (int *) NULL;
/********* DELETED FOR NOW **********
    if (alloc_some_ints (&(mat->columns_in), mat->ncol))
    {
	fprintf (stderr, "Allocation for %lu indicator bytes failed on %s\n",
			  mat->ncol, mat->name);
	return (NULL);
    }
    for (i = 0; i < columns; i ++)
	*(mat->columns_in + i) = TRUE;
********* DELETED FOR NOW **********/
/* ...plus figure out how many data items it needs, and get'em. */
    if (type == REGULAR)
	how_many_entries = mat->nrow * mat->ncol;
    else
	how_many_entries = (mat->nrow * (mat->nrow + 1) ) / 2;
    if (alloc_some_doubles (&(mat->data), how_many_entries))
    {
	sprintf (error_text, "Allocation for %lu doubles failed on %s\n",
			     how_many_entries, mat->name);
        matrix_error (error_text);
	return (NULL);
    }
    if (allocate_data == ZERO_THE_MATRIX)
    {
	for (i = 0; i < how_many_entries; i++)
	    mat->data[i] = 0.0;
    }

}
else
{
/*...if the user doesn't want data, make "data" and "columns in" NULL. */
    mat->data = (double *) NULL;
    mat->columns_in = (int *) NULL;
}
return (mat);
} /* end "make_matrix" */

/*=========================== make_long_matrix  ====================*/
LONG_MATRIX *make_long_matrix (unsigned long rows, unsigned long columns,
		     char *name, int type, int allocate_data)
{
/*
** This function returns a pointer to a LONG_MATRIX structure, with the
** given number of rows and columns, and the given name and storage type.
** If "allocate_data" is TRUE, it also allocates a data vector.
*/
LONG_MATRIX *mat;	     /* Matrix to be returned  */
unsigned long i,	     /* Loop counter	       */
	 how_many_entries;   /* Number of data entries */

/* Allocate space for the matrix. If it fails, quit. */
mat = (LONG_MATRIX *) calloc (sizeof (MATRIX), 1);
if (mat == (LONG_MATRIX *) NULL)
{
    fprintf (stderr, "Allocation for matrix body failed on %s\n", mat->name);
    return (NULL);
}

/* Fill entries with given information. */
mat->nrow = rows;
mat->ncol = columns;
strcpy (mat->name, name);
mat->sym_storage = LONG;

/*
** Whether the user asks nicely or not, don't allocate some "columns_in"
** ints...
*/
if (allocate_data)
{
    mat->columns_in = (int *) NULL;

/* ...plus figure out how many data items it needs, and get'em. */
    if (type == REGULAR)
	how_many_entries = mat->nrow * mat->ncol;
    else
	how_many_entries = (mat->nrow * (mat->nrow + 1) ) / 2;
    if (alloc_some_longs (&(mat->data), how_many_entries))
	{
	sprintf (error_text, "Allocation for %lu longs failed on %s\n",
			      how_many_entries, mat->name);
        matrix_error (error_text);
	return (NULL);
    }
    if (allocate_data == ZERO_THE_MATRIX)
    {
	for (i = 0; i < how_many_entries; i++)
	    mat->data[i] = (long) 0;
    }

}
else
{
/*...if the user doesn't want data, make "data" and "columns in" NULL. */
    mat->data = (long *) NULL;
    mat->columns_in = (int *) NULL;
}
return (mat);
} /* end "make_long_matrix" */


/*=========================== make_char_matrix	====================*/
CHAR_MATRIX *make_char_matrix (unsigned long rows, unsigned long columns,
                               char *name, int type,
			       int allocate_data, int string_len)
{
CHAR_MATRIX *mat;
long i, how_many_entries;
/*
** Create a character matrix.
*/

/* First, allocate space for the matrix. If it fails, quit. */
mat = (CHAR_MATRIX *) calloc (sizeof (CHAR_MATRIX), 1);
if (mat == (CHAR_MATRIX *) NULL)
{
    sprintf (error_text, "Allocation for char matrix body failed on %s\n",
		         mat->name);
    matrix_error (error_text);
    return (NULL);
}

/* Fill entries with given information. */
mat->nrow = rows;
mat->ncol = columns;
strcpy (mat->name, name);
mat->sym_storage = type;
mat->string_len = string_len;

/* If allocate_data is TRUE... */
if (allocate_data)
{
/* ...figure out how many data items it needs, and get'em. */
    how_many_entries = mat->nrow * mat->ncol;
    if (alloc_some_char_ptrs (&(mat->data), how_many_entries))
    {
	sprintf (error_text, "Allocation for %lu pointers failed on %s\n",
			     how_many_entries, mat->name);
        matrix_error (error_text);
	return (NULL);
    }
    for (i = 0L; i < how_many_entries; i++)
    {
	if (alloc_some_chars (&(mat->data[i]), (unsigned long) mat->string_len))
	{
	    sprintf (error_text, "Alloc. for %i chars failed on %s (%li)\n",
			         mat->string_len, mat->name, i);
            matrix_error (error_text);
	    return (NULL);
	}
    }
} /* end "if allocate_data" */

return (mat);

} /* end "make_char_matrix" */

/*=========================== matrix_invert  ====================*/
int matrix_invert (MATRIX *in_mat, MATRIX *out_mat, int special)
{
int sweep_result;

/*
** Call "invert_upper_triangular" if the input matrix is upper-
** triangular. Otherwise, copy the matrix from "in" to "out"
** and call "sweep" to invert it by sweeping all the columns.
*/
if (in_mat->sym_storage == UPPER_TRIANGULAR)
    return (invert_upper_triangle (in_mat, out_mat, special));

/* If there's no out_mat specified, invert the in_mat. */
if (out_mat == (MATRIX *) NULL)
    sweep_result = sweep (in_mat, NULL);
else
{
    matrix_copy (out_mat, in_mat);
    sweep_result = sweep (out_mat, NULL);
}

return (sweep_result);

} /* end "matrix_invert" */

/*======================== invert_upper_triangle ====================*/
int invert_upper_triangle (MATRIX *in_mat, MATRIX *out_mat, int special)
{
/*
** This inverts an upper-triangular matrix and puts the result in "out_mat".
** If there's no "out_mat", it does it in place. "special" currently can
** take the value "SHORTEN_BOTH", which inverts the top left (n-1)x(n-1)
** submatrix. So handy for the regression problem.
*/
long i, j;
long k;
unsigned long upper_limit;
double sum_holder;

if (out_mat == (MATRIX *) NULL)
{
    out_mat = in_mat;
}
/*
** "upper_limit" is one smaller than the number of columns in in_mat, because
** of zero-based addressing: if special is SHORTEN_BOTH, make it 2 smaller.
*/
if (special == SHORTEN_BOTH)
    upper_limit = in_mat->ncol - 2;
else
    upper_limit = in_mat->ncol - 1;
for (j = upper_limit; j >= 0; j--)
{
    if (fabs (*UPPER_SUB (in_mat, j, j)) < SQRT_EPS)
    {
	return (-2);
    }
    *UPPER_SUB (out_mat, j, j) = 1 / *UPPER_SUB (in_mat, j, j);
    for (k = j - 1; k >= 0; k--)
    {
	sum_holder = 0.0;
	for (i = k + 1; i <= j; i++)
	    sum_holder += *UPPER_SUB(in_mat, k, i) *
			      *UPPER_SUB(out_mat, i, j);
	*UPPER_SUB (out_mat, k, j)
	    = - sum_holder / *UPPER_SUB (in_mat, k, k);
    }
}
return(0);
} /* end "invert_upper_triangle" */

/*========================  matrix_extract ===========================*/
int matrix_extract (MATRIX *in_mat, int dim, unsigned long which_item,
		    MATRIX *out_mat, int delete_from_in_mat)
{
/*
** This function "extracts" a column, row or diagonal from the input matrix
** and puts it into the output matrix, which the user must have sized
** correctly and which should be REGULAR or UPPER_TRIANGULAR.  If
** "delete_from_in_mat" is TRUE, "delete" that column of the input matrix
** by setting the corresponding "columns_in" item to 0.
*/
unsigned long i, j;		/* Loop counter	  */
unsigned long counter;			/* Counts columns */
unsigned long nrow;		/* How many rows? */

/*
** For rows, grab every item from the appropriate row, except the ones
** where "column_in" is 0.
*/
if (dim == ROW)
{
    if (which_item >= in_mat->nrow)
    {
        sprintf (error_text, "Can't get row %ld; matrix has only %ld\n",
                              which_item, in_mat->nrow);
        matrix_error (error_text);
        return (CASE_ERROR);
    }
    counter = 0L;
    if (in_mat->sym_storage == REGULAR)
    {
	for (i = 0; i < in_mat->ncol; i++)
	{
	    if (in_mat->columns_in == NULL)
	    {
		*(out_mat->data + i) = *SUB (in_mat, which_item, i);
		continue;
	    }
	    if (in_mat->columns_in[i] == 0)
		continue;
	    *(out_mat->data + counter) = *SUB(in_mat, which_item, i);
	    counter++;
	}
    return (0);
    }
/* Currently, we only do REGULAR matrices, except for diagonals. */
    matrix_error ("Can't extract from non-regular matrices\n");
    return (NOT_YET_INSTALLED);
}

/* For diagonals, grab items, move 'em, split. */
if (dim == DIAGONAL)
{
    if (in_mat->sym_storage == REGULAR)
    {
	for (i = 0; i < in_mat->ncol; i++)
	    *(out_mat->data + i) = *SUB(in_mat, i, i);
	return (0);
    }
    if (in_mat->sym_storage == UPPER_TRIANGULAR)
    {
	for (i = 0; i < in_mat->ncol; i++)
	    *(out_mat->data + i) = *UPPER_SUB(in_mat, i, i);
	return (0);
    }
    matrix_error ("Can't extract from non-regular matrices II\n");
    return (NOT_YET_INSTALLED);
}

if (in_mat->sym_storage != REGULAR)
{
    matrix_error ("Can't extract from non-regular matrices III\n");
    return (NOT_YET_INSTALLED);
}

if (which_item >= in_mat->ncol)
{
    sprintf (error_text, "Can't get column %ld; matrix has only %ld\n",
                          which_item, in_mat->ncol);
    matrix_error (error_text);
    return (CASE_ERROR);
}

nrow = in_mat->nrow;
/* This gets the correct column by omitting those where "columns_in" is 0. */
counter = (long) -1;
if (in_mat->columns_in != NULL)
    for (j = 0; j < in_mat->ncol; j++)
    {
	if (in_mat->columns_in[j])
	    counter++;
	if (counter == which_item)
	    break;
    }
else
    j = which_item;

/* Copy the elements. */
for (i = 0; i < nrow; i++)
    *(out_mat->data + i) = * SUB (in_mat, i, j);

/* "Delete" that column if necessary. */
if (delete_from_in_mat && in_mat->columns_in != NULL)
    *(in_mat->columns_in + j) = FALSE;

return (0);
} /* end "matrix_extract" */

/*=========================  gs =======================================*/
int gs (MATRIX *in_mat, MATRIX *q_mat, MATRIX *r_mat,
	unsigned long columns_to_do)
{
/*
** This function does the gram-schmidt decomposition of a matrix. If
** q_mat is NULL, the in_matrix is orthogonalized; if q_mat is not
** NULL, we copy in_mat to q_mat and do it there. "Columns_to_do"
** tells us the number of columns of the matrix to orthogonalize;
** this will normally be the number of columns in in_mat, but may be
** one smaller for getting residuals in the regression case.
*/
unsigned long ncols, nrows;  /* Dim. of in_mat, to save time	    */
unsigned long i, j, k;	     /* Loop counters			    */
double *jth_diag_of_r;	     /* Points to jth diagonal element of R */
double *jk_th_element_of_r;  /* Points to jkth element of R	    */
/*
** If columns to do is 0, do 'em all. Otherwise, do that many, unless that
** number is too big.
*/
columns_to_do = (columns_to_do == 0 ?
		     in_mat->ncol : MIN (columns_to_do, in_mat->ncol));
ncols = in_mat->ncol;
nrows = in_mat->nrow;

/* If no "q" is supplied, do it in place. Otherwise, copy in_mat to q. */
if (q_mat == (MATRIX *) NULL)
    q_mat = in_mat;
else
    memcpy ((char *) q_mat->data, (char *) in_mat->data,
	    ncols * nrows * sizeof (double));

for (j = 0; j < columns_to_do; j++)
{
/* Point to the jth diagonal element of R... */
    if (r_mat->sym_storage == REGULAR)
	jth_diag_of_r = SUB (r_mat, j, j);
    else
	jth_diag_of_r = UPPER_SUB (r_mat, j, j);
/* ..and put the normalized length of the jth column of q_mat there. If that
** length is too small, quit. */
    *jth_diag_of_r = sqrt (matdot (q_mat, COLUMN, j, q_mat, COLUMN, j));
    if (*jth_diag_of_r < SQRT_EPS)
	return ((int) j);

/* Rescale the jth_colum of q by that amount....*/
    for (i = 0; i < nrows; i++)
	* SUB (q_mat, i, j) /= *jth_diag_of_r;
/* and fix up the elements to the right in both q and r. */
    for (k = j+1; k < ncols; k++)
    {
	jk_th_element_of_r = UPPER_SUB (r_mat, j, k);
	*jk_th_element_of_r = dot (q_mat->data + j, q_mat->data + k,
				   ncols, ncols, nrows);
	for (i = 0; i < nrows; i++)
	    *SUB(q_mat, i, k) -= *SUB(q_mat, i, j) * *jk_th_element_of_r;
    }
}

/* If r_mat is REGULAR, zero the lower triangle. */
if (r_mat->sym_storage == REGULAR)
    for (j = 0; j < ncols; j++)
	for (i = j+1; i < ncols; i++)
	    * SUB(r_mat, i, j) = 0.0;

return (QR_COMPLETE);
} /* end "gs" */

/*=======================  sum_of_squares   ===========================*/
double sum_of_squares (MATRIX *a)
{
/* This computes the sum of the squares of the entries in a matrix. */
unsigned long i, j, how_many;
double sum;
sum = 0.0;
switch (a->sym_storage)
{
    case REGULAR:
	how_many = a->ncol * a->nrow;
	for (i = 0; i < how_many; i++)
	    sum += pow (*(a->data + i), 2.0);
	break;
    case SYMMETRIC:
	for (i = 0; i < a->nrow; i++)
	    for (j = 0; j < a->ncol; j++)
		if (i == j)
		    sum += pow (*SYM_SUB (a, i, i), 2.0);
		else
		    sum += 2 * pow (*SYM_SUB (a, i, j), 2.0);
    case UPPER_TRIANGULAR:
    case LOWER_TRIANGULAR:
	how_many = (a->nrow * (a->nrow + 1)) / 2;
	for (i = 0; i < how_many; i++)
	    sum += pow (*(a->data + i), 2.0);
	break;
}
return (sum);
}

/*=======================  Cholesky  ================================*/
int Cholesky (MATRIX *in_mat, MATRIX *out_mat)
{
/*
** Do the Cholesky decomposition of in_mat; put result into out_mat if
** that thing is not NULL, or into in_mat if it is.
*/
unsigned long ncols, nrows;  /* Number of rows and columns		  */
unsigned long i, j;	     /* Loop counters				  */
double *ith_in_diag,
       *ith_out_diag;	     /* Diagonal elements of in and out matrices  */
double *i_jth_in_element,
       *i_jth_out_element;   /* Off-diagonal elements			  */

ncols = in_mat->ncol;
nrows = in_mat->nrow;

/* If no out_matrix is given, do this in place. */
if (out_mat == (MATRIX *) NULL)
{
    out_mat = in_mat;
}
/* Follow the algorithm blindly. It seems to work. */
for (i = 0; i < ncols; i++)
{
    if (in_mat->sym_storage == REGULAR)
	ith_in_diag = SUB (in_mat, i, i);
    else
	ith_in_diag = UPPER_SUB (in_mat, i, i);
    if (*ith_in_diag <= 0)
	return (-1);
    if (out_mat->sym_storage == REGULAR)
	ith_out_diag = SUB (out_mat, i, i);
    else
	ith_out_diag = UPPER_SUB (out_mat, i, i);
    *ith_out_diag = *ith_in_diag - dot (out_mat->data + i, out_mat->data + i,
					nrows, nrows, i);
    if (*ith_out_diag <= SQRT_EPS)
	return (-2);
    *ith_out_diag = sqrt (*ith_out_diag);
    for (j = i+1; j < ncols; j++)
    {
	i_jth_in_element = ith_in_diag + (j-i);
	i_jth_out_element = ith_out_diag + (j-i);
	*(i_jth_out_element) =
	    (*(i_jth_in_element) - dot (out_mat->data +i, out_mat->data +j,
				     nrows, nrows, i))
		/ *ith_out_diag;
    }
}

/* Zero the lower triangle if necessary. */
if (out_mat->sym_storage == REGULAR)
    for (i = 0; i < ncols; i++)
	for (j = 0; j < i; j++)
	    *(out_mat->data + (i * nrows) + j) = 0.0;

return (0);
} /* end "Cholesky" */

/*===========================  matrix_mean ========================*/
double matrix_mean (MATRIX *a)
{
/*
** Use matrix_sum to compute the mean of elements in a matrix.
** (I know I should use something from lab2, but...)
*/
unsigned long how_many;

if (a->sym_storage == SYMMETRIC)
    return (NOT_YET_INSTALLED);

if (a->sym_storage == REGULAR)
    how_many = a->ncol * a->nrow;
else
    how_many = (a->ncol * (a->ncol + 1)) / 2;

return (matrix_sum (a)/how_many);
}

/*=========================  matrix_sum	 =====================*/
double matrix_sum (MATRIX *a)
{
/*
** Find the sum of elements in a matrix. Symmetrics not handled.
*/
unsigned long i, how_many;
double sum;

if (a->sym_storage == SYMMETRIC)
    return (NOT_YET_INSTALLED);

/* find out how many there are. */
if (a->sym_storage == REGULAR)
    how_many = a->ncol * a->nrow;
else
    how_many = (a->ncol * (a->ncol + 1)) / 2;

/* Add 'em up. */
sum = 0.0;
for (i = 0; i < how_many; i++)
    sum += a->data[i];

return (sum);
} /* end "matrix_sum" */

/*=============================== matrix_min  ===============================*/
double matrix_min (MATRIX *mat, long *which_row, long *which_col,
                   long largest_row)
{
/*
** Go through a matrix and find the smallest value. Return that value; put
** the indices of the row and column into the arguments, if asked.
** If "largest_row" is supplied, only consider rows numbered 0 to
** largest_row - 1.
*/
long row_ctr, col_ctr;
long nrow, ncol;
double smallest_so_far, this_double;
long smallest_row, smallest_col;

nrow = MIN ((unsigned) largest_row, mat->nrow);
ncol = mat->ncol;
smallest_row    = 0L;
smallest_col    = 0L;
smallest_so_far = 0.0;
for (row_ctr = 0L; row_ctr < nrow; row_ctr++)
{
    for (col_ctr = 0L; col_ctr < ncol; col_ctr++)
    {
        this_double = *SUB (mat, row_ctr, col_ctr);
        if (smallest_row <= 0 ||  this_double < smallest_so_far)
        {
            smallest_row    = row_ctr;
            smallest_col    = col_ctr;
            smallest_so_far = this_double;
        }
    }
}

if (which_row != (long *) NULL)
    *which_row    = smallest_row;
if (which_col != (long *) NULL)
    *which_col    = smallest_col;

return (smallest_so_far);
} /* end "matrix_min" */


/*=========================  alloc_some_matrices  =====================*/
int alloc_some_matrices (MATRIX ***my_array, unsigned long how_many)
{
/* Allocate "how_many" matrix pointers in memory. */
if ( (*my_array = (MATRIX **) calloc (how_many * sizeof (MATRIX *), 1)) == NULL)
{
    sprintf (error_text, "Unable to alloc %lu matrix pointers\n", how_many);
    matrix_error (error_text);
    return (-1);
}
return (0);
} /* end "alloc_some_matrices" */

/*=========================  matrix_copy  =====================*/
int matrix_copy (MATRIX *to, MATRIX *from)
{
/*
** Copy a matrix from "from" to "to". Symmetrics not handled.
** Ignore "columns_in".
*/
unsigned long how_many;

if (from->sym_storage == SYMMETRIC)
{
    matrix_error ("Can't copy from a symmetric matrix (yet).\n");
    return (NOT_YET_INSTALLED);
}

/* The "to" matrix must exist and have its data allocated as well. */
if (to	     == (MATRIX *) NULL
||  to->data == (double *) NULL)
{
    matrix_error ("Can't copy to null matrix or one with no data\n");
    return (ALLOCATION_ERROR);
}

/* For the moment, we require the matrices be of the same size. */
if (to->nrow != from->nrow
||  to->ncol != from->ncol)
{
    matrix_error ("Can't copy matrices of different sizes\n");
    return (NON_CONFORMABLE);
}

/* Figure out how many data items to copy, and do so. */
how_many = (from->sym_storage == REGULAR ? from->nrow * from->ncol :
					   from->nrow * (from->nrow + 1) /  2);
memcpy ((char *) to->data, (char *) from->data, how_many * sizeof (double));

/* Then copy the matrix structure itself, except the data pointer. */
from->sym_storage = to->sym_storage;
from->nrow = to->nrow;
from->ncol = to->ncol;
return (0);
} /* end "matrix_copy" */

/*========================== matrix_copy_portion =====================*/
int matrix_copy_portion (MATRIX *to, MATRIX *from,
                         long row_count, long *rows,
                         long col_count, long *columns)
{
/*
** This copies the set of rows and columns named by the vectors "rows"
** and "columns" from "from" to "to." "To" must exist, have data already
** allocated, and be of the correct dimension. Special cases: if row_count
** <= 0, take all rows (and likewise for columns); in this case the pointers
** will be NULL. If the row pointers are null but the row count is > 0,
** take the first "row_count" rows, and similarly for columns.
*/
unsigned long i, j;
unsigned long source_row, source_col;

if (from->sym_storage == SYMMETRIC)
{
    matrix_error ("Can't copy portion from symmetric matrix\n");
    return (NOT_YET_INSTALLED);
}

/* Check that "to" exists and has data allocated. */
if (to	     == (MATRIX *) NULL
||  to->data == (double *) NULL)
{
    matrix_error ("Can't copy portion to null matrix or one with no data\n");
    return (ALLOCATION_ERROR);
}

zero_matrix (to);

/* If a count is <= 0, make it the number in the source matrix. */
if (row_count <= 0)
    row_count = from->nrow;
if (col_count <= 0)
    col_count = from->ncol;

for (i = 0; i < (unsigned) row_count; i++)
{
/*
** Set up the row we're extracting from: the ith element of rows, if
** rows was passed, or the ith row otherwise.
*/
    if (rows == (long *) NULL)
        source_row = i;
    else
        source_row = rows[i];
    for (j = 0; j < (unsigned) col_count; j++)
    {
        if (columns == (long *) NULL)
            source_col = j;
        else
            source_col = columns[j];
        *SUB (to, i, j) = *SUB (from, source_row, source_col);
    }
}

return (TRUE);

} /* end "matrix_copy_portion" */

/*===========================  matrix_duplicate  ============================*/
int matrix_duplicate (MATRIX *to, MATRIX *from)
/*
** Force a copy, by setting nrow and ncol in "to" to match those
** in "from" and allocating data for "to." We don't free the data
** pointer in to, though, since it may not have been allocated. (Thus
** there is a potential leak here.)
*/
{
to->nrow = from->nrow;
to->ncol = from->ncol;
if (alloc_some_doubles (&(to->data), to->nrow * to->ncol))
{
    matrix_error ("Can't allocate for duplication");
    return (ALLOCATION_ERROR);
}

matrix_copy (to, from);

return (0);

} /* end "matrix_duplicate" */

/*===========================  sweep  ================================*/
int sweep (MATRIX *in_matrix, int *which_cols)
{
/*
** Perform the Beaton sweep on whichever columns of the matrix have
** non-zero indicators in the "which_cols" array. If the pointer to
** that array is NULL, do it on all columns, thereby inverting the matrix.
*/
unsigned long i, j, k;
double b, d;

if (in_matrix->sym_storage != REGULAR)
{
    matrix_error ("Can't sweep non-regular matrix\n");
    return (NOT_YET_INSTALLED);
}

for (k = 0; k < in_matrix->ncol; k++)
{
    if (which_cols != NULL)
	if (which_cols[k] == 0)
	    continue;
    d = *SUB(in_matrix, k, k);
    if (fabs (d) < SQRT_EPS)
    {
        sprintf (error_text, "Sweep too small at column %li\n", k);
        matrix_error (error_text);
	return (TOO_SMALL);
    }
    for (i = 0; i < in_matrix->ncol; i++)
    {
	*SUB (in_matrix, k, i) /= d;
    }
    for (i = 0; i < in_matrix->ncol; i++)
    {
	if (i == k)
	    continue;
	b = *SUB (in_matrix, i, k);
	for (j = 0; j < in_matrix->ncol; j++)
	{
	    *SUB (in_matrix, i, j) -= b * *SUB (in_matrix, k, j);
	}
	*SUB (in_matrix, i, k) = -b / d;
    }
    *SUB (in_matrix, k, k) = 1.0/d;
}

return (0);
} /* end "sweep" */

int sweep_all_but_last (MATRIX *in_matrix)
{
/*
** It happens frequently in regression that we want to sweep all but the
** last column. So it has its own function.
*/
unsigned long i;
int *which_cols;

if (alloc_some_ints (&which_cols, in_matrix->ncol) != 0)
{
    sprintf (error_text, "Sweep all but last: Couldn't allocate %li ints\n",
                         in_matrix->ncol);
    matrix_error (error_text);
    return (ALLOCATION_ERROR);
}

for (i = 0; i < (in_matrix->ncol - 1); i++)
    which_cols[i] = 1;
which_cols[in_matrix->ncol - 1] = 0;
sweep (in_matrix, which_cols);
free (which_cols);

return (TRUE);
}

/*===========================  is_symmetric  ========================*/
int is_symmetric (MATRIX *a)
{
unsigned long i, j;

if (a->nrow != a->ncol)
    return (FALSE);
if (a->sym_storage == SYMMETRIC)
    return (TRUE);
for (i = 0; i < a->nrow; i++)
    for (j = i; j < a->ncol; j++)
	if (fabs (*SUB (a, i, j) - *SUB (a, j, i) ) > SYM_TOLERANCE)
	    return (FALSE);
return (TRUE);
}

/*=======================  regress_ls ================================*/

int regress_ls (MATRIX *X, MATRIX *y, MATRIX *beta, MATRIX *var_unscaled,
		int repeated)
{
/*
** Do least-squares regression of y on X, producing beta-hats. If ``repeated"
** is TRUE, only set up the Q, R, and Q-t-y matrices the first time (or
** if repeated turns FALSE again) for speed.
*/
static MATRIX *Q, *R, *Q_transpose_y, *variance;
int status;
static int initialized = FALSE;

if (repeated == FALSE)
{
    initialized = FALSE;
}

if (!initialized)
{
    Q = make_matrix (X->nrow, X->ncol, "Q matrix", REGULAR, TRUE);
    R = make_matrix (X->ncol, X->ncol, "R matrix", UPPER_TRIANGULAR, TRUE);
    Q_transpose_y = make_matrix (X->ncol, 1L, "Q_transpose y", REGULAR, TRUE);
/* This matrix should be symmetric. */
    if (var_unscaled == (MATRIX *) NULL)
	variance = (MATRIX *) NULL;
    else
	variance = make_matrix (X->ncol, X->ncol, "Variance", REGULAR, TRUE);
    initialized = TRUE;
}

/* Do the QR (gram-schmidt) decomposition of X... */
if ((status = gs (X, Q, R, 0L)) != QR_COMPLETE)
{
    if (!repeated)
    {
	free (Q->data);
	free (R->data);
	free (Q_transpose_y->data);
	if (var_unscaled != NULL)
	    free (variance->data);
    }
    return (-1);
}

/* ...and invert R. Recall that beta-hat = r-inv q-transpose y. */
if ((status = matrix_invert (R, NULL, 0)) != 0)
{
    if (!repeated)
    {
	free (Q->data);
	free (R->data);
	free (Q_transpose_y->data);
	if (var_unscaled != NULL)
	    free (variance->data);
    }
    return (status);
}

if ((status = matrix_multiply (Q, y, Q_transpose_y, TRANSPOSE_FIRST)) != 0)
{
    if (!repeated)
    {
	free (Q->data);
	free (R->data);
	free (Q_transpose_y->data);
	if (var_unscaled != NULL)
	    free (variance->data);
    }
    return (status);
}

/*
** If anyone asked for variances of the beta-hats, get ready to provide them
** by forming R-inverse times R-inv-transpose. Those entries will then have
** to be multiplied by an estimate of sigma-squared, but WE LEAVE THAT FOR
** THE CALLER (that's because that multiplication would mess us up when
** we're doing wls).
*/
if (var_unscaled != (MATRIX *) NULL && variance->data != (double *) NULL)
{
    matrix_multiply (R, R, variance, TRANSPOSE_SECOND);
    matrix_extract (variance, DIAGONAL, 1L, var_unscaled, FALSE);
}

if ((status = matrix_multiply (R, Q_transpose_y, beta, NO_TRANSPOSES)) != 0)
{
    if (!repeated)
    {
	free (Q->data);
	free (R->data);
	free (Q_transpose_y->data);
	if (var_unscaled != NULL)
	    free (variance->data);
    }
    return (status);
}

if (!repeated)
{
    free (Q->data);
    free (R->data);
    free (Q_transpose_y->data);
	if (var_unscaled != NULL)
	    free (variance->data);
}

return (0);

} /* end "regress_ls" */

/*=======================  regress_wls ================================*/

int regress_wls (MATRIX *X, MATRIX *y, MATRIX *wts, MATRIX *beta,
		 MATRIX *variances, MATRIX *new_X, MATRIX *new_y, int repeated)
{
/*
** Do weighted least-squares regression of y on X, producing beta-hats.
** Here "wts" is an nx1 (not nxn) matrix of weights. These weights are
** SDs, not variances, of observations.
*/
static int initialized = FALSE;
unsigned long i, j;

if (repeated == FALSE)
    initialized = FALSE;

if (!initialized)
    initialized = TRUE;
/*
** Here's the step that makes it weighted. Copy X to new_X, and y
** to new_y; then divide each row of X and each element of y by the
** square root of the corresponding element of "wts". Then we do
** OLS in the normal way on new_X and new_y.
*/
matrix_copy (new_X, X);
matrix_copy (new_y, y);
for (i = 0L; i < new_X->nrow; i++)
{
    new_y->data[i] /= wts->data[i];
    for (j = 0L; j < new_X->ncol; j++)
	*SUB (new_X, i, j) /= wts->data[i];
}
regress_ls (new_X, new_y, variances, beta, repeated);

return (0);

} /* end "regress_wls" */

/*=====================	 transpose_S_matrix =============================*/
int transpose_S_matrix (MATRIX *a, double *data)
{
/*
** This fills a matrix row-by-row with the data in S (that is, column-by-
** column) format. It's quick and dirty. "a" must be REGULAR.
*/
unsigned long i, j;

for (i = 0; i < a->nrow; i++)
    for (j = 0; j < a->ncol; j++)
    {
	*SUB (a, i, j) = data[j * a->nrow + i];
    }

return (0);

} /* end "transpose_S_matrix" */

/*=====================	 untranspose_S_matrix =============================*/
int untranspose_S_matrix (double *data, MATRIX *a)
{
/*
** This takes a matrix stored row-by-row and puts the data into "data"
** column-by-column (for example, for an S matrix).
*/
unsigned long i, j;

for (i = 0; i < a->nrow; i++)
    for (j = 0; j < a->ncol; j++)
    {
	data[j * a->nrow + i] = *SUB (a, i, j);
    }

return (0);

} /* end "untranspose_S_matrix" */
/*=====================	 norm_of_difference =============================*/
double norm_of_difference (MATRIX *a, MATRIX *b)
{
/*
** This function computes the (L2) norm of the difference between two
** one-column matrices, that is, the sum-over-i of (a-i - b-i)^2.
*/
double sum;
unsigned long i;

/* Require that these be one-column vectors of the same length. */
if (a->ncol > 1 || b->ncol > 1)
{
    matrix_error ("Norm only makes sense on two one-column vectors\n");
    return (NOT_YET_INSTALLED);
}
if (a->nrow != b->nrow)
{
    matrix_error ("Norm called, but vectors have different lengths\n");
    return (NON_CONFORMABLE);
}

sum = 0.0;
/* Loop: add (a-i - b-i)^2 for each element. */
for (i = 0L; i < a->nrow; i++)
    sum += pow (a->data[i] - b->data[i], 2.0);

/* Now return sum. */
return (sum);
}

/*=====================	 zero_matrix =============================*/
int zero_matrix (MATRIX *a)
{
/*
** Zero out a matrix using "memset." No tricks here.
*/
memset ((char *) a->data, '\0', sizeof (double) * (int) (a->nrow * a->ncol));
return (TRUE);
}

/*==================== invert_diagonal ===========================*/
int invert_diagonal (MATRIX *a)
{
/*
** This "inverts" a diagonal matrix by replacing each diagonal entry
** by its reciprocal. We don't require a matrix to be of type DIAGONAL,
** since this is rarely used. The check for zero entries is based
** on SYM_TOLERANCE, which was originally intended for something else,
** but hey. Check all entries before doing any of them.
*/
unsigned long i;
int any_too_small;

/* This only makes sense for square matrices. */

if (a->nrow != a->ncol)
{
    matrix_error ("Can't do invert-diagonal on non-square matrix\n");
    return (NON_CONFORMABLE);
}

/* Go through the diagonals, looking for one that's too small. */
any_too_small = FALSE;
for (i = 0L; i < a->nrow; i++)
{
    if (fabs (*SUB (a, i, i)) < SYM_TOLERANCE)
    {
         any_too_small = TRUE;
         break;
    }
}
if (any_too_small)
{
    matrix_error ("Invert-diagonal: some entry is too small\n");
    return (TOO_SMALL);
}

/* Okay; we're cool. Let's replace each diagonal by its reciprocal. */

for (i = 0L; i < a->nrow; i++)
{
    *SUB (a, i, i) = 1.0 / *SUB (a, i, i);
}

return (TRUE);

} /* end "invert_diagonal" */

/*================= divide_by_root_before_and_after ====================*/

int divide_by_root_before_and_after (MATRIX *in, MATRIX *diag)
{
/*
** This function takes a regular matrix and a diagonal and turns the matrix
** into (1/sqrt(diag)) * in * (1/(sqrt(diag)), where "*" means matrix
** multiplication and (1/sqrt(diag)) means the matrix diag
** (1/sqrt(diag_1), 1/sqrt(diag_2), ...).  To put it a different way, we
** divide the ij-th element of "in" by 1/sqrt(ith-element of diag) and
** by 1/sqrt(jth-element of diag).
**
** The adjusted matrix is left in "in."
*/
unsigned long i, j;
double ith_diag;
double jth_diag;

if (diag == (MATRIX *) NULL)
{
    matrix_error ("Can't <divide by root before and after>: null matrix\n");
    return (NO_DATA);
}
if (diag->data == (double *) NULL)
{
matrix_error ("Can't <divide by root before and after>: matrix has no data\n");
    return (NO_DATA);
}
for (i = 0; i < in->nrow; i++)
{
    ith_diag = *SUB (diag, i, i);
    for (j = 0; j < in->ncol; j++)
    {
        jth_diag = *SUB (diag, j, j);
        *SUB (in, i, j) = *SUB (in, i, j) / sqrt (ith_diag * jth_diag);
    }
}

return (TRUE);
}

/*========================== matrix_ridge ===============================*/
int matrix_ridge (MATRIX *a, double lambda)
{
/*
** This "ridges" a matrix, that is, adds "lambda" to every diagonal
** entry of a matrix. Works fine for non-square matrices, though
** when would you use it?
*/
long i, count;

/* The smaller of rows and columns is the number of elements to ridge. */
if (a->nrow < a->ncol)
    count = a->nrow;
else
    count = a->ncol;

for (i = 0; i < count; i ++)
{
    *SUB (a, i, i) = *SUB (a, i, i) + lambda;
}

return (TRUE);

} /* end "matrix_ridge" */


/*======================= matrix_copy_transpose ===========================*/
int matrix_copy_transpose (MATRIX *to, MATRIX *from)
{
/* Copy a matrix, transposing it. */
unsigned long row, col;

if (from == (MATRIX *) NULL || from->data == (double *) NULL)
{
    matrix_error ("Can't transpose null matrix or one with no data\n");
    return (NO_DATA);
}

if (to == (MATRIX *) NULL || to->data == (double *) NULL)
{
    matrix_error ("Can't transpose to a null matrix or one with no data\n");
    return (NO_DATA);
}

if (from->sym_storage == DIAGONAL)
    return (matrix_copy (to, from));

if (from->sym_storage != REGULAR)
{
    matrix_error ("Can't copy/transpose matrix w/o regular storage\n");
    return (NOT_YET_INSTALLED);
}

if (from->nrow != to->ncol || from->ncol != to->nrow)
{
    matrix_error ("Can't copy/transpose: sizes don't agree\n");
    return (NOT_YET_INSTALLED);
}

for (row = 0; row < from->nrow; row++)
    for (col = 0; col < from->ncol; col++)
        *SUB (to, col, row) = *SUB (from, row, col);

return (TRUE);
} /* end "matrix_copy_transpose" */

/*===================== change_to_the_identity ==========================*/
int change_to_the_identity (MATRIX *in)
{
unsigned long i, j;

if (in == (MATRIX *) NULL)
{
    matrix_error ("Can't change null matrix to identity");
    return (NO_DATA);
}
if (in->data == (double *) NULL)
{
    matrix_error ("Can't make matrix with no data into identity");
    return (NO_DATA);
}
if (in->nrow != in->ncol)
{
    matrix_error ("Can't make non-square matrix into identity");
    return (NON_CONFORMABLE);
}
for (i = 0L; i < in->nrow; i++)
{
    *SUB (in, i, i) = 1.0;
    for (j = (i + 1L); j < in->nrow; j++)
    {
            *SUB (in, i, j) = 0.0;
            *SUB (in, j, i) = 0.0;
    }
}
return (TRUE);

} /* end "change_to_the_identity" */


/*========================== scale_matrix_rows =========================*/
int scale_matrix_rows (MATRIX *in, int center, int scale, MATRIX *which,
				       int type, double *resulting_means, double *resulting_sds)
{
/*
** Scale the rows of in; at least, scale the rows for which the corresponding
** entries of "which" are non-zero. Save resulting means and sd's into results
** if asked. Center, scale, or both. If type is COMPUTE_SCALINGS, that means
** scalings will be computed and returned. If it's USE_THESE_SCALINGS, then the
** relevant scalings have been supplied.
*/
unsigned long i, j;
double this_mean, this_var, this_sd = 0.0;
int pm_result;

for (i = 0; i < in->nrow; i++)
{
/*
** Skip the rows if its entry in which is 0 (or negative)
*/
	if (which != (MATRIX *) NULL
	&&  which->data[i] <= 0)
		continue;

/*
** If we're supposed to compute the scalings, call provisional_means once for each
** entry in the row, then once more with the QUIT flag. Compute the SD.
*/
	if (type == COMPUTE_SCALINGS) {
		for (j = 0; j < in->ncol; j++) {
			pm_result = provisional_means (SUB (in, i, j), 1L, INCREMENT, &this_mean, &this_var);
		}

		pm_result = provisional_means ((double *) 0, 1L, QUIT, &this_mean, &this_var);
		this_sd = sqrt (this_var);

/* Store the resulting means and SDs if pointers to those results were passed. */
		if (resulting_means != (double *) 0)
			resulting_means[i] = this_mean;
		if (resulting_sds != (double *) 0)
			resulting_sds[i] = this_sd;
	}
	else
	{
/*
** If we're not computing the scaling, we must be actually scaling. Pick up the
** relevant means and SDs. Presumably if "center" is TRUE then we have some
** resulting_means lying around and similarly for SDs.
*/
		if (center)
			this_mean = resulting_means[i];
		if (scale)
			this_sd   = resulting_sds[i];
	}

/* Now it's time to scale. If center and scale are both false, I guess someone just wanted
** to get the means and SDs. Well, it *could* happen. Anyway, time to move on.
*/
	if (!center && !scale)
		continue;

/* Quick check: if SD is nearly 0, we're in trouble. Set it to 1.0 for now (?). */
	if (this_sd < SQRT_EPS)
		this_sd = 1.0;

/* If centering, subtract the mean. If scaling, divide by the SD. */
	if (center) {
		if (scale)
			for (j = 0; j < in->ncol; j++)
				*SUB (in, i, j) = (*SUB (in, i, j) - this_mean) / this_sd;
		else
			for (j = 0; j < in->ncol; j++)
				*SUB (in, i, j) = (*SUB (in, i, j) - this_mean);
	}
	else
	{ /* If center is FALSE, then scale must be TRUE here. */
			for (j = 0; j < in->ncol; j++)
				*SUB (in, i, j) = *SUB (in, i, j) / this_sd;
	}


}

return (0);
} /* end "scale_matrix_rows" */

/*======================= scale_matrix_rows_with_mad ===================*/
int scale_matrix_rows_with_mad (MATRIX *mat, MATRIX *which, double *resulting_mads)
{
unsigned long i, j;
int mad_status;
double compute_mad();

for (i = 0; i < mat->nrow; i++)
{
	if (which != (MATRIX *) NULL
	&&  which->data[i] <= 0)
		continue;
	resulting_mads[i] = compute_mad (SUB (mat, i, 0L), (long) mat->ncol, &mad_status);

	for (j = 0; j < mat->ncol; j++)
		*SUB (mat, i, j) = *SUB (mat, i, j) / resulting_mads[i];
}

return (0);

}

/*========================== free_matrix ===============================*/
int free_matrix (MATRIX *mat)
{
    if (mat->data != (double *) NULL)
        free (mat->data);
    free (mat);

return (0);

}

/*========================== matrix_error ===============================*/
int matrix_error (char *message)
{
/* Report errors to stderr. Someday this may change. */

fprintf (stderr, message);
fflush (stderr);

return (0);

}
