/*
**		  Include file matrix.h
**
**   Defines for matrix routines
**
*/

/* Defines for finding matrix entries. */
#define UPPER_COL(mat, j) ( (mat->data) + (((j) * (j+1)) / 2))
#define UPPER_SUB(mat, i, j) (UPPER_COL((mat), (j)) + (i))
#define LOWER_ROW(mat, i) ( (mat->data) + (((i) * (i+1)) / 2))
#define LOWER_SUB(mat, i, j) (LOWER_COL((mat), (i)) + (j))
#define SYM_TRI(mat, i)	 ((mat->data) + ((i) * (i+1))/2)
#define SYM_SUB(mat,i,j) ((mat->data) + ((i) * (i+1))/2 + j)
#define SUB(mat, i, j) ((mat->data) + ((i) * (mat->ncol)) + (j))
#define RSUB(mat, i, j) ((mat->data) + ((j) * (mat->ncol)) + (i))
#define CHAR_SUB(mat, i, j) ((mat->data)[((i) * (mat->ncol)) + (j)])

/* Defines for which/whether matrices are transposed, and other stuff. */
#define NO_TRANSPOSES	  (0x00)
#define TRANSPOSE_FIRST	  (0x01)
#define TRANSPOSE_SECOND  (0x02)
#define TRANSPOSE_BOTH	  (0x03)

/* Define for matrix dimensions. */
#define ROW		  1
#define COLUMN		  2
#define DIAGONAL	  3

/* Defines for "sym_storage" element of MATRIX structure */
#define REGULAR		  0
#define UPPER_TRIANGULAR  1
#define LOWER_TRIANGULAR  2
#define SYMMETRIC	  3
#define CHARACTER	  4
#define LONG	          5

/* Special "cheating" define for "invert_upper_triangle" */
#define SHORTEN_BOTH	  1

/* Define for zeroing matrices made by "make_matrix" */
#define ZERO_THE_MATRIX	  2

/* Defines and flags for matrix printing */
#define PRECISION	  5
#define PER_LINE	  5
#define ROWS_PER_PANEL	  15
#define S_FORMAT	  (0x08)
#define EASY_FORMAT	  (0x10)
#define NO_TITLE	  (0x20)
#define HIGH_PRECISION    (0x40)

/* Defines for "matrix_add": is it adding or subtracting? */
#define ADD		   1
#define SUBTRACT	   2

/* Matrix routine return codes. Note that 0 is often an error. */
#define LONG_STRING	   1
#define MATRICES_EQUAL	  -1
#define QR_COMPLETE	  -1
#define NON_CONFORMABLE	  -2
#define TRANPOSE_ERROR	  -4
#define ALLOCATION_ERROR  -5
#define NOT_YET_INSTALLED -6
#define CASE_ERROR	  -7
#define TOO_SMALL	  -8
#define SEEK_FAILED	  -9
#define BAD_FORMAT	 -10
#define NO_DATA		 -11

/* Define to make clear that NULL parameters sometimes mean "do it in place" */
#define IN_PLACE	  NULL

/* Define for testing symmetry of a matrix. */
#define SYM_TOLERANCE	1.0e-8

/* Defines for "type" in scale_matrix_rows */
#define COMPUTE_SCALINGS   0
#define USE_THESE_SCALINGS 1

/* The MATRIX structure. */
#define MATRIX_NAME_LEN	  40
typedef struct matrix
{
    double *data;
    unsigned long nrow, ncol;
    int sym_storage;
    int *columns_in;
    char name[MATRIX_NAME_LEN];
} MATRIX;
/* The seldom-used but oh-so-useful "long matrix" */
typedef struct long_matrix
{
    long *data;
    unsigned long nrow, ncol;
    int sym_storage;
    int *columns_in;
    char name[MATRIX_NAME_LEN];
} LONG_MATRIX;

/* This structure defines a character matrix, a matrix of char. strings.
** Although it looks like a matrix to the user, now it can be told; it's
** just a vector of length (nrow x ncol).
*/
typedef struct char_matrix
{
    char **data;
    unsigned long nrow, ncol;
    int sym_storage;
    int string_len;
    char name[MATRIX_NAME_LEN];
} CHAR_MATRIX;

/* Defines for non-int functions in matrix.c */

double dot (double *first, double *second,
	    unsigned long first_stride, unsigned long second_stride,
	    unsigned long length);

double matdot (MATRIX *first,  int dim_1, unsigned long item_1,
	       MATRIX *second, int dim_2, unsigned long item_2);

int matrix_multiply (MATRIX *a, MATRIX *b, MATRIX *result, char transposes);
int test_multiply (MATRIX *a, MATRIX *b, MATRIX *result, char transposes);

int set_up_vector (MATRIX *in, int dim, unsigned long item, double *result);

int matrix_multiply_all_regular (MATRIX *a, MATRIX *b, MATRIX *result,
				  char transposes);

int scalar_multiply (MATRIX *a, MATRIX *result, double scalar);

int matrix_add (MATRIX *first, MATRIX *second, MATRIX *result, int which);

unsigned long  matrices_equal (MATRIX *a, MATRIX *b,
			       double tolerance, double *max_diff);

int print_matrix (MATRIX *a, char flags);

int print_char_matrix (CHAR_MATRIX *a, char flags);

int print_columns (MATRIX *a, MATRIX *b);

int read_matrix (MATRIX *mat, FILE *in_file);

int read_char_matrix (CHAR_MATRIX *mat, FILE *in_file);

int get_S_format_dimension (FILE *in_file,
			    unsigned long *rows, unsigned long *columns);

int read_X_and_y (MATRIX *X, MATRIX *y, FILE *in_file,
		  int dimension, unsigned long which);

MATRIX *make_matrix (unsigned long rows, unsigned long columns,
		     char *name, int type, int allocate_data);

LONG_MATRIX *make_long_matrix (unsigned long rows, unsigned long columns,
		     char *name, int type, int allocate_data);

CHAR_MATRIX *make_char_matrix (unsigned long rows, unsigned long columns,
		     char *name, int type, int allocate_data, int str_len);

int matrix_invert (MATRIX *in_mat, MATRIX *out_mat, int special);

int invert_upper_triangle (MATRIX *in_mat, MATRIX *out_mat, int special);

int matrix_extract (MATRIX *in_mat, int dim, unsigned long which_item,
		    MATRIX *out_mat, int delete_from_in_mat);

int gs (MATRIX *in_mat, MATRIX *q_mat, MATRIX *r_mat,
	unsigned long columns_to_do);

double sum_of_squares (MATRIX *a);

int Cholesky (MATRIX *in_mat, MATRIX *out_mat);

double matrix_mean(MATRIX *a);

double matrix_sum(MATRIX *a);

double matrix_min (MATRIX *mat, long *which_row, long *which_column,
                  long largest_row);

int alloc_some_matrices (MATRIX ***my_array, unsigned long how_many);

int matrix_copy (MATRIX *to, MATRIX *from);

int matrix_copy_portion (MATRIX *to, MATRIX *from,
                         long row_count, long *rows,
                         long col_count, long *columns);

int sweep (MATRIX *in_matrix, int *which_cols);
int sweep_all_but_last (MATRIX *in_matrix);

int is_symmetric (MATRIX *a);

int regress_ls (MATRIX *X, MATRIX *y, MATRIX *beta, MATRIX *var_unscaled,
		int repeated);

int regress_wls (MATRIX *X, MATRIX *y, MATRIX *wts, MATRIX *beta,
		 MATRIX *variances, MATRIX *new_X, MATRIX *new_y, int repeated);

int transpose_S_matrix (MATRIX *a, double *data);
int untranspose_S_matrix (double *data, MATRIX *a);
int zero_matrix(MATRIX *a);
int invert_diagonal(MATRIX *a);
int divide_by_root_before_and_after(MATRIX *a, MATRIX *b);
int matrix_ridge(MATRIX *a, double lambda);
int matrix_copy_transpose (MATRIX *to, MATRIX *from);
int change_to_the_identity (MATRIX *in);
int scale_matrix_rows (MATRIX *in, int center, int scale, MATRIX *which,
				       int type, double *resulting_means, double *resulting_sds);
int scale_matrix_rows_with_mad (MATRIX *mat, MATRIX *which, double *resulting_mads);
int free_matrix (MATRIX *a);
int matrix_error (char *message);
#if 0
double dot ();
double matdot ();
int matrix_multiply ();
int set_up_vector ();
int matrix_multiply_all_regular ();
int scalar_multiply ();
int matrix_add ();
unsigned long  matrices_equal ();
int print_matrix ();
int print_char_matrix ();
int print_columns ();
int read_matrix ();
int read_char_matrix ();
int get_S_format_dimension ();
int read_X_and_y ();
MATRIX *make_matrix (unsigned long rows, unsigned long cols, char *name, int type, int allocate);
LONG_MATRIX *make_long_matrix();
CHAR_MATRIX *make_char_matrix ();
int matrix_invert ();
int invert_upper_triangle ();
int matrix_extract ();
int gs ();
double sum_of_squares ();
int Cholesky ();
double matrix_mean();
double matrix_sum();
double matrix_min ();
int alloc_some_matrices ();
int matrix_copy ();
int matrix_copy_portion ();
int sweep ();
int sweep_all_but_last ();
int is_symmetric ();
int regress_ls ();
int regress_wls ();
int transpose_S_matrix ();
int untranspose_S_matrix ();
int zero_matrix();
int invert_diagonal();
int divide_by_root_before_and_after();
int matrix_ridge();
int matrix_copy_transpose ();
int change_to_the_identity ();
int free_matrix();
int matrix_error();
#endif
