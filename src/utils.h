/*
**		       Include file "utils.h"
**
**	Includes handy defines for general use
**
*/
#define TRUE  1
#define FALSE 0
#define NOT   !
#define SQRT_EPS (1e-10)

#define TEST_BIT(a, b) ((a) & (1 << (b)))
#define TEST_PATTERN(a, b) ((a) & (b))

#define	 MIN(a,b) ((a) < (b) ? (a) : (b))
#define	 MAX(a,b) ((a) > (b) ? (a) : (b))
#define	 ABS(a)	  ((a) < 0 ? (-(a)) : (a))

/* Includes for provisional means. */
#define INCREMENT      (long) -3
#define END_GROUP      (long) -2
#define QUIT	       (long) -1
#define ILLEGAL_LENGTH (long) -1
#define DONT_QUIT      (long)  0

int count_lines (FILE *in_file, long starting_point, unsigned long *line_count);
int alloc_some_double_pointers (double ***my_array, unsigned long how_many);
int alloc_some_doubles (double **my_array, unsigned long how_many);
int alloc_some_floats (float **my_array, unsigned long how_many);
int alloc_some_long_ptrs (long ***my_array, unsigned long how_many);
int alloc_some_longs (long **my_array, unsigned long how_many);
int alloc_some_u_longs (unsigned long **my_array, unsigned long how_many);
int alloc_some_ints (int **my_array, unsigned long how_many);
int alloc_some_char_ptrs (char ***my_array, unsigned long how_many);
int alloc_some_chars (char **my_array, unsigned long how_many);
int provisional_means (double *new_vector, long vector_length, long status,
		       double *mean, double *var);
