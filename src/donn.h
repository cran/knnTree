#define DO_NN_INITIALIZE     1
#define DO_NN_INITIALIZE_BIG 2
#define DO_NN_COMPUTE        3
#define DO_NN_QUIT           0

#define BF_WITHIN_ONLY       2

#ifdef BUILDING_FOR_R
#define INT_OR_LONG int
#else
#define INT_OR_LONG long
#endif

int do_nn (INT_OR_LONG action, INT_OR_LONG first_time, MATRIX *training, 
           MATRIX *test,
           MATRIX *c, INT_OR_LONG *k, INT_OR_LONG how_many_ks,
           INT_OR_LONG theyre_the_same, INT_OR_LONG number_of_classes,
           MATRIX *cost, MATRIX *prior, double *error_rates,
           MATRIX *misclass_mat, INT_OR_LONG return_classifications,
		   INT_OR_LONG *classifications, INT_OR_LONG verbose, 
                   FILE *status_file);



