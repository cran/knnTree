#define DO_NN_INITIALIZE     1
#define DO_NN_INITIALIZE_BIG 2
#define DO_NN_COMPUTE        3
#define DO_NN_QUIT           0

#define BF_WITHIN_ONLY       2


int do_nn (long action, long first_time, MATRIX *training, MATRIX *test,
           MATRIX *c, long *k, long how_many_ks,
           long theyre_the_same, int number_of_classes,
           MATRIX *cost, MATRIX *prior, double *error_rates,
           MATRIX *misclass_mat, long return_classifications,
		   long *classifications, int verbose, FILE *status_file);



