#include <math.h>
#include <stdlib.h>

void dsort_ (double *DX, long *N, long *KFLAG)
{
/*************
   dsort: C translation of Fortran slatec routine, with calls to
   XERRMSG removed. Also there's no optional "DY" array.

C***BEGIN PROLOGUE  DSORT
C***PURPOSE  Sort an array and optionally make the same interchanges in
C            an auxiliary array.  The array may be sorted in increasing
C            or decreasing order.  A slightly modified QUICKSORT
C            algorithm is used.
C***LIBRARY   SLATEC
C***CATEGORY  N6A2B
C***TYPE      DOUBLE PRECISION (SSORT-S, DSORT-D, ISORT-I)
C***KEYWORDS  SINGLETON QUICKSORT, SORT, SORTING
C***AUTHOR  Jones, R. E., (SNLA)
C           Wisniewski, J. A., (SNLA)
C***DESCRIPTION
C
C   DSORT sorts array DX in increasing or descresing order.
C   slightly modified quicksort algorithm is used.
C
C   Description of Parameters
C      DX - array of values to be sorted   (usually abscissas)
C      N  - number of values in array DX to be sorted
C      KFLAG - control parameter
C            =  1  means sort DX in increasing order (ignoring DY)
C            = -1  means sort DX in decreasing order (ignoring DY)
       On return, KFLAG indicates status.
C
C***REFERENCES  R. C. Singleton, Algorithm 347, An efficient algorithm
C                 for sorting with minimal storage, Communications of
C                 the ACM, 12, 3 (1969), pp. 185-187.
C***END PROLOGUE  DSORT
**/
/*    .. Local Scalars .. */
      double R, T, TT; /* TTY, TY; */
      int I, IJ, J, K, KK, L, M, NN;
/*    .. Local Arrays ..  */
      int IL[21], IU[21];
/**********************************
** BEGIN HERE
***********************************/
/* Make sure that n is greater than 1.*/
      NN = *N;
      if (NN < 1)
      {
          *KFLAG = 1L;
          return;
      }

/* Make sure that KFLAG has a recognizable value.*/
      KK = labs(*KFLAG);
      if (KK != 1 && KK != 2)
      {
          *KFLAG = 2L;
          return;
      }

/* If we're asked for a decreasing sort, simply reverse signn on DX. */

      if (*KFLAG <= -1) {
         for (I = 0; I < NN-1; I++)
            DX[I] = -DX[I];
      }
/*
C     Sort DX only. This routine used to allow for a second array, but
      I got rid of that.
*/
      M = 1;
      I = 1;
      J = NN;
      R = 0.375;

   A20:
      if (I == J) goto A60;

      if (R <= 0.5898437)
         R = R+3.90625e-2;
      else
         R = R-0.21875;


   A30:
      K = I;
/*
C     Select a central element of the array and save it in location T
*/
      IJ = I + (int)((J-I)*R);
      T = DX[IJ];
/*
C     If first element of array is greater than T, interchange with T
*/
      if (DX[I] > T)
      {
         DX[IJ] = DX[I];
         DX[I] = T;
         T = DX[IJ];
      }

      L = J;
/*
C     If last element of array is less than than T, interchange with T
*/
      if (DX[J] < T)
      {
         DX[IJ] = DX[J];
         DX[J] = T;
         T = DX[IJ];
/*
C        If first element of array is greater than T, interchange with T
*/
         if (DX[I] > T)
         {
            DX[IJ] = DX[I];
            DX[I] = T;
            T = DX[IJ];
         }
      }
/*
C     Find an element in the second half of the array which is smaller
C     than T
*/
   A40:
      L = L-1;
      if (DX[L] > T) goto A40;
/*
C     Find an element in the first half of the array which is greater
C     than T
*/
   A50:
      K = K+1;
      if (DX[K] < T) goto A50;
/*
C     Interchange these elements
*/
      if (K <= L)
      {
         TT = DX[L];
         DX[L] = DX[K];
         DX[K] = TT;
         goto A40;
      }
/*
C     Save upper and lower subscripts of the array yet to be sorted
*/
      if (L-I > J-K)
      {
         IL[M] = I;
         IU[M] = L;
         I = K;
         M = M+1;
      }
      else
      {
         IL[M] = K;
         IU[M] = J;
         J = L;
         M = M+1;
      }
      goto A70;
/*
C     Begin again on another portion of the unsorted array
*/
   A60:
      M = M-1;
      if (M == 0) goto A190;
      I = IL[M];
      J = IU[M];

   A70:
      if (J-I >= 1) goto A30;
      if (I == 1) goto A20;
      I = I-1;

   A80:
      I = I+1;
      if (I == J) goto A60;
      T = DX[I+1];
      if (DX[I] <= T) goto A80;
      K = I;

   A90:
      DX[K+1] = DX[K];
      K = K-1;
      if (T < DX[K]) goto A90;
      DX[K+1] = T;
      goto A80;
/*
C     Clean up
*/
  A190:
      if (*KFLAG <= -1)
      {
         for (I = 0; I < NN; I++)
            DX[I] = -DX[I];
      }
      return;
}
