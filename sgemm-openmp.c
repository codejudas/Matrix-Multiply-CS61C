#include <nmmintrin.h> 
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void sgemm( int m_a, int n_a, float *A, float *B, float *C ) {
  static __m128 _not_used  __attribute__ ((aligned (16)));  // Forces alignment of subsequent data on 16 byte boundary
  
  #define MAX_NA 100
  #define CB_SIZE 32 //Cache block size

  static __m128 subMatrixA1[MAX_NA] __attribute__ ((aligned (16)));		// holds 4xn_a submatrix of A (n_a<=100)
  static __m128 subMatrixA2[MAX_NA] __attribute__ ((aligned (16)));		// holds 4xn_a submatrix of A (n_a<=100)
  static __m128 subMatrixA3[MAX_NA] __attribute__ ((aligned (16)));		// holds 4xn_a submatrix of A (n_a<=100)
  static __m128 subMatrixA4[MAX_NA] __attribute__ ((aligned (16)));		// holds 4xn_a submatrix of A (n_a<=100)
  static __m128 subMatrixA5[MAX_NA] __attribute__ ((aligned (16)));		// holds 4xn_a submatrix of A (n_a<=100)
  static __m128 subMatrixA6[MAX_NA] __attribute__ ((aligned (16)));		// holds 4xn_a submatrix of A (n_a<=100)
  static __m128 subMatrixA7[MAX_NA] __attribute__ ((aligned (16)));		// holds 4xn_a submatrix of A (n_a<=100)
  static __m128 subMatrixA8[MAX_NA] __attribute__ ((aligned (16)));		// holds 4xn_a submatrix of A (n_a<=100)
  static __m128 colElemB __attribute__ ((aligned (16)));	
  static __m128 colElemC1 __attribute__ ((aligned (16)));		
  static __m128 colElemC2 __attribute__ ((aligned (16)));		
  static __m128 colElemC3 __attribute__ ((aligned (16)));		
  static __m128 colElemC4 __attribute__ ((aligned (16)));
  static __m128 colElemC5 __attribute__ ((aligned (16)));		
  static __m128 colElemC6 __attribute__ ((aligned (16)));		
  static __m128 colElemC7 __attribute__ ((aligned (16)));		
  static __m128 colElemC8 __attribute__ ((aligned (16)));  
  
  int n = (m_a / CB_SIZE)*CB_SIZE;	// # of main loop iterations
  int r = (m_a % CB_SIZE);	// # of rows of A that are left over from main loop

  register int v;			// declared outside loop so it can be reused
  register int i;
  register int j;
  register int v1;
  register int ixm_a;
  
  omp_set_num_threads(8); //How many threads to execute
  #pragma omp parallel
  {
  // Optimized to process 32 rows of A at a time
  #pragma omp for private(ixm_a,v,i,j,v1,subMatrixA1,subMatrixA2,subMatrixA3,subMatrixA4,subMatrixA5,subMatrixA6,subMatrixA7,subMatrixA8,colElemB,colElemC1,colElemC2,colElemC3,colElemC4,colElemC5,colElemC6,colElemC7,colElemC8)
  for(v = 0; v < n; v += 32){	
	// load next 32xn_a submatrix of A
	// load next column of B
	v1 = v;
	
	for(i = 0; i < n_a; i++){
		ixm_a = (i*m_a)+v1;
		subMatrixA1[i] = _mm_loadu_ps(A+(ixm_a));
		subMatrixA2[i] = _mm_loadu_ps(A+(ixm_a)+4);
		subMatrixA3[i] = _mm_loadu_ps(A+(ixm_a)+8);
		subMatrixA4[i] = _mm_loadu_ps(A+(ixm_a)+12);
		subMatrixA5[i] = _mm_loadu_ps(A+(ixm_a)+16);
		subMatrixA6[i] = _mm_loadu_ps(A+(ixm_a)+20);
		subMatrixA7[i] = _mm_loadu_ps(A+(ixm_a)+24);
		subMatrixA8[i] = _mm_loadu_ps(A+(ixm_a)+28);
	}

	for (i = 0; i < m_a; i++) {
		ixm_a = i*m_a+v1;
		colElemC1 = _mm_setzero_ps();
		colElemC2 = _mm_setzero_ps();
		colElemC3 = _mm_setzero_ps();
		colElemC4 = _mm_setzero_ps();
		colElemC5 = _mm_setzero_ps();
		colElemC6 = _mm_setzero_ps();
		colElemC7 = _mm_setzero_ps();
		colElemC8 = _mm_setzero_ps();
		
		for(j = 0; j < n_a; j++){
			colElemB = _mm_load1_ps(B+(j*m_a)+i);
			colElemC1 = _mm_add_ps(colElemC1, _mm_mul_ps(subMatrixA1[j], colElemB));
			colElemC2 = _mm_add_ps(colElemC2, _mm_mul_ps(subMatrixA2[j], colElemB));
			colElemC3 = _mm_add_ps(colElemC3, _mm_mul_ps(subMatrixA3[j], colElemB));
			colElemC4 = _mm_add_ps(colElemC4, _mm_mul_ps(subMatrixA4[j], colElemB));
			colElemC5 = _mm_add_ps(colElemC5, _mm_mul_ps(subMatrixA5[j], colElemB));
			colElemC6 = _mm_add_ps(colElemC6, _mm_mul_ps(subMatrixA6[j], colElemB));
			colElemC7 = _mm_add_ps(colElemC7, _mm_mul_ps(subMatrixA7[j], colElemB));
			colElemC8 = _mm_add_ps(colElemC8, _mm_mul_ps(subMatrixA8[j], colElemB));
		}
		_mm_storeu_ps(C+(ixm_a), colElemC1);
		_mm_storeu_ps(C+(ixm_a)+4, colElemC2);
		_mm_storeu_ps(C+(ixm_a)+8, colElemC3);
		_mm_storeu_ps(C+(ixm_a)+12, colElemC4);
		_mm_storeu_ps(C+(ixm_a)+16, colElemC5);
		_mm_storeu_ps(C+(ixm_a)+20, colElemC6);
		_mm_storeu_ps(C+(ixm_a)+24, colElemC7);
		_mm_storeu_ps(C+(ixm_a)+28, colElemC8);
	}
  }
  
  if(CB_SIZE > 16 && r > 0){
	v = m_a/32*32;
	
	n = (m_a / 16)*16;	// # of main loop iterations
	r = (m_a % 16);	// # of rows of A that are left over from main loop
	
	// Optimized to process 16 rows of A at a time
	#pragma omp for private(ixm_a,v,i,j,v1,subMatrixA1,subMatrixA2,subMatrixA3,subMatrixA4,colElemB,colElemC1,colElemC2,colElemC3,colElemC4)
	for(v=v; v < n; v += 16){	
		//load next 16xn_a submatrix of A
		//load next column of B
		v1 = v;

		for(i = 0; i < n_a; i++){
			ixm_a = i*m_a+v1;
			subMatrixA1[i] = _mm_loadu_ps(A+(ixm_a));
			subMatrixA2[i] = _mm_loadu_ps(A+(ixm_a)+4);
			subMatrixA3[i] = _mm_loadu_ps(A+(ixm_a)+8);
			subMatrixA4[i] = _mm_loadu_ps(A+(ixm_a)+12);
		}

		for (i = 0; i < m_a; i++) {
			ixm_a = i*m_a+v1;
			colElemC1 = _mm_setzero_ps();
			colElemC2 = _mm_setzero_ps();
			colElemC3 = _mm_setzero_ps();
			colElemC4 = _mm_setzero_ps();
			
			for(j = 0; j < n_a; j++){
				colElemB = _mm_load1_ps(B+(j*m_a)+i);
				colElemC1 = _mm_add_ps(colElemC1, _mm_mul_ps(subMatrixA1[j], colElemB));
				colElemC2 = _mm_add_ps(colElemC2, _mm_mul_ps(subMatrixA2[j], colElemB));
				colElemC3 = _mm_add_ps(colElemC3, _mm_mul_ps(subMatrixA3[j], colElemB));
				colElemC4 = _mm_add_ps(colElemC4, _mm_mul_ps(subMatrixA4[j], colElemB));
			}			
			_mm_storeu_ps(C+(ixm_a), colElemC1);
			_mm_storeu_ps(C+(ixm_a)+4, colElemC2);
			_mm_storeu_ps(C+(ixm_a)+8, colElemC3);
			_mm_storeu_ps(C+(ixm_a)+12, colElemC4);
		}
	}
  }  
  } //End of pragma omp
  
  //process 4 rows at a time
  if(CB_SIZE > 4 && m_a%16 > 0){ 
	v = ((m_a/16)*16);
	n = (m_a / 4)*4;	// # of main loop iterations
	
	for(v=v; v < n; v += 4){	
		//load next 4xn_a submatrix of A
		//load next column of B
		for(i = 0; i < n_a; i++){
			subMatrixA1[i] = _mm_loadu_ps(A+(i*m_a)+v);
		}

		for(i = 0; i < m_a; i++) {
			colElemC1 = _mm_setzero_ps();
			for(j = 0; j < n_a; j++){
				colElemB = _mm_load1_ps(B+(j*m_a)+i);
				colElemC1 = _mm_add_ps(colElemC1, _mm_mul_ps(subMatrixA1[j], colElemB));
			}
			_mm_storeu_ps(C+(i*m_a)+v, colElemC1);
		}
	}
  }
  
  //process rows of A one by one
  if(m_a%4 > 0){ 
    float RowA[n_a];
	v = (m_a/4*4);
	//#pragma omp for private(ixm_a,v, i, j, RowA)
	for(v=v; v < m_a; v++){
		for(i = 0; i < n_a; i++){
			RowA[i] = A[(i*m_a)+v];
		}
		for(i = 0; i < m_a; i++){
			ixm_a = i*m_a;
			for(j = 0; j < n_a; j++){
				C[(ixm_a)+v] += RowA[j]*B[(j*m_a)+i];
			}
		}
	}
  }
}
