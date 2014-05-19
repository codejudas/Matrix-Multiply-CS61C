#include <nmmintrin.h> 

void sgemm( int m_a, int n_a, float *A, float *B, float *C ) {
  __m128 partialRowB1,partialRowB2,partialRowB3;
  __m128 partialSumC1,partialSumC2,partialSumC3;
  __m128 partialColA1,partialColA2,partialColA3,partialColA4,partialColA5,partialColA6,partialColA7,partialColA8,partialColA9;
 
  if(m_a%36 == 0 && n_a%36 == 0){ //Optimized for 36x36 matrix
	  for( int v = 0; v < m_a; v++){
		  int j;
		  //Break into 12x12 matrices
		  for( j = 0; j < n_a/3*3; j+= 3 ) {
			//load next 3 elems of row B
			partialRowB1 = _mm_load1_ps(B+(v+m_a*j)); 
			partialRowB2 = _mm_load1_ps(B+(v+m_a*(j+1)));
			partialRowB3 = _mm_load1_ps(B+(v+m_a*(j+2)));
			int i;
			for( i = 0; i < m_a/12*12; i += 12 ) {
			  //Load 12 elems from column 1 of A
			  partialColA1 = _mm_loadu_ps(A+(i+m_a*j)); 
			  partialColA2 = _mm_loadu_ps(A+(i+m_a*j+4));
			  partialColA3 = _mm_loadu_ps(A+(i+m_a*j+8));
			  //load 12 elems from column 2 of A
			  partialColA4 = _mm_loadu_ps(A+(i+m_a*(j+1)));
			  partialColA5 = _mm_loadu_ps(A+(i+m_a*(j+1)+4));
			  partialColA6 = _mm_loadu_ps(A+(i+m_a*(j+1)+8));
			  //load 12 elems from column 3 of A
			  partialColA7 = _mm_loadu_ps(A+(i+m_a*(j+2)));
			  partialColA8 = _mm_loadu_ps(A+(i+m_a*(j+2)+4));
			  partialColA9 = _mm_loadu_ps(A+(i+m_a*(j+2)+8));
			  
			  //Multiply first col of A with first elem of B & sum into respective elem of C
			  partialSumC1 = _mm_add_ps(_mm_loadu_ps(C+(i+v*m_a)), _mm_mul_ps(partialColA1, partialRowB1));
			  partialSumC2 = _mm_add_ps(_mm_loadu_ps(C+(i+v*m_a+4)), _mm_mul_ps(partialColA2, partialRowB1));
			  partialSumC3 = _mm_add_ps(_mm_loadu_ps(C+(i+v*m_a+8)), _mm_mul_ps(partialColA3, partialRowB1));
			  
			  //Multiply second col of A with second elem of B & sum into respective elem of C
			  partialSumC1 = _mm_add_ps(partialSumC1, _mm_mul_ps(partialColA4, partialRowB2));
			  partialSumC2 = _mm_add_ps(partialSumC2, _mm_mul_ps(partialColA5, partialRowB2));
			  partialSumC3 = _mm_add_ps(partialSumC3, _mm_mul_ps(partialColA6, partialRowB2));
			  
			  //Multiply last col of A with last elem of B & sum into respective C & store
			  _mm_storeu_ps(C+i+v*m_a, _mm_add_ps(partialSumC1, _mm_mul_ps(partialColA7, partialRowB3)));
			  _mm_storeu_ps(C+i+v*m_a+4, _mm_add_ps(partialSumC2, _mm_mul_ps(partialColA8, partialRowB3)));
			  _mm_storeu_ps(C+i+v*m_a+8, _mm_add_ps(partialSumC3, _mm_mul_ps(partialColA9, partialRowB3)));
			}
		  }
	  }
   }
   //Handles matrices of size other than 36x36
   else{
	  for(int v = 0; v < n_a; v++){ //goes through output column in C
		int m_axv = m_a*v;
		for( int j = 0; j < m_a; j++ ) {
			int jxm_a = j*m_a;
			partialRowB1 = _mm_load1_ps(B+(j+m_axv)); //load current elem of row in B
			int i;
			for( i = 0; i < m_a/16*16; i += 16 ) {
			  //load next 16 col elems of A into packed sp
			  partialColA1 = _mm_loadu_ps(A+(i+m_axv)); 
			  partialColA2 = _mm_loadu_ps(A+(i+m_axv+4));
			  partialColA3 = _mm_loadu_ps(A+(i+m_axv+8));
			  partialColA4 = _mm_loadu_ps(A+(i+m_axv+12));
			  
			  //Compute part of elem in C, store in C
			  _mm_storeu_ps((C + i + jxm_a), _mm_add_ps(_mm_loadu_ps(C+i+jxm_a), _mm_mul_ps(partialColA1, partialRowB1)));
			  _mm_storeu_ps((C + i + jxm_a+4), _mm_add_ps(_mm_loadu_ps(C+i+jxm_a+4), _mm_mul_ps(partialColA2, partialRowB1)));
			  _mm_storeu_ps((C + i + jxm_a+8), _mm_add_ps(_mm_loadu_ps(C+i+jxm_a+8), _mm_mul_ps(partialColA3, partialRowB1)));
			  _mm_storeu_ps((C + i + jxm_a+12), _mm_add_ps(_mm_loadu_ps(C+i+jxm_a+12), _mm_mul_ps(partialColA4, partialRowB1)));
			}
			//fringe case
			for( i = m_a/16*16; i < m_a/4*4; i += 4 ) {
			  partialColA1 = _mm_loadu_ps(A+(i+m_axv)); //load next 4 col elems of A into packed sp
			  _mm_storeu_ps((C + i + jxm_a), _mm_add_ps(_mm_loadu_ps(C+i+jxm_a), _mm_mul_ps(partialColA1, partialRowB1)));
			}
			//finish off matrices with dimensions %4 != 0
			for( i = m_a/4*4; i < m_a; i++){
				C[i + jxm_a] += A[i+m_axv] * B[j+m_axv];
			}
		  }
	  }
	}
}