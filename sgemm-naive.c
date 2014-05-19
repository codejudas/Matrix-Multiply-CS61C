void sgemm( int m_a, int n_a, float *A, float *B, float *C ) {
  for( int i = 0; i < m_a; i++ ) {
    for( int k = 0; k < n_a; k++ ) {
      for( int j = 0; j < m_a; j++ ) {
	C[i+j*m_a] += A[i+k*m_a] * B[j+k*m_a];
      }
    }
  }
}
