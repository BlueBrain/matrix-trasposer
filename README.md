# Distributed Sparse Matrix Transposer
## Usage
- If rank offsets is known:
`MatrixTransposer<T>::Transpose(rank_end_offsets, cells, counts, displs, cell_counts, comm);`
- If rank offsets is unknown:
`MatrixTransposer<T>::Transpose(row_count, cells, counts, displs, cell_counts, comm);`

For an example see `main.cpp`
