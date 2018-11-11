# Distributed Sparse Matrix Transposer
## Usage
- If rank_offsets was pre-computed:
```MatrixTransposer<T>::Transpose(rank_end_offsets, cells, counts, displs, cell_counts, mpicomm);```
- If rank offsets is unknown:
```MatrixTransposer<T>::Transpose(row_count, cells, counts, displs, cell_counts, mpicomm);```

