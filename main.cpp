#include "MatrixTransposer.hpp"

#include <time.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>

/**
 * @brief GenerateRandomNumber returns a random value in interval [min,max)
 * @param min minimum random value
 * @param max maximum random value
 * @return
 */
int GenerateRandomNumber(int min, int max) {
  return rand() % (max - min) + min;
}

/**
 * @brief main Generates a random sparse matrix and transposes it
 * @return
 */
int main(int argv, char** argc) {
  // MPI and random seed init
  int mpi_size = -1, mpi_rank = -1;
  MPI_Init(&argv, &argc);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  srand(time(NULL) / (mpi_rank + 1));

  unsigned int row_count = GenerateRandomNumber(3, 4);

  // rank end offsets and first row id of this rank
  unsigned int* rank_row_counts = new unsigned int[mpi_size];
  MPI_Allgather(&row_count, 1, MPI_UNSIGNED, rank_row_counts, 1, MPI_UNSIGNED,
                MPI_COMM_WORLD);

  unsigned int* rank_end_offsets = new unsigned int[mpi_size];
  for (int r = 0; r < mpi_size; r++)
    rank_end_offsets[r] = r == 0 ? rank_row_counts[r] - 1
                                 : rank_row_counts[r] + rank_end_offsets[r - 1];
  unsigned int first_row_id =
      mpi_rank == 0 ? 0 : rank_end_offsets[mpi_rank - 1] + 1;
  unsigned int total_row_count = rank_end_offsets[mpi_size - 1]+1;

  unsigned int* counts = new unsigned int[row_count];

  long long col_count = 0, total_col_count=0;
  for (unsigned int i = 0; i < row_count; i++) {
    // counts[i] = GenerateRandomNumber (1, row_count);
    counts[i] = 2;
    col_count += counts[i];
  }

  unsigned int* displs = new unsigned int[col_count];
  unsigned int* cell_counts = new unsigned int[col_count];
  long long col_id = 0, cell_count = 0, total_cell_count = 0;
  for (unsigned int i = 0; i < row_count; i++) {
    for (unsigned int j = 0; j < counts[i]; j++) {
      displs[col_id] = total_row_count * j / counts[i];
      assert(displs[col_id] < total_row_count);
      cell_counts[col_id] = 1;
      // cell_count[col_id]= GenerateRandomNumber(2,20);
      cell_count += cell_counts[col_id];
      col_id++;
    }
  }

  MPI_Allreduce(&col_count, &total_col_count, 1, MPI_LONG_LONG, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(&cell_count, &total_cell_count, 1, MPI_LONG_LONG, MPI_SUM,
                MPI_COMM_WORLD);
  if (mpi_rank == 0) printf("mpi size: %d; row count: %d; column count:%lld; cell count: %lld\n",
                             mpi_size, total_row_count, total_col_count, total_cell_count);

  int* cells = new int[cell_count];
  for (unsigned int i = 0; i < cell_count; i++)
    cells[i] = GenerateRandomNumber(0, 8);

#ifndef NDEBUG
  col_id = 0;
  unsigned long long cell_id = 0;
  for (int i = 0; i < mpi_size; i++) {
    if (mpi_rank == i) {
      fprintf(stderr, "mpi rank: %d\n", mpi_rank);
      for (unsigned int r = 0; r < row_count; r++) {
        fprintf(stderr, "row: %d (%d cols) ::", r + first_row_id, counts[r]);
        for (unsigned int c = 0; c < counts[r]; c++) {
          fprintf(stderr, "[id %d, %d cell] ", displs[col_id],
                  cell_counts[col_id]);
          for (unsigned int e = 0; e < cell_counts[col_id]; e++)
            fprintf(stderr, "%d ", cells[cell_id++]);
          col_id++;
          fprintf(stderr, ", ");
        }
        fprintf(stderr, "\n");
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
#endif
  MPI_Barrier(MPI_COMM_WORLD);
  clock_t t_start = clock();
  MatrixTransposer<int>::Transpose(rank_end_offsets, cells, counts, displs,
                                   cell_counts, MPI_COMM_WORLD);
  if (mpi_rank == 0)
    fprintf(stderr, "Finished. Time taken: %.2fs\n",
            (double)(clock() - t_start) / CLOCKS_PER_SEC);
  MPI_Barrier(MPI_COMM_WORLD);

#ifndef NDEBUG
  col_id = 0, cell_id = 0;
  for (int i = 0; i < mpi_size; i++) {
    if (mpi_rank == i) {
      fprintf(stderr, "rank %d\n", mpi_rank);
      for (unsigned int r = 0; r < row_count; r++) {
        fprintf(stderr, "row %d (%d cols) ::", r + first_row_id, counts[r]);
        for (unsigned int c = 0; c < counts[r]; c++) {
          fprintf(stderr, "[id %d, %d cell] ", displs[col_id],
                  cell_counts[col_id]);
          for (unsigned int e = 0; e < cell_counts[col_id]; e++)
            fprintf(stderr, "%d ", cells[cell_id++]);
          col_id++;
          fprintf(stderr, ", ");
        }
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    fprintf(stderr, "\n");
  }
#endif

  return 0;
}
