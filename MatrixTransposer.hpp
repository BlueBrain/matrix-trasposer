#pragma once

#include <mpi.h>
#include <cassert>
#include <cstdlib>

template <class T>
class MatrixTransposer {
 private:
  struct Metadata {
    void* cells;
    int rank;
    unsigned int row;
    unsigned int col;
    unsigned int cell_count;
  };

  static int QSortCompare(const void* md_a, const void* md_b) {
    Metadata* a = static_cast<Metadata*>((void*)md_a);
    Metadata* b = static_cast<Metadata*>((void*)md_b);
    if (a->rank != b->rank) return (a->rank - b->rank);
    if (a->row != b->row) return (a->row - b->row);
    assert(a->col != b->col);
    return (a->col - b->col);
  }

  static void SortCellsByCpuRowColumn(unsigned int col_count,
                                      unsigned int cell_count,
                                      Metadata* metadatas, T*& cells) {
    qsort(metadatas, col_count, sizeof(Metadata), QSortCompare);

    /* now that the qsort is ordered correctly, we will shuffle the elements
       of cells to follow the same order (by looking at the pointers of the
       structure data) */
    unsigned long long cell_id = 0;
    T* cells_temp = new T[cell_count];
    for (unsigned int c = 0; c < col_count; c++) {
      // copy column's cells
      void* firstCell = metadatas[c].cells;
      memcpy(&(cells_temp[cell_id]), firstCell,
             sizeof(T) * metadatas[c].cell_count);

      cell_id += metadatas[c].cell_count;
    }

    delete[] cells;
    cells = NULL;
    cells = cells_temp;
  }

 public:
  /// Use-case 1: no pre-computed rank end offsets
  static void Transpose(int row_count, T*& cells, unsigned int*& counts,
                        unsigned int*& displs, unsigned int*& cell_counts,
                        MPI_Comm comm) {
    int mpi_size = -1;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    unsigned int* rank_row_counts = new unsigned int[mpi_size];
    unsigned int* rank_end_offsets = new unsigned int[mpi_size];
    MPI_Allgather(&row_count, 1, MPI_UNSIGNED, rank_row_counts, 1, MPI_UNSIGNED,
                  MPI_COMM_WORLD);
    for (int i = 0; i < mpi_size; i++)
      rank_end_offsets[i] = i == 0
                                ? rank_row_counts[i] - 1
                                : rank_row_counts[i] + rank_end_offsets[i - 1];
    Transpose(rank_end_offsets, cells, counts, displs, cell_counts, comm);
    delete[] rank_row_counts;
    delete[] rank_end_offsets;
  }

  /// Use-case 2: with pre-computed rank end offsets
  static void Transpose(unsigned int* rank_end_offsets, T*& cells,
                        unsigned int*& counts, unsigned int*& displs,
                        unsigned int*& cell_counts, MPI_Comm comm) {
    // MPI variables
    int mpi_size = -1, mpi_rank = -1;
    MPI_Comm_rank(comm, &mpi_rank);
    MPI_Comm_size(comm, &mpi_size);

    unsigned int my_first_row_id =
        mpi_rank == 0 ? 0 : rank_end_offsets[mpi_rank - 1] + 1;
    unsigned int my_row_count =
        rank_end_offsets[mpi_rank] -
        (mpi_rank == 0 ? -1 : rank_end_offsets[mpi_rank - 1]);

    // get total number of cells and columns
    unsigned int my_col_count = 0;
    for (unsigned int r = 0; r < my_row_count; r++) my_col_count += counts[r];

    // local transpose
    Metadata* metadatas = new Metadata[my_col_count];

    /* we divide this L lines in 'mpi_size' matrices, and transpose each
       them: we perform a sorting based on target rank, then col id (since its
       the row for transposed matrix), and then on row (transposed column)*/

    unsigned int total_row_count = rank_end_offsets[mpi_size - 1];
#ifndef NDEBUG  // TODO delete (debug step only)
    unsigned int total_row_count_1 = 0;
    MPI_Allreduce(&my_row_count, &total_row_count_1, 1, MPI_UNSIGNED, MPI_SUM,
                  MPI_COMM_WORLD);
    assert(total_row_count == total_row_count_1);
#endif

    // builds a map of which rank is assinged to each row
    unsigned int* rank_per_row_id = new unsigned int[total_row_count];
    unsigned int rank_id = 0;
    for (unsigned int n = 0; n < total_row_count; n++) {
      assert(rank_id < (unsigned int)mpi_size);
      rank_per_row_id[n] = rank_id;
      while (rank_id < static_cast<unsigned int>(mpi_size) &&
             rank_end_offsets[rank_id] <= n)
        rank_id++;
    }

    // create metadata structures
    unsigned int cell_id = 0, col_id = 0, total_cell_count = 0;
    for (unsigned int rowId = 0; rowId < my_row_count; rowId++) {
      for (unsigned int c = 0; c < counts[rowId]; c++) {
        metadatas[col_id].cells = &(cells[cell_id]);
        metadatas[col_id].cell_count = cell_counts[col_id];

        total_cell_count += cell_counts[col_id];

        // we swap row and column, to force qsort to transpose when sorting
        unsigned int& columnId = displs[col_id];
        metadatas[col_id].row = columnId;
        metadatas[col_id].col = my_first_row_id + rowId;
        metadatas[col_id].rank = rank_per_row_id[columnId];
        cell_id += cell_counts[col_id];
        col_id++;
      }
    }

    delete[] rank_per_row_id;
    rank_per_row_id = NULL;
    delete[] displs;
    displs = NULL;
    delete[] cell_counts;
    cell_counts = NULL;
    delete[] counts;
    counts = NULL;

    SortCellsByCpuRowColumn(my_col_count, total_cell_count, metadatas, cells);

    // View Swap step 1/2: metadata matrix transose
    int* sendcounts = new int[mpi_size]();
    int* recvcounts = new int[mpi_size]();
    int* sentdispls = new int[mpi_size]();
    int* recvdispls = new int[mpi_size]();

#ifndef NDEBUG  // TODO delete
    for (int r = 0; r < mpi_size; r++) assert(sendcounts[r]) == 0;
#endif

    for (unsigned int c = 0; c < my_col_count; c++)
      sendcounts[metadatas[c].rank] += sizeof(Metadata);

    // exchange metadata sizes to be received by each rank
    MPI_Alltoall(sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT, comm);

    // calculate offset for the data sent/received
    for (int r = 0; r < mpi_size; r++)
      sentdispls[r] = r == 0 ? 0 : sentdispls[r - 1] + sendcounts[r - 1];

    for (int r = 0; r < mpi_size; r++)
      recvdispls[r] = r == 0 ? 0 : recvdispls[r - 1] + recvcounts[r - 1];

    // calculate total data size to be received
    unsigned int my_col_count_T =
        (recvdispls[mpi_size - 1] + recvcounts[mpi_size - 1]) /
        sizeof(Metadata);
    Metadata* metadatas_T = new Metadata[my_col_count_T];

    // exchange metadata structures
    MPI_Alltoallv(metadatas, sendcounts, sentdispls, MPI_BYTE, metadatas_T,
                  recvcounts, recvdispls, MPI_BYTE, comm);

    // View Swap step 2/2: elements exchange
    for (int r = 0; r < mpi_size; r++) sendcounts[r] = 0;

    for (unsigned int c = 0; c < my_col_count; c++)
      sendcounts[metadatas[c].rank] += metadatas[c].cell_count * sizeof(T);

    delete[] metadatas;
    metadatas = NULL;

    // exchange element sizes to be received by each rank
    MPI_Alltoall(sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT, comm);

    // calculate offset for the data sent/received
    for (int r = 0; r < mpi_size; r++)
      sentdispls[r] = r == 0 ? 0 : sentdispls[r - 1] + sendcounts[r - 1];

    for (int r = 0; r < mpi_size; r++)
      recvdispls[r] = r == 0 ? 0 : recvdispls[r - 1] + recvcounts[r - 1];

    // exchange elements
    unsigned int total_cell_count_T =
        (recvdispls[mpi_size - 1] + recvcounts[mpi_size - 1]) / sizeof(T);
    T* cells_T = new T[total_cell_count_T];
    MPI_Alltoallv(cells, sendcounts, sentdispls, MPI_BYTE, cells_T, recvcounts,
                  recvdispls, MPI_BYTE, comm);

    delete[] cells;
    cells = NULL;
    delete[] sendcounts;
    sendcounts = NULL;
    delete[] recvcounts;
    recvcounts = NULL;
    delete[] sentdispls;
    sentdispls = NULL;
    delete[] recvdispls;
    recvdispls = NULL;

    // 4th we will reconvert the multiple transposed matrices, into a single
    // matrix
    cells = cells_T;
    metadatas = metadatas_T;
    my_col_count = my_col_count_T;
    total_cell_count = total_cell_count_T;

    // we set the pointers to the correct address of cells
    cell_id = 0;
    for (unsigned int c = 0; c < my_col_count; c++) {
      metadatas[c].cells = &(cells[cell_id]);
      cell_id += metadatas[c].cell_count;

      // make sure that all columns received were meant for this rank
      assert(metadatas[c].rank == mpi_rank);
    }

    // we sort by row, not by rank, therefore converting into single matrix
    SortCellsByCpuRowColumn(my_col_count, total_cell_count, metadatas, cells);

    // final data structures: use metadata to retrieve sparse matrix arrays
    displs = new unsigned int[my_col_count];
    counts = new unsigned int[my_row_count];
    cell_counts = new unsigned int[my_col_count];

    for (unsigned int r = 0; r < my_row_count; r++) counts[r] = 0;

    for (unsigned int c = 0; c < my_col_count; c++) {
      unsigned int& row = metadatas[c].row;
      counts[row - my_first_row_id]++;

      displs[c] = metadatas[c].col;
      cell_counts[c] = metadatas[c].cell_count;
    }

    delete[] metadatas;
  }
};
