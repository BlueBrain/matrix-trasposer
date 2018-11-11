#include <mpi.h>
#include <numeric> //std::accumulate
#include <map>
#include <vector>
#include <tuple>

using namespace std;

/// Row id, Column id, and number of elements in cell
typedef std::tuple<int,int,int> Metadata;

template <typename T>
void LocalTranspose(int & row_count, T *& cells, int *& counts,
                    int *& displs, int *& cell_counts, MPI_Comm comm)
{
  // build map of row -> col -> (cells + cell count)
  map<int, map<int,vector<T*> > > matrix_T;
  int c=0;
  T* cell_it = cells;
  for (int i=0; i<row_count;i++)
    for (int j=0; j<counts[i]; j++, cell_it += cell_counts[c], c++)
      matrix_T[displs[c]][i] = vector<int>(cell_it, cell_it + cell_counts[c]);

  // new number of rows, same cols, same cells
  delete [] counts;
  row_count = matrix_T.size();
  counts = new int[row_count]();

  //final transposed data layout
  int row_it=0, col_it=0;
  cell_it=cells;
  for (auto & row=matrix_T.begin(); row!=matrix_T.end(); row++, row_it++)
    for (auto & col = row.begin(); col != row.end(); col++, col_it++)
    {
      counts[row_it]++;
      displs[col_it]=col.first;
      cell_counts[col_it] = col.second.size();
      memcpy(&cells[cell_it], col.second.data(), sizeof(T)*cell_counts[col_it]);
      cell_it += cell_counts;
    }
}

int* GetRankOffsets(int row_count, int mpi_size, MPI_Comm comm)
{
  int *rank_offsets = new int[mpi_size+1]();
  int *row_counts   = new int[mpi_size]();
  MPI_Allgather( &row_count, 1, MPI_INT, row_counts, 1, MPI_INT, comm);
  for (int r=1; r<=mpi_size; r++)
    rank_offsets[r] = rank_offsets[r-1] + row_counts[r-1];
  delete [] row_counts;
  return rank_offsets;
}


template <typename T>
void MetadataTranspose(int row_count, T * cells, int *counts,
                       int *displs, int *cell_counts, MPI_Comm comm,
                       Metadata *& metadata_T, int & col_count_T)
{
  // SET-UP: get offset of ranks final id, and network size
  int mpi_size=-1, mpi_rank=-1;
  MPI_Comm_size(comm, &mpi_size);

  int *rank_offsets = GetRankOffsets(row_count, mpi_size, comm);

  // all-to-all auxiliary data structs
  int *sendcounts = new int[mpi_size] ();
  int *recvcounts = new int[mpi_size] ();
  int *sdispls    = new int[mpi_size] ();
  int *rdispls    = new int[mpi_size] ();

  // STEP 1: Transposition of Matrix holding matrix metadata (row, column, count)
  int first_row = mpi_rank==0 ? 0 : rank_offsets[mpi_rank-1]+1;

  int col_count = std::accumulate(counts, counts+row_count,0);
  Metadata *metadata = new Metadata[col_count];

  int col=0, rank=0;
  for (int i=0; i<row_count; i++, rank=0)
    for (int j=0; j<counts[i]; j++, col++)
    {
      for (; displs[col] < rank_offsets[rank]; rank++);
      sendcounts[rank] += sizeof (metadata);
      metadata[col] = make_tuple(first_row+i, displs[col], cell_counts[col]);
    }
  MPI_Alltoall(sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT, comm);

  // STEP 2: Transposition of Matrix holding sparse metadata

  for (int r=0; r<mpi_size; r++)
   sdispls[r] = r==0 ? 0 : sdispls[r-1] + sendcounts[r-1];

  for (int r=0; r<mpi_size; r++)
   rdispls[r] = r==0 ? 0 : rdispls[r-1] + recvcounts[r-1];

  col_count_T = (rdispls[mpi_size-1] + recvcounts[mpi_size-1])
                    / sizeof(Metadata);

  metadata_T = new Metadata[col_count_T];
  MPI_Alltoallv(metadata,   sendcounts, sdispls, MPI_BYTE,
                metadata_T, recvcounts, rdispls, MPI_BYTE, comm);

  //[...] clean-up
}


template <typename T>
Metadata* ViewSwap(int row_count, T *& cells, int *& counts,
                   int *& displs, int *& cell_counts, MPI_Comm comm)
{
  int mpi_size=-1, mpi_rank=-1;
  MPI_Comm_size(comm, &mpi_size);
  MPI_Comm_rank(comm, &mpi_rank);
  int *rank_offsets = GetRankOffsets(row_count, mpi_size, comm);

  int col_count_T;
  Metadata *metadata_T = nullptr;
  MetadataTranspose(row_count, cells, counts, displs, cell_counts, comm,
                    metadata_T, col_count_T);

  // all-to-all auxiliary data structs
  int *sendcounts = new int[mpi_size] ();
  int *recvcounts = new int[mpi_size] ();
  int *sdispls    = new int[mpi_size] ();
  int *rdispls    = new int[mpi_size] ();

  //Transposition of Matrix holding counts of elements
  int col=0, rank=0;
  for (int i=0; i<row_count; i++, rank=0)
    for (int j=0; j<counts[i]; j++, col++)
    {
      for (; displs[col] < rank_offsets[rank]; rank++);
      sendcounts[rank] += std::get<2>(metadata_T[col]) * sizeof(T);
    }

  MPI_Alltoall(sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT, comm);

  for (int r=0; r<mpi_size; r++)
    sdispls[r] = r==0 ? 0 : sdispls[r-1] + sendcounts[r-1];

  for (int r=0; r<mpi_size; r++)
    rdispls[r] = r==0 ? 0 : rdispls[r-1] + recvcounts[r-1];

  //Transposition of Matrix the elements
  int cell_count_T= (rdispls[mpi_size-1]+recvcounts[mpi_size-1])/sizeof(T);
  T* cells_T = new T[cell_count_T];
  MPI_Alltoallv(cells,   sendcounts, sdispls, MPI_BYTE,
                cells_T, recvcounts, rdispls, MPI_BYTE, comm);

  //convert multiple transposed matrices into a single matrix

  //Clean-up and in-place assignment
  delete [] cells; cells=NULL;
  delete [] sendcounts; sendcounts = NULL;
  delete [] recvcounts; recvcounts = NULL;
  delete [] sdispls; sdispls=NULL;
  delete [] rdispls; rdispls=NULL;

  cells = cells_T;
  counts = new int[row_count] ();
  displs  = new int[col_count_T] ();
  cell_counts = new int[col_count_T] ();

  //point wrappers to correct offset based on received cells
  int cell = 0;
  for (int c = 0; c < col_count_T; c++)
  {
    Metadata & md = metadata_T[c];
    counts[ std::get<0>(md)]++;
    displs[c] = std::get<1>(md);
    cell_counts[c] = std::get<2>(md);
  }

  //TOOD missin RearrangeWrappedCells ,,,,
  delete [] rank_offsets;
  delete [] metadata_T;
}

template<typename T>
int DistTranspose( int row_count, T *& cells, int *& counts,
                   int *& displs, int *& cell_counts, MPI_Comm comm)
{
  LocalTranspose (row_count, cells, counts, displs, cell_counts, comm);
  ViewSwap (row_count, cells, counts, displs, cell_counts, comm);
  return MPI_SUCCESS;
}
