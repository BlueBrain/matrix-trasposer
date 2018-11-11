#pragma once

#include <mpi.h>
#include <cassert>
#include <cstdlib>

template<class T>
class MatrixTransposer
{
   
private:
  struct Metadata
  {
    void * cells;
    int cpu; 
    unsigned int row; 
    unsigned int col; 
    unsigned int cell_count;
  };

  static int QSortCompare (const void * _a, const void * _b)
  {
    Metadata * a = static_cast<Metadata*> ((void*)_a);
    Metadata * b = static_cast<Metadata*> ((void*)_b);
    if (a->cpu != b->cpu) return ( a->cpu - b->cpu );
    if (a->row != b->row) return ( a->row - b->row );
    assert(a->col != b->col);
    return ( a->col - b->col );
  }

  static void SortCellsByCpuRowColumn(unsigned int col_count, unsigned int cell_count, Metadata * metadatas, T*& cells)
  {
    qsort(metadatas,col_count, sizeof(Metadata), QSortCompare);
 
    //now that the qsort ir ordered correctly, we will shuffle the elements 
    //of cells to follow the same order (by looking at the pointers of the structure data)
    unsigned long long cell_id=0;
    T * cells_temp = new T[cell_count];
    for (unsigned int c=0; c<col_count; c++)
    {
        //copy column's cells
        void * firstCell = metadatas[c].cells;
        memcpy(&(cells_temp[cell_id]), firstCell, sizeof(T)*metadatas[c].cell_count);
        
        cell_id += metadatas[c].cell_count;
    }

    delete [] cells; cells=NULL;
    cells = cells_temp;
  }  
  
public:

  /// Use-case 1: no pre-computed rank end offsets
  static void Transpose(int row_count, T *& cells, unsigned int *& counts, unsigned int *& displs, unsigned int *& cell_counts, MPI_Comm mpiComm)
  {
      int mpi_size=-1;
      MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
      unsigned int * rank_row_counts = new unsigned int[mpi_size];
      unsigned int * rank_end_offsets = new unsigned int[mpi_size];
      MPI_Allgather(&row_count, 1, MPI_UNSIGNED, rank_row_counts, 1, MPI_UNSIGNED, MPI_COMM_WORLD);
      for (int i=0; i<mpi_size; i++)
        rank_end_offsets[i] = i==0 ? rank_row_counts[i]-1 : rank_row_counts[i] + rank_end_offsets[i-1];
      Transpose(rank_end_offsets, cells, counts, displs, cell_counts, mpiComm);
      delete [] rank_row_counts;
      delete [] rank_end_offsets;
  }

  /// Use-case 2: with pre-computed rank end offsets
  static void Transpose(unsigned int * rank_end_offsets, T *& cells, unsigned int *& counts, unsigned int *& displs, unsigned int *& cell_counts, MPI_Comm mpiComm)
  {
    //MPI variables
    int mpi_size=-1, mpi_rank=-1;;
    MPI_Comm_rank(mpiComm, &mpi_rank);
    MPI_Comm_size(mpiComm, &mpi_size);

    unsigned int my_first_row_id = mpi_rank==0 ? 0 : rank_end_offsets[mpi_rank-1]+1;
    unsigned int my_row_count = rank_end_offsets[mpi_rank] - (mpi_rank==0 ? -1 : rank_end_offsets[mpi_rank-1]);

    //get total number of cells and columns
    unsigned int my_col_count=0;
    for (unsigned int r=0; r<my_row_count; r++)
    my_col_count+=counts[r];

    //zero step: replace a matrix into 'mpiSize' sub-matrices transposed
    Metadata * metadatas = new Metadata[my_col_count];

    //we will divide this L lines in 'mpiSize' matrices, and transpose each
    //one of them. For that we perform a sorting based on destinationCPU, then
    //column Id (since its the row for transposed matrix), and then on row
    //(transposed matrix column)

    //keeps the map of which cpu takes each neuron
    unsigned int total_row_count=0;
    MPI_Allreduce( &my_row_count, &total_row_count, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);

    unsigned int * cpu_per_row_id = new unsigned int[total_row_count];
    unsigned int cpu_id=0;
    for (unsigned int n=0; n< (unsigned int) total_row_count; n++)
    {
        assert(cpu_id < (unsigned int) mpi_size);

        cpu_per_row_id[n] = cpu_id;
        while (cpu_id < static_cast<unsigned int>(mpi_size) && rank_end_offsets[cpu_id] <= n){
            cpu_id++;
        }
    }

    //converting the Sparse matrix arrays into a qsort wrapper structure
    unsigned int cell_id=0, col_id=0, total_cell_count=0;
    for (unsigned int rowId=0; rowId<my_row_count; rowId++)
    {
    for (unsigned int c=0; c<counts[rowId]; c++)
    {
      metadatas[col_id].cells = &(cells[cell_id]);
          metadatas[col_id].cell_count = cell_counts[col_id];

      total_cell_count += cell_counts[col_id];

          //we intentionally swapped row and column, to force
          //the qsort to sort them like this (therefor transposing it)
      unsigned int & columnId = displs[col_id];
      metadatas[col_id].row = columnId;
      metadatas[col_id].col = my_first_row_id + rowId;
          metadatas[col_id].cpu = cpu_per_row_id[columnId];

      //increment cell and column Id
      cell_id += cell_counts[col_id];
          col_id++;
    }
    }

    delete [] cpu_per_row_id; cpu_per_row_id=NULL;
    delete [] displs; displs=NULL;
    delete [] cell_counts; cell_counts=NULL;
    delete [] counts; counts=NULL;

    SortCellsByCpuRowColumn(my_col_count, total_cell_count, metadatas, cells);

   //2nd step: now we will send the respective table info to each CPU (qsortStruct)
    int * sendcounts = new int[mpi_size];
    int * recvcounts = new int [mpi_size];
    int * sentdispls = new int[mpi_size];
    int * recvdispls = new int[mpi_size];

    for (int cpu=0; cpu<mpi_size; cpu++)
    sendcounts[cpu]=0;

    for (unsigned int c=0; c<my_col_count; c++)
    {
    int & cpu = metadatas[c].cpu;
        sendcounts[cpu] += sizeof(Metadata);
    }

    //share the amount of data to be received by each other cpu
    MPI_Alltoall(sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT, mpiComm);

    //calculate offset for the data sent/received
    for (int cpu=0; cpu<mpi_size; cpu++)
    sentdispls[cpu] = cpu==0 ? 0 : sentdispls[cpu-1] + sendcounts[cpu-1];

    for (int cpu=0; cpu < mpi_size; cpu++)
        recvdispls[cpu] = cpu==0 ? 0 : recvdispls[cpu-1] + recvcounts[cpu - 1];

    //calculate total data size to be received
    unsigned int my_col_count_T = (recvdispls[mpi_size - 1] + recvcounts[mpi_size - 1])/sizeof(Metadata);
    Metadata * metadatas_T = new Metadata[my_col_count_T];

    //send around the table structure of the elements to be received
    MPI_Alltoallv(metadatas, sendcounts, sentdispls, MPI_BYTE, metadatas_T, recvcounts, recvdispls, MPI_BYTE, mpiComm);

    //3rd step: now we will send the respective cells to each CPU
    for (int cpu=0; cpu<mpi_size; cpu++)
    sendcounts[cpu]=0;

    for (unsigned int c=0; c<my_col_count; c++)
    {
    int & cpu = metadatas[c].cpu;
        sendcounts[cpu] += metadatas[c].cell_count * sizeof(T);
    }

    delete [] metadatas; metadatas=NULL;

    //share the amount of data to be received by each other cpu
    MPI_Alltoall(sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT, mpiComm);

    //calculate offset for the data sent/received
    for (int cpu=0; cpu<mpi_size; cpu++)
    sentdispls[cpu] = cpu==0 ? 0 : sentdispls[cpu-1] + sendcounts[cpu-1];

    for (int cpu=0; cpu<mpi_size; cpu++)
        recvdispls[cpu] = cpu==0 ? 0 : recvdispls[cpu-1] + recvcounts[cpu - 1];

    //receives data
    unsigned int total_cell_count_T = (recvdispls[mpi_size-1] + recvcounts[mpi_size-1])/sizeof(T);
    T * cells_T = new T[total_cell_count_T];
    MPI_Alltoallv(cells, sendcounts, sentdispls, MPI_BYTE, cells_T, recvcounts, recvdispls, MPI_BYTE, mpiComm);

    delete [] cells; cells=NULL;
    delete [] sendcounts; sendcounts = NULL;
    delete [] recvcounts; recvcounts = NULL;
    delete [] sentdispls; sentdispls=NULL;
    delete [] recvdispls; recvdispls=NULL;

    //4th we will reconvert the multiple transposed matrices, into a single matrix
    cells = cells_T;
    metadatas = metadatas_T;
    my_col_count = my_col_count_T;
    total_cell_count = total_cell_count_T;

    //we set the pointers to the correct address of cells
    cell_id = 0;
    for (unsigned int c = 0; c < my_col_count; c++)
    {
        metadatas[c].cells = &(cells[cell_id]);
        cell_id += metadatas[c].cell_count;

    //make sure that all columns received were meant for this cpu
    assert(metadatas[c].cpu == mpi_rank);
    }

    //we sort by row, not by cpu (therefore converting into a single sparse matrix)
    SortCellsByCpuRowColumn(my_col_count, total_cell_count, metadatas, cells);

    //set final data structures: convert qsort wrapper to sparse matrix arrays
    displs = new unsigned int[my_col_count];
    counts = new unsigned int[my_row_count];
    cell_counts = new unsigned int [my_col_count];

    for (unsigned int r=0; r<my_row_count; r++)
    counts[r]=0;

    for (unsigned int c = 0; c < my_col_count; c++)
    {
    unsigned int & row = metadatas[c].row;
    counts[row - my_first_row_id]++;

    displs[c]=metadatas[c].col;
    cell_counts[c]= metadatas[c].cell_count;
    }

    delete [] metadatas;
  }
};

