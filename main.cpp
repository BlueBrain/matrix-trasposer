#include "MatrixTransposer.hpp"

#include <cstdlib>
#include <cstdio>
#include <cassert>

#define OUTPUT_DEBUG_INFO

int GenerateRandomNumber (int min, int max)
{    return rand() % (max-min) + min; }

int main(int argv, char** argc)
{
    MPI_Init(&argv, &argc);

    //MPI variables
    int mpi_size=-1, mpi_rank=-1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    //TEST CASE 2
    srand (time(NULL)/(mpi_rank+1));

    unsigned int row_count=0;
    unsigned int * rank_end_offsets;
    unsigned int * counts;
    unsigned int * displs;
    unsigned int * cell_counts;
    int * cells;

    unsigned long long col_id=0, cell_id=0;
    (void) col_id; (void) cell_id; //clear Unused Var warning;

    rank_end_offsets = new unsigned int[mpi_size];

    //row_count = GenerateRandomNumber (10000, 100000); //1M and 10M;
    row_count = GenerateRandomNumber (3, 4); //1M and 10M;

    //rank_end_offsets
    unsigned int * rank_row_counts = new unsigned int[mpi_size];
    MPI_Allgather(&row_count, 1, MPI_UNSIGNED, rank_row_counts, 1, MPI_UNSIGNED, MPI_COMM_WORLD);
    for (int r=0; r<mpi_size; r++)
    rank_end_offsets[r] = r==0 ? rank_row_counts[r]-1 : rank_row_counts[r] + rank_end_offsets[r-1];
    unsigned int my_first_row_id = mpi_rank==0 ? 0 : rank_end_offsets[mpi_rank-1]+1;

    unsigned int total_row_count = 0;
    MPI_Allreduce ( &row_count, &total_row_count, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);

    //colsPerRow
    counts = new unsigned int [row_count];

    long long my_col_count=0, total_column_count=0;;
    for (unsigned int i=0; i<row_count; i++)
    {
        //colsPerRow[i] = generateRandomNumber (100, 1000);
        counts[i] = 2;
        my_col_count += counts[i];
    }

    long long my_cell_count=0, total_cell_count = 0;

    displs = new unsigned int[my_col_count];
    cell_counts = new unsigned int [my_col_count];
    col_id=0;
    for (unsigned int i=0; i<row_count; i++)
    {
        for (unsigned int j=0; j<counts[i]; j++)
        {
            displs[col_id]= total_row_count*j/counts[i];
            assert(displs[col_id]<total_row_count);
            cell_counts[col_id]= 1;
            //cellsPerCol[colId]= generateRandomNumber(2,20);
            my_cell_count+=cell_counts[col_id];
            col_id++;
        }
    }

    MPI_Allreduce ( &my_cell_count, &total_cell_count, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce ( &my_col_count, &total_column_count, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (mpi_rank==0) printf ("TOTAL COLUMNS COUNT %lld\n", total_column_count);
    if (mpi_rank==0) printf ("TOTAL CELLS COUNT %lld\n", total_cell_count);

    cells = new int[my_cell_count];
    for (unsigned int i=0; i<my_cell_count; i++)
        cells[i]=GenerateRandomNumber(0,8);

/*    //TEST CASE 1

        unsigned int _endRowCpus[4] = {3,8,11,16};
        endRowCpus = _endRowCpus;

      if (mpiRank % 2 == 0)
      {
        numberOfRows=3;
        unsigned int _colsPerRow[3] = {4,4,0};
        unsigned int _colsIndex[8] =  {0, 3, 9, 13, 3, 8, 12, 15};
        unsigned int _cellsPerCol[8] = {1,4,1,1,3,1,4,1};
        int _cells[16] = {0,1,2,3,4,2,3,4,5,6,5,6,7,8,9,7};

        colsPerRow = _colsPerRow;
        colsIndex = _colsIndex;
        cellsPerCol = _cellsPerCol;
        cells = _cells;
      }
      else
      {
        numberOfRows=5;
        unsigned int _colsPerRow[5] = {4,3,3,0,1};
        unsigned int _colsIndex[11] =  {1, 4, 10, 13, 1, 5, 9, 3, 8, 12, 15};
        unsigned int _cellsPerCol[11] = {1,4,1,1,3,1,4,1,1,2,3};
        int _cells[22] = {0,1,2,3,4,2,3,4,5,6,5,6,7,8,9,7,8,9,0,0,1,2};

        colsPerRow = _colsPerRow;
        colsIndex = _colsIndex;
        cellsPerCol = _cellsPerCol;
        cells = _cells;
     }
     */

#ifdef OUTPUT_DEBUG_INFO
    col_id=0, cell_id=0;
    for (int i=0; i<mpi_size; i++)
    {
        if (mpi_rank==i)
        {
           fprintf(stderr,"RANK %d\n", mpi_rank);
    for (unsigned int r=0; r<row_count; r++)
    {
        fprintf(stderr,"ROW %d (%d cols) ::", r+my_first_row_id, counts[r]);
        for (unsigned int c=0; c<counts[r]; c++)
        {
            fprintf(stderr,"[id %d, %d cell] ", displs[col_id], cell_counts[col_id]);
            for (unsigned int e=0; e<cell_counts[col_id]; e++)
                fprintf(stderr, "%d ", cells[cell_id++]);
            col_id++;
            fprintf(stderr,", ");
        }
        fprintf(stderr,"\n");
    }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        fprintf (stderr,"\n");
    }
    if (mpi_rank==0) fprintf(stderr,"Start\n");
#endif

    MatrixTransposer<int>::Transpose(rank_end_offsets, cells, counts, displs, cell_counts, MPI_COMM_WORLD);

#ifdef OUTPUT_DEBUG_INFO
    col_id=0, cell_id=0;
    for (int i=0; i<mpi_size; i++)
    {
        if (mpi_rank==i)
        {
           fprintf(stderr,"RANK %d\n", mpi_rank);
    for (unsigned int r=0; r<row_count; r++)
    {
        fprintf(stderr,"ROW %d (%d cols) ::", r+my_first_row_id, counts[r]);
        for (unsigned int c=0; c<counts[r]; c++)
        {
            fprintf(stderr,"[id %d, %d cell] ", displs[col_id], cell_counts[col_id]);
            for (unsigned int e=0; e<cell_counts[col_id]; e++)
                fprintf(stderr, "%d ", cells[cell_id++]);
            col_id++;
            fprintf(stderr,", ");
        }
        fprintf(stderr,"\n");
    }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        fprintf (stderr,"\n");
    }
    if (mpi_rank==0) fprintf(stderr,"Finish\n");
#endif

    return 0;
}
