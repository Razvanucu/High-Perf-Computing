#include <iostream>
#include "mpi.h"

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 5)
    {
        if (rank == 0)
            std::cout << "Please run with exactly 5 processes.\n";
        MPI_Finalize();
        return 0;
    }

    if (rank == 0)
    {
        int number[] = { 2,3,4,5 };
        int recv = 0;

        for (int dest = 0; dest < 4; dest++)
        {
            MPI_Send(&number[dest], 1, MPI_INT, dest+1, 0, MPI_COMM_WORLD);
            std::cout << "Master sends: " << number[dest] << std::endl;
        }

        for (int src = 0; src < 4; src++)
        {
            MPI_Recv(&recv, 1, MPI_INT, src+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::cout << "Master sends: " << recv << std::endl;
        }
    }
    else if (rank >= 1 && rank <= 5)
    {
        int number;
        MPI_Recv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        std::cout << "Slave received "<< rank <<": " << number << std::endl;
        number++;
        MPI_Send(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();

}