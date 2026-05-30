/*-------------------------------------------
                INCLUDES
-------------------------------------------*/
#include "mpi.h"
#include "stdio.h"
#include "stdlib.h"

/*-------------------------------------------
             LITERAL CONSTANTS
-------------------------------------------*/
#define MASTER          (  0  )
#define SIZE_ARR_WORKER ( 10  )

#define BAD_NUM_ARGS    (  1  )
#define EXP_NUM_ARGS    (  2  )
#define FIRST_ARG       (  1  )          /* Excluding the name of the program */

/*-------------------------------------------
             MEMORY CONSTANTS
-------------------------------------------*/

/*-------------------------------------------
                MACROS
-------------------------------------------*/
#define check_correct_num_args( argc )     \
            if( EXP_NUM_ARGS != ( argc ) ) \
            {                              \
                return BAD_NUM_ARGS;       \
            }

#define get_rank( rank )                          \
        MPI_Comm_rank( MPI_COMM_WORLD, &( rank ) );

#define get_size( size )                          \
        MPI_Comm_size( MPI_COMM_WORLD, &( size ) );

/*-------------------------------------------
                VARIABLES
-------------------------------------------*/
int size = 0;
int rank = 0;

int main(int argc, char *argv[])
{
int m = atoi(argv[FIRST_ARG]);
int size_global_arr = 0;
int* local_arr = NULL;
int* global_arr = NULL;
int sum = 0;
int indx = 0;

MPI_Init(&argc, &argv);
{
    check_correct_num_args(argc);
    
    get_rank(rank);
    
    get_size(size);

    MPI_Bcast(&m, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

    size_global_arr = (MASTER == rank) ? SIZE_ARR_WORKER * size : 0;
    local_arr = (int*)malloc(sizeof(int) * SIZE_ARR_WORKER);

    if (MASTER == rank)
    {
        global_arr = (int*)malloc(sizeof(int) * size_global_arr);
    }
    else
    {
        for (indx = 0; indx < SIZE_ARR_WORKER; indx++)
        {
            local_arr[indx] = indx + m * rank;
        }
    }

    MPI_Gather(local_arr, SIZE_ARR_WORKER, MPI_INT, global_arr, SIZE_ARR_WORKER, MPI_INT, MASTER, MPI_COMM_WORLD);

    if (MASTER == rank)
    {
        for (indx = SIZE_ARR_WORKER; indx < SIZE_ARR_WORKER * size; indx++)
        {
            sum += global_arr[indx];
        }

        printf("Master got sum: %d", sum);
    }

    free(local_arr);

    if (NULL != global_arr)
    {
        free(global_arr);
    }
}
MPI_Finalize();

}