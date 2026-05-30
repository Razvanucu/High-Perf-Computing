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

    local_arr = (int*)malloc(sizeof(int) * SIZE_ARR_WORKER);
    if (MASTER != rank)
    {
        for (indx = 0; indx < SIZE_ARR_WORKER; indx++)
        {
            local_arr[indx] = indx + m * rank;
        }
    }
    else
    {
        global_arr = (int*)malloc(sizeof(int) * SIZE_ARR_WORKER);
        for (indx = 0; indx < SIZE_ARR_WORKER; indx++)
        {
            local_arr[indx] = 0;
        }
    }

    MPI_Reduce(local_arr, global_arr, SIZE_ARR_WORKER, MPI_INT, MPI_SUM, MASTER, MPI_COMM_WORLD);

    if (MASTER == rank)
    {
        for (indx = 0; indx < SIZE_ARR_WORKER; indx++)
        {
            sum += global_arr[indx];
        }
        printf("Master got sum: %d", sum);
    }

    if (NULL != global_arr)
    {
        free(global_arr);
    }
    free(local_arr);

    global_arr = NULL;
    local_arr = NULL;
}
MPI_Finalize();

}