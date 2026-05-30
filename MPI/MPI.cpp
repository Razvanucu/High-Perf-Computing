/*-------------------------------------------
          PROJECT SIDE DEFINITIONS
-------------------------------------------*/
//#define DEBUG
//#define SEQUENTIAL_IMPLEMENTATION
//#define KERNEL_IMRPOVED
//#define SHARED_MEMORY_IMPLEMENTATION
//#define SHARED_MEMORY_IMPLEMENTATION_IMPROVED
#define MESSAGE_PASSING_IMPLEMENTATION
#define MESSAGE_PASSING_PRINT_ITERAIONS
#define MESSAGE_PASSING_IMPLEMENTATION_IMPROVED

/*-------------------------------------------
                INCLUDES
-------------------------------------------*/
#include "assert.h"
#include "fcntl.h"
#include "io.h"
#include "mpi.h"
#include "stdio.h"
#include "stdlib.h"
#include "sys/stat.h"
#include <thread>
#include "time.h"
#include <Windows.h>
#include <process.h>

/*-------------------------------------------
             LITERAL CONSTANTS
-------------------------------------------*/
#define MASTER          (     0      )
#define ERROR_FILE_OP   (     -1     )
#define DUMMY           (     0      )
#define MAX_UINT32      ( 0xFFFFFFFF )
#define NONE            (     0      )
#define EXP_NUM_ARGS    (     3      )
#define NUM_DIMS        (     2      )
#define GRID_2_P_INDX   (     0      )   
#define GRID_4_P_INDX   (     1      )   
#define GRID_8_P_INDX   (     2      )   
#define GRID_16_P_INDX  (     3      )   
#define GRID_32_P_INDX  (     4      )   
#define GRID_64_P_INDX  (     5      )   

/*-------------------------------------------
               DATA TYPES
-------------------------------------------*/
typedef char int8;
typedef short int int16;
typedef int int32;
typedef long long int int64;

typedef unsigned char uint8;
typedef unsigned short int uint16;
typedef unsigned int uint32;
typedef unsigned long long uint64;

typedef uint8 byte_type;
typedef uint16 word_type;
typedef uint32 dword_type;
typedef uint64 qword_type;

typedef int32 rank_type;

typedef uint8 status_file_op_type;
enum
{
    /*--------------------
      Successful states
    --------------------*/
    FILE_OP_OK,
    FILE_OPEN_OK,
    FILE_READ_OK,
    FILE_REDIRECTION_OK,
    FILE_DUPLICATION_OK,
    FILE_CLOSE_OK,

    /* Limits of successful states */
    FILE_MAX_STATES_OK,
    FILE_MIN_STATES_OK = FILE_OP_OK,
    FILE_NUM_STATES_OK = FILE_MAX_STATES_OK - FILE_MIN_STATES_OK,

    /*-----------------------
        Internal states
    -----------------------*/
    FILE_READ_REACH_END = FILE_MAX_STATES_OK,

    /* Limits of internal states */
    FILE_MAX_STATES_INTERN,
    FILE_MIN_STATES_INTERN = FILE_MAX_STATES_OK,
    FILE_NUM_STATES_INTERN = FILE_MAX_STATES_INTERN - FILE_MIN_STATES_INTERN,

    /*-----------------------
          Faulty states
    -----------------------*/
    FILE_OP_ERROR = FILE_MAX_STATES_INTERN,
    FILE_OPEN_ERROR,
    FILE_READ_ERROR,
    FILE_CLOSE_ERROR,
    FILE_REDIRECTION_ERROR,
    FILE_DUPLICATION_ERROR,

    /* Limits of faulty states */
    FILE_MAX_STATES_ERROR,
    FILE_MIN_STATES_ERROR = FILE_MAX_STATES_INTERN,
    FILE_NUM_STATES_ERROR = FILE_MAX_STATES_ERROR - FILE_MIN_STATES_ERROR
};

typedef struct ocean_matrix_type
{
    float* Ocean;
    uint16 physical_rows;
    uint16 physical_cols;
    uint16 rows;
    uint16 cols;
    uint16 starting_row;
    uint16 starting_col;
    uint16 end_row;
    uint16 end_col;
    float  local_diff;
    float  threshold;
}ocean_matrix_type;

/*-------------------------------------------
             MEMORY CONSTANTS
-------------------------------------------*/
#ifdef MESSAGE_PASSING_IMPLEMENTATION_IMPROVED
static const uint8 dimensions[6][2] =
    {
        { 1, 2 },   /* for 2 processes  */
        { 2, 2 },   /* for 4 processes  */
        { 2, 4 },   /* for 8 processes  */
        { 4, 4 },   /* for 16 processes */
        { 4, 8 },   /* for 32 processes */
        { 8, 8 }    /* for 64 processes */
    };
#endif
/*-------------------------------------------
                VARIABLES
-------------------------------------------*/
int32 size = 0;
int32 rank = 0;
#ifdef SHARED_MEMORY_IMPLEMENTATION
uint8 number_of_threads = 0;
#endif 

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

#define rotate_left( num, pos, type )                          \
        ( ( ( num ) << ( pos ) ) |                             \
          ( ( num ) >> ( ( sizeof( type ) << 3 ) - ( pos ) ) ) )

#define rotate_right( num, pos, type)                          \
        ( ( ( num ) >> ( pos ) ) |                             \
          ( ( num ) << ( ( sizeof( type ) << 3 ) - ( pos ) ) ) )

#define pow_2( exp )   \
        ( 1 << ( exp ) )

#define mod_pow_2( num, power_of_2 )       \
        ( ( num ) & ( ( power_of_2 ) - 1 ) )

#define square( num )       \
        ( ( num ) * ( num ) )

/*-------------------------------------------
                PROTOTYPES
-------------------------------------------*/
float fast_expo(float base, int32 exp);

/*-------------------------------------------
                PROCEDURES
-------------------------------------------*/
status_file_op_type open_file(const char* f_path, int32* fd, const int32 o_flags, const int32 p_mode_flags)
{
    if (((*fd) = _open(f_path, o_flags, p_mode_flags)) == ERROR_FILE_OP)
    {
        return FILE_OPEN_ERROR;
    }

    return FILE_OPEN_OK;

}


status_file_op_type duplicate_fd(const int32 fd, int32* dup_fd)
{
    if (((*dup_fd) = _dup(fd)) == ERROR_FILE_OP)
    {
        return FILE_DUPLICATION_ERROR;
    }

    return FILE_DUPLICATION_OK;

}


status_file_op_type redirect_fd(int32* old_fd, int32* new_fd)
{

    if ((_dup2((*old_fd), (*new_fd))) == ERROR_FILE_OP)
    {
        return FILE_REDIRECTION_ERROR;
    }

    return FILE_REDIRECTION_OK;

}


status_file_op_type close_file(const int32 fd)
{
    if ((_close(fd)) == ERROR_FILE_OP)
    {
        return FILE_CLOSE_ERROR;
    }

    return FILE_CLOSE_OK;

}


status_file_op_type read_number(int32 fd, int32* num)
{
    /*---------------------------
            LOCAL MACROS
    ---------------------------*/
#define is_digit( c )                          \
        ( ( '0' <= ( c ) ) && ( ( c ) <= '9' ) )

#define get_digit( digit_c ) \
        ( digit_c - '0' )

#define build_num( numb, digit )                            \
        ( ( numb ) = ( numb ) * 10 + ( get_digit( digit ) ) )

    /*---------------------------
            VARIABLES
    ---------------------------*/
    byte_type           byte = 0;
    int32               bytes_read = 0;
    boolean             is_negative = FALSE;
    status_file_op_type status = FILE_READ_OK;

    (*num) = 0;
    while (((bytes_read = _read(fd, &byte, 1)) == 1) && (!is_digit(byte) && byte != '-')); /* skip non-numeric characters or minus character */

    switch (bytes_read)
    {
    case -1:
        status = FILE_READ_ERROR;
        break;
    case 0:
        status = FILE_READ_REACH_END;
        break;
    default:
        (is_digit(byte)) ? build_num((*num), byte) :
            ('-' == byte) ? (is_negative = TRUE) : (status = FILE_READ_REACH_END);

        if (FILE_READ_OK == status)
        {
            while (((bytes_read = _read(fd, &byte, 1)) == 1) && (is_digit(byte)))
            {
                build_num((*num), byte);
            }

            if (is_negative)
            {
                (*num) = -(*num);
            }
        }
        is_negative = FALSE;

#ifdef DEBUG
        printf("%d\n", (*num));
#endif

        break;
    }

    /*---------------------------
           UNDEFINE MACROS
    ---------------------------*/
#undef is_digit
#undef build_num
#undef get_digit

    return status;

}


status_file_op_type read_float(const int32 fd, float* num)
{
    /*---------------------------
            LOCAL MACROS
    ---------------------------*/
#define is_digit( c )                          \
        ( ( '0' <= ( c ) ) && ( ( c ) <= '9' ) )

#define get_digit( digit_c ) \
        ( digit_c - '0' )

#define build_num( numb, digit )                            \
        ( ( numb ) = ( numb ) * 10 + ( get_digit( digit ) ) )

    /*---------------------------
            VARIABLES
    ---------------------------*/
    byte_type           byte = 0;
    int32               bytes_read = 0;
    boolean             is_negative = FALSE;
    boolean             decimal_point_found = FALSE;
    int64               integer_part = 0;
    int64               decimal_part = 0;
    int32               decimal_part_size = 0;
    status_file_op_type status = FILE_READ_OK;

    while (((bytes_read = _read(fd, &byte, 1)) == 1) && (!is_digit(byte) && byte != '-')); /* skip non-numeric characters or minus character */

    switch (bytes_read)
    {
    case -1:
        status = FILE_READ_ERROR;
        break;
    case 0:
        status = FILE_READ_REACH_END;
        break;
    default:
        
        if ('-' == byte)
        {
            is_negative = TRUE;
        }
        else if (is_digit(byte))
        {
            build_num(integer_part, byte);
        }

        while (((bytes_read = _read(fd, &byte, 1)) == 1) && is_digit(byte))
        {
            build_num(integer_part, byte);
        }

        if ('.' == byte)
        {
            while (((bytes_read = _read(fd, &byte, 1)) == 1) && is_digit(byte))
            {
                decimal_part_size += 1;
                build_num(decimal_part, byte);
            }
        }

        if (!is_negative)
        {
            (*num) = ((float)integer_part) +
                     ((float)decimal_part / fast_expo(10, decimal_part_size));
        }
        else 
        {
            (*num) = -((float)integer_part) -
                      ((float)decimal_part / fast_expo(10, decimal_part_size));
        }

#ifdef DEBUG
        printf("%f\n", (*num));
#endif

        break;
    }

    /*---------------------------
           UNDEFINE MACROS
    ---------------------------*/
#undef is_digit
#undef build_num
#undef get_digit

    return status;

}


status_file_op_type read_floats(const int32 fd, float* arr, int32 nums)
{
/*---------------------------
        VARIABLES
---------------------------*/
    int32               indx = 0;
    status_file_op_type status = FILE_READ_OK;

    for (indx = 0; (indx < nums) && (FILE_READ_OK == status); indx++)
    {
        arr[indx] = 0;
        status = read_float(fd, &arr[indx]);
    }

    return status;
}

status_file_op_type read_floats_fast(const int32 fd, float* arr, const int32 nums, const int32 rows, const int32 cols)
{
/*---------------------------
    LOCAL LITERAL CONSTANTS
---------------------------*/
#define BUFFER_SIZE ( 65536 )

/*---------------------------
        LOCAL TYPES
---------------------------*/
typedef uint8 read_state;
enum
{
    READ,
    SCAN_BUFFER,
    BUILD_INTEGER_PART,
    BUILD_DECIMAL_PART,
    EXIT,
    COMMIT_EXIT,
    ERROR_READ,
};

/*---------------------------
        LOCAL MACROS
---------------------------*/
#define is_digit( c )                          \
        ( ( '0' <= ( c ) ) && ( ( c ) <= '9' ) )

#define get_digit( digit_c ) \
        ( digit_c - '0' )

#define build_num( numb, digit )                            \
        ( ( numb ) = ( numb ) * 10 + ( get_digit( digit ) ) )

/*---------------------------
       LOCAL VARIABLES
---------------------------*/
int32               indx = 0;
int32               indx_arr = 0;
int32               nums_read = 0;
int32               bytes_read = 0;
boolean             is_negative = FALSE;
boolean             curr_num_built = TRUE;
int64               integer_part = 0;
int64               decimal_part = 0;
int64               decimal_divisor = 1;
status_file_op_type status = FILE_READ_OK;
char                buff[BUFFER_SIZE];
read_state          fsm_state = READ;
read_state          fsm_build_state = BUILD_INTEGER_PART;


indx_arr = cols; /* first row is a ghost row */
indx_arr++; /* first cell of a row is a ghost cell */

while (COMMIT_EXIT != fsm_state)
{
    switch (fsm_state)
    {
        case READ:
            bytes_read = _read(fd, buff, BUFFER_SIZE);
            switch (bytes_read)
            {
            case -1:
                fsm_state = ERROR_READ;
                break;
            case 0:
                if (!curr_num_built && nums_read < nums)
                {
                    if(indx_arr % cols == (cols - 1) ) // last cell of a row is a ghost cell
                    {
                        indx_arr = indx_arr + 2;
                    }

                    if (!is_negative)
                    {
                        arr[indx_arr++] = ((float)integer_part) +
                                         (((float)decimal_part) / ((float)decimal_divisor));
                    }
                    else
                    {
                        arr[indx_arr++] = -((float)integer_part) -
                                          (((float)decimal_part) / ((float)decimal_divisor));
                    }
                    nums_read++;
                }
                fsm_state = EXIT;
                break;
            default:
                fsm_state = SCAN_BUFFER;
                break;
            }
            break;
        case SCAN_BUFFER:
            for (indx = 0; indx < bytes_read; indx++)
            {
                if (is_digit(buff[indx]))
                {   
                    if (BUILD_INTEGER_PART == fsm_build_state)
                    {
                        build_num(integer_part, buff[indx]);
                    }
                    else
                    {
                        build_num(decimal_part, buff[indx]);
                        decimal_divisor *= 10;
                    }

                    curr_num_built = FALSE;
                }
                else if ('.' == buff[indx])
                {
                    fsm_build_state = BUILD_DECIMAL_PART;
                }
                else if ('-' == buff[indx])
                {
                    is_negative = TRUE;
                }
                else if(!curr_num_built)
                {
                    if (nums_read < nums)
                    {
                        if (indx_arr % cols == (cols - 1)) // last cell of a row is a ghost cell
                        {
                            indx_arr = indx_arr + 2;
                        }

                        if (!is_negative)
                        {
                            arr[indx_arr++] = ((float)integer_part) +
                                (((float)decimal_part) / ((float)decimal_divisor));
                        }
                        else
                        {
                            arr[indx_arr++] = -((float)integer_part) -
                                (((float)decimal_part) / ((float)decimal_divisor));
                        }
                        nums_read++;

                        #ifdef DEBUG
                        // printf("%f\n", arr[indx_arr - 1]);
                        #endif

                        integer_part = 0;
                        decimal_part = 0;
                        decimal_divisor = 1;
                        is_negative = FALSE;
                        curr_num_built = TRUE;
                        fsm_build_state = BUILD_INTEGER_PART;
                    }
                    else
                    {
                        fsm_state = ERROR_READ;
                        break;
                    }
                }
            }
            fsm_state = (fsm_state == ERROR_READ) ? fsm_state : READ;
            break;
        case EXIT:
            status = (nums_read == nums ) ? FILE_READ_OK : FILE_READ_ERROR;
            fsm_state = COMMIT_EXIT;
            #ifdef DEBUG
            // printf("Number of floats read: %d\n", nums_read);
            #endif
            break;
        default:
            status = FILE_READ_ERROR;
            fsm_state = EXIT;
            break;
            
    }
}
/*--------------------------------------
 Undefine macros and literal constants
--------------------------------------*/
#undef BUFFER_SIZE

#undef is_digit
#undef build_num
#undef get_digit

return status;

}

/*-------------------------------------------
            Helper Functions
-------------------------------------------*/
float fast_expo(float base, int32 exp)
{
/*---------------------------
        VARIABLES
---------------------------*/
    float rez = 1;

    while (exp)
    {
        if (exp & 1)
        {
            rez = rez * base;
        }

        base *= base;
        exp >>= 1;
    }

    return rez;
}

void create_2D_cartesian_topology(MPI_Comm* comm_cart, const int32 processes_per_row, const int32 processes_per_col, const boolean row_periodic, const boolean col_periodic, const boolean reorder)
{
/*---------------------------
        VARIABLES
---------------------------*/
    int32 dims[NUM_DIMS] = { processes_per_col, processes_per_row };
    uint32 indx = 0;
    int32 periods[NUM_DIMS] = { row_periodic, col_periodic };

    MPI_Cart_create(MPI_COMM_WORLD, NUM_DIMS, dims, periods, reorder, comm_cart);
}

void get_neighbors_processes_ranks(const MPI_Comm comm_cart, const int32 displacement, rank_type* proc_up, rank_type* proc_right, rank_type* proc_down, rank_type* proc_left)
{
/*---------------------------
   LOCAL LITERAL CONSTANTS
---------------------------*/
#define VERTICAL_DIR   ( 0 )
#define HORIZONTAL_DIR ( 1 )

    MPI_Cart_shift(comm_cart, VERTICAL_DIR, displacement, proc_up, proc_down);
    MPI_Cart_shift(comm_cart, HORIZONTAL_DIR, displacement, proc_left, proc_right);

#ifdef DEBUG
    printf("Process rank %d has neighbors: up %d, right %d , down %d, left %d", rank, (*proc_up), (*proc_right), (*proc_down), (*proc_left));
#endif
/*-------------------------------
Undefine local literal constants
 --------------------------------*/
#undef VERTICAL_DIR
#undef HORIZONTAL_DIR

}

#ifdef MESSAGE_PASSING_IMPLEMENTATION_IMPROVED
static inline int32 get_grid_rows()
{
    switch ( size )
    {
    case 2:
        return dimensions[GRID_2_P_INDX][0];
    case 4:
        return dimensions[GRID_4_P_INDX][0];
    case 8:
        return dimensions[GRID_8_P_INDX][0];
    case 16:
        return dimensions[GRID_16_P_INDX][0];
    case 32:
        return dimensions[GRID_32_P_INDX][0];
    case 64:
        return dimensions[GRID_64_P_INDX][0];
    default:
        return -1; /* invalid number of processes */
        break;
    }
}

static inline int32 get_grid_cols()
{
    switch (size)
    {
    case 2:
        return dimensions[GRID_2_P_INDX][1];
    case 4:
        return dimensions[GRID_4_P_INDX][1];
    case 8:
        return dimensions[GRID_8_P_INDX][1];
    case 16:
        return dimensions[GRID_16_P_INDX][1];
    case 32:
        return dimensions[GRID_32_P_INDX][1];
    case 64:
        return dimensions[GRID_64_P_INDX][1];
    default:
        return -1; /* invalid number of processes */
        break;
    }
}
#endif

/*-------------------------------------------
        Read Inputs ( MASTER PART )
-------------------------------------------*/
ocean_matrix_type* read_args(int32 argc, char* argv[])
{
/*---------------------------
        VARIABLES
---------------------------*/
    int32 fd_matrix = 0;
    status_file_op_type status_op_file = FILE_OP_OK;
    int32 matrix_size = 0;
    float* arr = NULL;
    ocean_matrix_type* ocean = NULL;

#ifndef SHARED_MEMORY_IMPLEMENTATION
    if (argc != 3)
    {
        perror("Bad number of arguments! Expected 2 arguments: <threshold> <path_to_ocean_matrix>");
        return NULL;
    }
#else
    if (argc != 4)
    {
        perror("Bad number of arguments! Expected 3 arguments: <threshold> <path_to_ocean_matrix> <number_of_threads>");
        return NULL;
    }
    number_of_threads = atoi(argv[3]);

    if(0 == number_of_threads)
    {
        perror("Bad number of threads! Expected a positive integer value for number of threads.");
        return NULL;
    }
#endif

    status_op_file = open_file(argv[2], &fd_matrix, _O_RDONLY, NONE);
    if (FILE_OP_ERROR == status_op_file)
    {
        perror("Could not open file!");
        return NULL;
    }

    status_op_file = read_number(fd_matrix, &matrix_size);
    if (FILE_READ_ERROR == status_op_file)
    {
        perror("Could not read matrix size!");
        return NULL;
    }

    arr = (float*)calloc( ( matrix_size + 2 ) * ( matrix_size + 2 ), sizeof(float));
    if (NULL == arr)
    {
        perror("Could not allocate matrix size!");
        return NULL;
    }

    status_op_file = read_floats_fast(fd_matrix, arr, matrix_size * matrix_size, matrix_size+2, matrix_size+2);
    if (FILE_READ_ERROR == status_op_file)
    {
        perror("Could not read the elements of the matrix");
        free(arr);
        arr = NULL;
        return NULL;
    }

    ocean = (ocean_matrix_type*)malloc(sizeof(ocean_matrix_type));
    if (NULL == ocean)
    {
        perror("Could not allocate Ocean type!");
        free(arr);
        arr = NULL;
        return NULL;
    }

    ocean->physical_cols = ocean->physical_rows = matrix_size + 2;
    ocean->cols = ocean->rows = matrix_size;
    ocean->starting_col = ocean->starting_row = 1;
    ocean->end_col = ocean->end_row = matrix_size + 1;
    ocean->local_diff = 0.0f;

    if (sscanf(argv[1], "%f", &ocean->threshold) != 1)
    {
        perror("Could not read threshold value");
        free(arr);
        arr = NULL;
        return NULL;
    }

    ocean->Ocean = arr;
   
    status_op_file = close_file(fd_matrix);
    if (FILE_CLOSE_ERROR == status_op_file)
    {
        perror("Error when closing the file!");
        free(arr);
        arr = NULL;
        return NULL;
    }

    return ocean;

}


/*-------------------------------------------
            Ocean Kernel Solver
-------------------------------------------*/
void* ocean_kernel_solver(void* ocean)
{
/*---------------------------
   LOCAL LITERAL CONSTANTS
---------------------------*/
#define NUM_NEIGHBORS (    4   )
#define UP            { -1,  0 }
#define RIGHT         {  0,  1 }
#define DOWN          {  1,  0 }
#define LEFT          {  0, -1 }

/*---------------------------
    LOCAL MEMORY CONSTANTS
---------------------------*/
static const struct
{
    int16 di;
    int16 dj;
}neighbors_dist[NUM_NEIGHBORS] = {UP, RIGHT, DOWN, LEFT};

/*---------------------------
        LOCAL MACROS
---------------------------*/
#define Ocean_at(i, j)                                                         \
        ( ocean_local->Ocean )[ ( i ) * ( ocean_local->physical_cols ) + ( j ) ]

/*---------------------------
        VARIABLES
---------------------------*/
int16 row = 0;
int16 col = 0;
uint8  neighbor = 0;
float  temp = 0.0f;
ocean_matrix_type *ocean_local = (ocean_matrix_type*) ocean;

    ocean_local->local_diff = 0.0f;
    for (row = ocean_local->starting_row; row < ocean_local->end_row; row++)
    {
        
        for (col = ocean_local->starting_col; col < ocean_local->end_col; col++)
        {
            temp = Ocean_at(row, col);
            for (neighbor = 0; neighbor < NUM_NEIGHBORS; neighbor++)
            {
                Ocean_at(row, col) += Ocean_at(row + neighbors_dist[neighbor].di, col + neighbors_dist[neighbor].dj);
            }
            Ocean_at(row, col) *= 0.2f;

            ocean_local->local_diff += fabsf(Ocean_at(row, col) - temp);
        }
       
    }

/*--------------------------------
 Undefine local literal constants
--------------------------------*/
#undef NUM_NEIGHBORS
#undef UP
#undef RIGHT
#undef DOWN
#undef LEFT

/*---------------------------
    Undefine local macros
---------------------------*/
#undef Ocean_at

    return NULL;

}


/*-------------------------------------------
            Sequential algorithm    
-------------------------------------------*/
#ifdef SEQUENTIAL_IMPLEMENTATION
void sequential_implementation(ocean_matrix_type* ocean)
{
/*---------------------------
        VARIABLES
---------------------------*/
double t_Start = 0.0;
double t_End = 0.0;
boolean done = FALSE;
uint64  iterations_until_convergence = 0;

    t_Start = MPI_Wtime();
    while (FALSE == done)
    {
        ocean_kernel_solver(ocean);
        if (ocean->local_diff / (((float)ocean->rows) * ((float)ocean->cols)) < ocean->threshold)
        {
            done = TRUE;
        }
        iterations_until_convergence++;
    }
    t_End = MPI_Wtime();

    printf("Sequential algorithm time to execute: %lf\n", t_End - t_Start);
    printf("Sequential algorithm iterations to converge: %llu\n", iterations_until_convergence);
}
#endif 

/*-------------------------------------------
            Shared Memory algorithm
-------------------------------------------*/
#ifdef SHARED_MEMORY_IMPLEMENTATION
unsigned __stdcall ocean_kernel_solver_worker(void* args)
{
/*---------------------------
    LOCAL LITERAL CONSTANTS
---------------------------*/
#define LOCAL_DIFF_PADDING ( 64 - sizeof(float) )

/*---------------------------
        LOCAL TYPES
---------------------------*/
#ifdef SHARED_MEMORY_IMPLEMENTATION_IMPROVED
typedef struct thread_local_diff_type
{
    float local_diff;
    char padding[ LOCAL_DIFF_PADDING ]; // for cache contention avoidance
}thread_local_diff_type;
#endif

typedef struct thread_args_type
{
    ocean_matrix_type* ocean;
    uint8 thread_id;
    uint8 num_threads;
#ifndef SHARED_MEMORY_IMPLEMENTATION_IMPROVED
    float* thread_local_diffs;
#else
    thread_local_diff_type *thread_local_diffs;
#endif

    LPSYNCHRONIZATION_BARRIER barrier;
} thread_args_type;

/*---------------------------
        SHARED VARIABLES
---------------------------*/
static volatile boolean done = FALSE;
static uint64  iterations_until_convergence = 0;
static float global_diff = 0;
static double t_Start = 0.0;
static double t_End = 0.0;

/*---------------------------
        LOCAL VARIABLES
---------------------------*/
LPSYNCHRONIZATION_BARRIER barrier = ((thread_args_type*)args)->barrier;
ocean_matrix_type* ocean_local = ((thread_args_type *)args)->ocean;
uint8 thread_id = ((thread_args_type *)args)->thread_id;
#ifndef SHARED_MEMORY_IMPLEMENTATION_IMPROVED
float* threads_local_diff = ((thread_args_type*)args)->thread_local_diffs;
#else
thread_local_diff_type* threads_local_diff = ((thread_args_type*)args)->thread_local_diffs;
#endif 


uint8 num_threads = ((thread_args_type*)args)->num_threads;

    if(MASTER == thread_id)
    {
        t_Start = MPI_Wtime();
    }
    
    while (!done)
    {
        ocean_kernel_solver(ocean_local);
        
    #ifndef SHARED_MEMORY_IMPLEMENTATION_IMPROVED
        threads_local_diff[thread_id] = ocean_local->local_diff;
    #else
        threads_local_diff[thread_id].local_diff = ocean_local->local_diff;
    #endif
        EnterSynchronizationBarrier(barrier, SYNCHRONIZATION_BARRIER_FLAGS_BLOCK_ONLY);

        if(MASTER == thread_id)
        {
            global_diff = 0;
            for (uint8 i = 0; i < num_threads; i++)
            {
            #ifndef SHARED_MEMORY_IMPLEMENTATION_IMPROVED
                global_diff += threads_local_diff[i];
            #else
                global_diff += threads_local_diff[i].local_diff;
            #endif
            }
            if (global_diff / (((float)ocean_local->rows) * ((float)ocean_local->cols)) < ocean_local->threshold)
            {
                done = TRUE;
            }
            iterations_until_convergence++;
        }

        EnterSynchronizationBarrier(barrier, SYNCHRONIZATION_BARRIER_FLAGS_BLOCK_ONLY);

    }

    if(MASTER == thread_id)
    {
        t_End = MPI_Wtime();
        printf("Shared memory algorithm time to execute: %lf\n", t_End - t_Start);
        printf("Shared memory algorithm iterations to converge: %llu\n", iterations_until_convergence);
    }

#undef LOCAL_DIFF_PADDING

    return 0;
}

void shared_memory_implementation(ocean_matrix_type* ocean)
{
/*---------------------------
    LOCAL LITERAL CONSTANTS
---------------------------*/
#define LOCAL_DIFF_PADDING ( 64 - sizeof(float) )

/*---------------------------
        LOCAL TYPES
---------------------------*/
#ifdef SHARED_MEMORY_IMPLEMENTATION_IMPROVED
typedef struct thread_local_diff_type
{
    float local_diff;
    char padding[LOCAL_DIFF_PADDING]; // for cache contention avoidance
}thread_local_diff_type;
#endif

typedef struct thread_args_type
{
    ocean_matrix_type* ocean;
    uint8 thread_id;
    uint8 num_threads;
#ifndef SHARED_MEMORY_IMPLEMENTATION_IMPROVED
    float* thread_local_diffs;
#else
    thread_local_diff_type* thread_local_diffs;
#endif
    LPSYNCHRONIZATION_BARRIER barrier;
} thread_args_type;

/*---------------------------
        VARIABLES
---------------------------*/
SYNCHRONIZATION_BARRIER barrier;
thread_args_type* thread_args = NULL;
HANDLE* thread_handles = NULL;
#ifndef SHARED_MEMORY_IMPLEMENTATION_IMPROVED
float* threads_local_diff = NULL;
#else
thread_local_diff_type* threads_local_diff = NULL;
#endif
ocean_matrix_type* thread_ocean = NULL;
uint8 thread = 0;
uint16 rows_remainder = ocean->rows % number_of_threads;
uint16 rows_per_thread = ocean->rows / number_of_threads;
#ifdef SHARED_MEMORY_IMPLEMENTATION_IMPROVED
uint16 cols_remainder = ocean->cols % number_of_threads;
uint16 cols_per_thread = ocean->cols / number_of_threads;
#endif

    thread_args = (thread_args_type*)malloc(sizeof(thread_args_type) * number_of_threads);
    if(thread_args == NULL)
    {
        perror("Could not allocate thread arguments!");
        exit(ERROR);
    }

    thread_handles = (HANDLE*)malloc(sizeof(HANDLE) * number_of_threads);
    if(thread_handles == NULL)
    {
        perror("Could not allocate thread handles!");
        free(thread_args);
        thread_args = NULL;
        exit(ERROR);
    }

    thread_ocean = (ocean_matrix_type*)malloc(sizeof(ocean_matrix_type) * number_of_threads);
    if(thread_ocean == NULL)
    {
        perror("Could not allocate thread ocean matrices!");
        free(thread_args);
        thread_args = NULL;
        free(thread_handles);
        thread_handles = NULL;
        exit(ERROR);
    }

#ifndef SHARED_MEMORY_IMPLEMENTATION_IMPROVED
    threads_local_diff = (float*)malloc(sizeof(float) * number_of_threads);
#else
    threads_local_diff = (thread_local_diff_type*)malloc(sizeof(thread_local_diff_type) *   number_of_threads);
#endif
    if(threads_local_diff == NULL)
    {
        perror("Could not allocate thread local diffs!");
        free(thread_args);
        thread_args = NULL;
        free(thread_handles);
        thread_handles = NULL;
        free(thread_ocean);
        thread_ocean = NULL;
        exit(ERROR);
    }

    if( FALSE == InitializeSynchronizationBarrier(&barrier, number_of_threads, SYNCHRONIZATION_BARRIER_FLAGS_BLOCK_ONLY) )
    {
        perror("Could not initialize the synchronization barrier!");
        free(thread_args);
        thread_args = NULL;
        free(thread_handles);
        thread_handles = NULL;
        free(thread_ocean);
        thread_ocean = NULL;
        free(threads_local_diff);
        threads_local_diff = NULL;
        exit(ERROR);
    }

    for(thread = 0; thread < number_of_threads; thread++)
    {
        thread_ocean[thread] = *ocean;
        thread_ocean[thread].starting_row = (0 == thread) ? 1 : thread_ocean[thread - 1].end_row;
        if (rows_remainder == 0)
        {
            thread_ocean[thread].end_row = thread_ocean[thread].starting_row + rows_per_thread;
        }
        else
        {
            thread_ocean[thread].end_row = thread_ocean[thread].starting_row + rows_per_thread + 1;
            rows_remainder = rows_remainder - 1;
        }
        thread_args[thread].ocean = &thread_ocean[thread];
        thread_args[thread].num_threads = number_of_threads;
        thread_args[thread].thread_id = thread;
        
#ifndef SHARED_MEMORY_IMPLEMENTATION_IMPROVED
        threads_local_diff[thread] = 0.0f;
        thread_args[thread].thread_local_diffs = threads_local_diff;
#else
        threads_local_diff[thread].local_diff = 0.0f;
        thread_args[thread].thread_local_diffs = threads_local_diff;
#endif
        thread_args[thread].barrier = &barrier;

#ifdef DEBUG
        printf("Thread %d will process rows from %d to %d\n", thread, thread_ocean[thread].starting_row, thread_ocean[thread].end_row - 1);
#endif
    }

    for (thread = 0; thread < number_of_threads; thread++)
    {
        thread_handles[thread] = (HANDLE)_beginthreadex(NULL, 0, ocean_kernel_solver_worker, &thread_args[thread], 0, NULL);
        if (thread_handles[thread] == NULL)
        {
            perror("Could not create thread!");
            free(thread_args);
            thread_args = NULL;
            free(thread_handles);
            thread_handles = NULL;
            free(thread_ocean);
            thread_ocean = NULL;
            free(threads_local_diff);
            threads_local_diff = NULL;
            DeleteSynchronizationBarrier(&barrier);
            exit(ERROR);
        }
    }

    WaitForMultipleObjects(number_of_threads, thread_handles, TRUE, INFINITE);

    for (thread = 0; thread < number_of_threads; thread++)
    {
        CloseHandle(thread_handles[thread]);
    }

    free(thread_args);
    thread_args = NULL;
    free(thread_handles);
    thread_handles = NULL;
    free(thread_ocean);
    thread_ocean = NULL;
    free(threads_local_diff);
    threads_local_diff = NULL;
    DeleteSynchronizationBarrier(&barrier);

#undef LOCAL_DIFF_PADDING

}
#endif

/*-------------------------------------------
          Message Passing algorithm
-------------------------------------------*/
void message_passing_implementation(ocean_matrix_type* ocean, MPI_Comm communicator, uint16* global_rows, uint16* global_cols, const int32 *send_counts_elems, const int32 *displacements_send_elems, const int32 *recv_counts_gather_elems, const int32 *displacements_gather_elems)
{
/*---------------------------
    LOCAL LITERAL CONSTANTS
---------------------------*/
#define ADI_MINUNE_TAG  ( 69 )
#define ROMEO_FNTSK_TAG ( 67 )
#define BORDER_ROWS     (  2 )
#define BORDER_COLS     (  2 )

/*---------------------------
        VARIABLES
---------------------------*/
boolean done = FALSE;
float global_diff = 0.0f;
uint16 my_rows = 0;
uint16 my_cols = 0;
MPI_Status status;
uint64 iterations_until_convergence = 0;
float* halo_scratch = NULL;

ocean_matrix_type* ocean_local = NULL;

MPI_Bcast(global_rows, 1, MPI_UINT16_T, MASTER, communicator);
MPI_Bcast(global_cols, 1, MPI_UINT16_T, MASTER, communicator);

my_rows = ((*global_rows) / size) + ((rank < ((*global_rows) % size)) ? 1 : 0) + BORDER_ROWS;
my_cols = (*global_cols) + BORDER_COLS;

ocean_local = (ocean_matrix_type*)malloc(sizeof(ocean_matrix_type));
if( NULL == ocean_local )
{
    perror("Could not allocate local ocean matrix!");
    MPI_Abort(communicator, ERROR);
    exit(ERROR);
}

ocean_local->physical_rows = my_rows;
ocean_local->physical_cols = my_cols;
ocean_local->rows = my_rows - BORDER_ROWS;
ocean_local->cols = my_cols - BORDER_COLS;
ocean_local->starting_row = 1;
ocean_local->starting_col = 1;
ocean_local->end_row = ocean_local->starting_row + ocean_local->rows;
ocean_local->end_col = ocean_local->starting_col + ocean_local->cols;
ocean_local->local_diff = 0.0f;

ocean_local->Ocean = (float*)malloc(sizeof(float) * my_rows * my_cols);
if (NULL == ocean_local->Ocean)
{
    perror("Could not allocate local ocean matrix data!");
    free(ocean_local);
    ocean_local = NULL;
    MPI_Abort(communicator, ERROR);
    exit(ERROR);
}
#ifdef MESSAGE_PASSING_IMPLEMENTATION_IMPROVED
MPI_Bcast(&ocean_local->threshold, 1, MPI_FLOAT, MASTER, communicator);
#endif

MPI_Scatterv((MASTER == rank) ? ocean->Ocean: NULL, send_counts_elems, displacements_send_elems, MPI_FLOAT, ocean_local->Ocean, my_rows * my_cols, MPI_FLOAT, MASTER, communicator);

halo_scratch = (float*)malloc(sizeof(float) * my_cols);   // Spare row in case the last process has only 2 total rows so that buffers don't overlap

    while (!done)
    {   
        ocean_kernel_solver(ocean_local);

        MPI_Reduce(&(ocean_local->local_diff), &global_diff, 1, MPI_FLOAT, MPI_SUM, MASTER, communicator);
        
        if(MASTER == rank)
        {   
            iterations_until_convergence++;
            if (global_diff / ((float)(*global_rows) * (float)(*global_cols)) < ocean->threshold)
            {
                done = TRUE;
            }
        }
        
        MPI_Bcast(&done, 1, MPI_UINT8_T, MASTER, communicator);

        MPI_Sendrecv((size - 1 == rank) ? halo_scratch : (ocean_local->Ocean + (my_rows - BORDER_ROWS) * my_cols), my_cols, MPI_FLOAT, (size - 1 == rank) ? MPI_PROC_NULL : (rank + 1), ADI_MINUNE_TAG,
            (MASTER == rank) ? halo_scratch : (ocean_local->Ocean), my_cols, MPI_FLOAT, (MASTER == rank) ? MPI_PROC_NULL : (rank - 1), ADI_MINUNE_TAG,
            communicator, &status);

        MPI_Sendrecv((MASTER == rank) ? halo_scratch : (ocean_local->Ocean + my_cols), my_cols, MPI_FLOAT, (MASTER == rank) ? MPI_PROC_NULL : (rank - 1), ROMEO_FNTSK_TAG,
            (size - 1 == rank) ? halo_scratch : (ocean_local->Ocean + (my_rows - BORDER_ROWS + 1) * my_cols), my_cols, MPI_FLOAT, (size - 1 == rank) ? MPI_PROC_NULL : (rank + 1), ROMEO_FNTSK_TAG,
            communicator, &status);
    }

MPI_Gatherv(ocean_local->Ocean, (my_rows - BORDER_ROWS + 1) * my_cols, MPI_FLOAT, (MASTER == rank) ? ocean->Ocean : NULL, recv_counts_gather_elems, displacements_gather_elems, MPI_FLOAT, MASTER, communicator);

free(ocean_local->Ocean);
ocean_local->Ocean = NULL;
free(ocean_local);
ocean_local = NULL;
free(halo_scratch);
halo_scratch = NULL;

#ifdef MESSAGE_PASSING_PRINT_ITERAIONS
if (MASTER == rank)
{
    printf("Message passing algorithm iterations to converge: %llu\n", iterations_until_convergence);
}
#endif

#undef ADI_MINUNE_TAG
#undef ROMEO_FNTSK_TAG

}

#ifdef MESSAGE_PASSING_IMPLEMENTATION_IMPROVED
/*-------------------------------------------
    Message Passing algorithm improved
-------------------------------------------*/
void message_passing_implementation_improved(ocean_matrix_type* ocean, MPI_Comm communicator, uint16* global_rows, uint16* global_cols, uint16 *block_size_rows, uint16 *block_size_cols, const int32* send_counts_elems, const int32* displacements_send_elems, const int32* recv_counts_gather_elems, const int32* displacements_gather_elems)
{
/*---------------------------
    LOCAL LITERAL CONSTANTS
---------------------------*/
#define NUM_NEIGHBORS   ( 4 )
#define UPPER_NEIGHBOR  ( 0 )
#define RIGHT_NEIGHBOR  ( 1 )
#define DOWN_NEIGHBOR   ( 2 )
#define LEFT_NEIGHBOR   ( 3 )
#define DISPLACEMENT    ( 1 )

/*---------------------------
        VARIABLES
---------------------------*/
boolean done = FALSE;
uint16  grid_rows = get_grid_rows();
uint16  grid_cols = get_grid_cols();
rank_type neighbors_ranks[NUM_NEIGHBORS] = { NONE, NONE, NONE, NONE };
float* row_halo= NULL;
float* col_halo= NULL;
ocean_matrix_type* ocean_local = NULL;
float global_diff = 0.0f;

int32 sizes_block[NUM_DIMS] = { NONE, NONE };
int32 subsizes_block[NUM_DIMS] = { NONE, NONE };
int32 starts_block[NUM_DIMS] = { NONE, NONE };
MPI_Datatype block_type;
MPI_Datatype block_temp_type;

int32 sizes_block_col[NUM_DIMS] = { NONE, NONE };
int32 subsizes_block_col[NUM_DIMS] = { NONE, NONE };
int32 starts_block_col[NUM_DIMS] = { NONE, NONE };
MPI_Datatype col_halo_type;
MPI_Datatype col_temp_bock_type;

create_2D_cartesian_topology(&communicator, grid_rows, grid_cols, FALSE, FALSE, TRUE);

get_neighbors_processes_ranks(communicator, DISPLACEMENT, &neighbors_ranks[UPPER_NEIGHBOR], &neighbors_ranks[RIGHT_NEIGHBOR], &neighbors_ranks[DOWN_NEIGHBOR], &neighbors_ranks[LEFT_NEIGHBOR]);
    
MPI_Bcast(global_rows, 1, MPI_UINT16_T, MASTER, communicator);
MPI_Bcast(global_cols, 1, MPI_UINT16_T, MASTER, communicator);
    
MPI_Bcast(block_size_rows, 1, MPI_UINT16_T, MASTER, communicator);
MPI_Bcast(block_size_cols, 1, MPI_UINT16_T, MASTER, communicator);

ocean_local = (ocean_matrix_type*)malloc(sizeof(ocean_matrix_type));
if (NULL == ocean_local)
{
    perror("Could not allocate local ocean matrix!");
    MPI_Abort(communicator, ERROR);
    exit(ERROR);
}

ocean_local->Ocean = (float*)malloc(sizeof(float) * (*block_size_rows) * (*block_size_cols));
if (NULL == ocean_local->Ocean)
{
    perror("Could not allocate local ocean matrix data!");
    free(ocean_local);
    ocean_local = NULL;
    MPI_Abort(communicator, ERROR);
    exit(ERROR);
}

MPI_Scatter((MASTER == rank) ? (&(ocean->threshold)) : NULL, 1, MPI_FLOAT, &(ocean_local->threshold), 1, MPI_FLOAT, MASTER, communicator);

sizes_block[0] = (int32)(*global_rows);
sizes_block[1] = (int32)(*global_cols);
subsizes_block[0] = (int32)(*block_size_rows);
subsizes_block[1] = (int32)(*block_size_cols);
starts_block[0] = 0;
starts_block[1] = 0;

MPI_Type_create_subarray(NUM_DIMS, sizes_block, subsizes_block, starts_block, MPI_ORDER_C, MPI_FLOAT, &block_temp_type);
MPI_Type_create_resized(block_temp_type, 0, (int32)(*block_size_cols) * sizeof(float), &block_type);

MPI_Type_vector((int32)(*block_size_rows), 1, (int32)(int32)(*block_size_cols), MPI_FLOAT, &col_halo_type);
    
MPI_Type_commit(&col_halo_type);
MPI_Type_commit(&block_type);

// Scatter blocks of the matrix to processes
MPI_Scatterv((MASTER == rank) ? ocean->Ocean : NULL, send_counts_elems, displacements_send_elems, block_type, ocean_local->Ocean, (*block_size_rows) * (*block_size_cols), MPI_FLOAT, MASTER, communicator);

while (!done)
{
    // Sendrecv border rows and columns to neighbors
    // Empty have to think
    // Ocean Kernel blabalabla
    ocean_kernel_solver(ocean_local);

    // Allreduce local diffs to global diff
    MPI_Allreduce(&(ocean_local->local_diff), &global_diff, 1, MPI_FLOAT, MPI_SUM, communicator);

    // Check for convergence and broadcast the result (allreduce)
    if( global_diff / ((float)(*global_rows) * (float)(*global_cols)) < ocean_local->threshold )
    {
        done = TRUE;
    }
}

    
// Gather retarded blocks back to master
MPI_Gatherv(ocean_local->Ocean, (*block_size_rows) * (*block_size_cols), block_type, (MASTER == rank) ? ocean->Ocean : NULL, recv_counts_gather_elems, displacements_gather_elems, block_type, MASTER, communicator);
    
// free local resources
free(ocean_local->Ocean);
ocean_local->Ocean = NULL;
free(ocean_local);
ocean_local = NULL;
free(row_halo);
row_halo = NULL;
free(col_halo);
col_halo = NULL;

// free created types
MPI_Type_free(&block_temp_type);
MPI_Type_free(&block_type);
MPI_Type_free(&col_halo_type);

#undef NUM_NEIGHBORS 
#undef UPPER_NEIGHBOR
#undef RIGHT_NEIGHBOR
#undef DOWN_NEIGHBOR 
#undef LEFT_NEIGHBOR 
#undef DISPLACEMENT  

}
#endif

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    {   
        
        ocean_matrix_type* ocean = NULL;

#ifdef MESSAGE_PASSING_IMPLEMENTATION

        MPI_Comm communicator = MPI_COMM_WORLD; 
        int32* send_counts_elems = NULL;
        int32* displacements_send_elems = NULL;
        int32* recv_counts_gather_elems = NULL;
        int32* displacements_gather_elems = NULL;
        uint8  node = 0;
        uint16 used_rows = 0;
        uint16 used_cols = 0;
        uint16 interior_rows = 0;
        uint16 interior_cols = 0;
        uint32 matrix_start_offset = 0;
        uint16 global_rows = 0;
        uint16 global_cols = 0;

#endif

        get_size(size);
        get_rank(rank);
        
        if (MASTER == rank)
        {   
            double t_Start = 0.0;
            double t_End = 0.0;

            t_Start = MPI_Wtime();
            ocean = read_args(argc, argv);
            t_End = MPI_Wtime();
            
            if (NULL == ocean)
            {
                MPI_Abort(MPI_COMM_WORLD, ERROR);
                exit(ERROR);
            }

            printf("Time to read the matrix %lf\n", t_End - t_Start);

    #ifdef MESSAGE_PASSING_IMPLEMENTATION
            
            send_counts_elems = (int32*)malloc(sizeof(int32) * size);
            if( NULL == send_counts_elems )
            {
                perror("Could not allocate send counts array!");
                free(ocean->Ocean);
                ocean->Ocean = NULL;
                free(ocean);
                ocean = NULL;
                MPI_Abort(MPI_COMM_WORLD, ERROR);
                exit(ERROR);
            }

            displacements_send_elems = (int32*)malloc(sizeof(int32) * size);
            if( NULL == displacements_send_elems )
            {
                perror("Could not allocate send displacements array!");
                free(ocean->Ocean);
                ocean->Ocean = NULL;
                free(ocean);
                ocean = NULL;
                free(send_counts_elems);
                send_counts_elems = NULL;
                MPI_Abort(MPI_COMM_WORLD, ERROR);
                exit(ERROR);
            }

            recv_counts_gather_elems = (int32*)malloc(sizeof(int32) * size);
            if( NULL == recv_counts_gather_elems )
            {
                perror("Could not allocate gather counts array!");
                free(ocean->Ocean);
                ocean->Ocean = NULL;
                free(ocean);
                ocean = NULL;
                free(send_counts_elems);
                send_counts_elems = NULL;
                free(displacements_send_elems);
                displacements_send_elems = NULL;
                MPI_Abort(MPI_COMM_WORLD, ERROR);
                exit(ERROR);
            }
            
            displacements_gather_elems = (int32*)malloc(sizeof(int32) * size);
            if(NULL == displacements_gather_elems)
            {
                perror("Could not allocate gather displacements array!");
                free(ocean->Ocean);
                ocean->Ocean = NULL;
                free(ocean);
                ocean = NULL;
                free(send_counts_elems);
                send_counts_elems = NULL;
                free(displacements_send_elems);
                displacements_send_elems = NULL;
                free(recv_counts_gather_elems);
                recv_counts_gather_elems = NULL;
                MPI_Abort(MPI_COMM_WORLD, ERROR);
                exit(ERROR);
             }

            for(node = 0; node < size; node++)
            {   
                interior_rows = ((ocean->rows / size) + ((node < (ocean->rows % size)) ? 1 : 0));
                interior_cols = ocean->cols;

                used_rows = interior_rows + 2;
                used_cols = interior_cols + 2;

                send_counts_elems[node] = used_rows * used_cols;

                displacements_send_elems[node] = ( 0 == node ) ? 0 : matrix_start_offset;

                recv_counts_gather_elems[node] = (used_rows - 1 ) * used_cols; // Process puts only its top border row

                displacements_gather_elems[node] = ( 0 == node ) ? 0 : (displacements_gather_elems[node - 1] + recv_counts_gather_elems[node - 1] - used_cols);

                matrix_start_offset += (interior_rows * used_cols); // bottom row belongs to next process
            #ifdef DEBUG
                printf("Node %d:\n interior rows %d, interior cols %d,\n used rows %d, used cols %d,\n send count %d, send displacement %d,\n gather count %d, gather displacement %d\n", node, interior_rows, interior_cols, used_rows, used_cols, send_counts_elems[node], displacements_send_elems[node], recv_counts_gather_elems[node], displacements_gather_elems[node]);
            #endif
            }

            global_rows = ocean->rows;
            global_cols = ocean->cols;
    #endif
        }

#ifdef SEQUENTIAL_IMPLEMENTATION
        if (MASTER == rank)
        {
            sequential_implementation(ocean);
        }
#endif

#ifdef SHARED_MEMORY_IMPLEMENTATION
        if (MASTER == rank)
        {
            if(1 != size)
            {
                perror("Recommend use of the executable is to run with exactly 1 process for the shared memory implementation.");
                MPI_Abort(MPI_COMM_WORLD, ERROR);
                exit(ERROR);
            }
            shared_memory_implementation(ocean);
        }
#endif

#ifdef MESSAGE_PASSING_IMPLEMENTATION

        if (size < 2)
        {
            perror("Recommend use of the executable is to run with at least 2 processes for the message passing implementation.");
            MPI_Abort(MPI_COMM_WORLD, ERROR);
            exit(ERROR);
        }

        double t_Start = 0.0;
        double t_End = 0.0;

        MPI_Barrier(communicator);
        {
            t_Start = MPI_Wtime();

           message_passing_implementation(ocean, communicator, &global_rows, &global_cols, send_counts_elems, displacements_send_elems, recv_counts_gather_elems, displacements_gather_elems);
        }
        MPI_Barrier(communicator);
        t_End = MPI_Wtime();

        if (MASTER == rank)
        {
            printf("Message passing algorithm time to execute: %lf\n", t_End - t_Start);
        }

#endif

         if (MASTER == rank)
         {
             free(ocean->Ocean);
             ocean->Ocean = NULL;
             free(ocean);
             ocean = NULL;

#ifdef MESSAGE_PASSING_IMPLEMENTATION
             
             free(send_counts_elems);
             send_counts_elems = NULL;
             free(displacements_send_elems);
             displacements_send_elems = NULL;
             free(recv_counts_gather_elems);
             recv_counts_gather_elems = NULL;
             free(displacements_gather_elems);
             displacements_gather_elems = NULL;

#endif

         }
    }
    MPI_Finalize();
}


