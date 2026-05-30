/*-------------------------------------------
                INCLUDES
-------------------------------------------*/
#include "fcntl.h"
#include "io.h"
#include "mpi.h"
#include "stdio.h"
#include "stdlib.h"
#include "sys/stat.h"
#include "time.h"
#include "Windows.h"

/*-------------------------------------------
             LITERAL CONSTANTS
-------------------------------------------*/
#define MASTER          (     0      )
#define ERROR_FILE_OP   (     -1     )
#define DUMMY           (     0      )
#define MAX_UINT32      ( 0xFFFFFFFF )

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

typedef uint8 status_args_type;
enum
{

};

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

typedef struct hash_type
{
    uint32 seed;
    uint32(*hash_func)(uint32, uint32);
};

typedef uint8 count_min_sketch_data_p_type;
enum
{
    BYTE_TYPE,
    WORD_TYPE,
    DWORD_TYPE,
    QWORD_TYPE,

    /* Limits of data types for CMS */
    NUM_TYPES
};

typedef struct count_min_sketch_type
{
    void                         *data_p;
    hash_type                    *hash_funcs_arr;
    uint32                       *counters_matrix;
    uint32                       num_arrays;
    uint32                       num_counters_per_array;
    count_min_sketch_data_p_type data_type;
};

typedef uint8 status_count_min_sketch_op_type;
enum
{
    /*----------------------------------
      Successful Initialization states
    ----------------------------------*/
    CMS_SUCCESSFUL_INITIALIZATION,
    CMS_INITIALIZE_COUNTER_MATRIX_OK,
    CMS_INITIALIZE_COUNTERS_OK,
    CMS_INITIALIZE_HASH_FUNCTIONS_OK,
    CMS_INITIALIZE_DATA_OK,

    /* Limits of faulty initialization states */
    CMS_MAX_STATES_INIT_OK,
    CMS_MIN_STATES_INIT_OK = CMS_SUCCESSFUL_INITIALIZATION,
    CMS_NUM_STATES_INIT_OK = CMS_MAX_STATES_INIT_OK - CMS_MIN_STATES_INIT_OK,

    /*----------------------------------
      Faulty Initialization states
    ----------------------------------*/
    CMS_ALREADY_INITIALIZED = CMS_MAX_STATES_INIT_OK,
    CMS_UNABLE_TO_INITIALIZE,
    CMS_UNABLE_TO_INITIALIZE_COUNTER_MATRIX,
    CMS_UNABLE_TO_INITIALIZE_COUNTER_ROWS,
    CMS_UNABLE_TO_INITIALIZE_HASH_FUNCTIONS,
    CMS_INVALID_DATA_TYPE,
    CMS_UNABLE_TO_INITIALIZE_DATA,

    /* Limits of faulty initialization states */
    CMS_MAX_STATES_INIT_FAULT,
    CMS_MIN_STATES_INIT_FAULT = CMS_ALREADY_INITIALIZED,
    CMS_NUM_STATES_INIT_FAULT = CMS_MAX_STATES_INIT_FAULT - CMS_MIN_STATES_INIT_FAULT,

    /* Free state */
    CMS_INSERT_SUCCESS = CMS_MAX_STATES_INIT_FAULT,

    /* Free state */
    CMS_FREE_OK

};

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

#define rotate_left( num, pos, type )                          \
        ( ( ( num ) << ( pos ) ) |                             \
          ( ( num ) >> ( ( sizeof( type ) << 3 ) - ( pos ) ) ) )

#define rotate_right( num, pos, type)                          \
        ( ( ( num ) >> ( pos ) ) |                             \
          ( ( num ) << ( ( sizeof( type ) << 3 ) - ( pos ) ) ) )

#define pow_2( exp )   \
        ( 1 << ( exp ) )

#define mod_pow_2( num, power_of_2 )\
        ( num & ( power_of_2 - 1 ) )

/*-------------------------------------------
                VARIABLES
-------------------------------------------*/
int size = 0;
int rank = 0;

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


status_file_op_type read_number(int32 fd, uint32* num)
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
    status_file_op_type status = FILE_READ_OK;

    (*num) = 0;
    while (((bytes_read = _read(fd, &byte, 1)) == 1) && (!is_digit(byte))); /* skip non-numeric characters */

    switch (bytes_read)
    {
    case -1:
        status = FILE_READ_ERROR;
        break;
    case 0:
        status = FILE_READ_REACH_END;
        break;
    default:
        (is_digit(byte)) ? build_num((*num), byte) : status = FILE_READ_REACH_END;

        if (FILE_READ_OK == status)
        {
            while (((bytes_read = _read(fd, &byte, 1)) == 1) && (is_digit(byte)))
            {
                build_num((*num), byte);
            }
        }
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


status_file_op_type read_numbers(int32 fd, uint32* arr, uint32 nums)
{
    /*---------------------------
            VARIABLES
    ---------------------------*/
    uint32              indx   = 0;
    status_file_op_type status = FILE_READ_OK;

    for (indx = 0; (indx < nums) && (FILE_READ_OK == status); indx++)
    {
        arr[indx] = 0;
        status = read_number(fd, &arr[indx]);
    }

    return status;

}


status_file_op_type close_file(int32 fd)
{
    if ((_close(fd)) == ERROR_FILE_OP)
    {
        return FILE_CLOSE_ERROR;
    }

    return FILE_CLOSE_OK;

}


uint32 murmur3_hash(uint32 key, uint32 seed)
{
    /*---------------------------
            VARIABLES
    ---------------------------*/
    uint32 hash = 0;

    /* Key mix */
    key = (uint32)(((uint64)key) * 0x00000000CC9E2D51);
    key = rotate_left(key, 15, uint32);
    key = (uint32)(((uint64)key) * 0x000000001B873593);

    /* hash mixed into key */
    hash = seed ^ key;
    hash = rotate_left(hash, 13, uint32);
    hash = (uint32)(((uint64)hash) * 0x0000000000000005);
    hash = hash + 0xE6546B64;
    hash ^= sizeof(uint32);

    hash ^= (hash >> 16);
    hash = (uint32)(((uint64)hash) * 0x0000000085EBCA6B);
    hash ^= (hash >> 13);
    hash = (uint32)(((uint64)hash) * 0x00000000C2B2AE35);
    hash ^= (hash >> 16);

    return hash;

}


static status_count_min_sketch_op_type initialize_counters(count_min_sketch_type** count_min_sketch, uint32 num_arrays, uint32 num_counters_per_array)
{
    /*---------------------------
           VARIABLES
    ---------------------------*/
    uint32  i = 0;
    uint32  j = 0;

    (*count_min_sketch)->counters_matrix = (uint32*)calloc(num_arrays * num_counters_per_array, sizeof(uint32));

    if (NULL == (*count_min_sketch)->counters_matrix)
    {
        return CMS_UNABLE_TO_INITIALIZE_COUNTER_MATRIX;
    }

    return CMS_INITIALIZE_COUNTERS_OK;

}


static status_count_min_sketch_op_type initialize_hash_funcs_array(count_min_sketch_type** count_min_sketch, uint32 num_arrays)
{
    /*---------------------------
            VARIABLES
    ---------------------------*/
    uint32  i = 0;

    (*count_min_sketch)->hash_funcs_arr = (hash_type*)malloc(sizeof(hash_type) * num_arrays);

    if (NULL == (*count_min_sketch)->hash_funcs_arr)
    {
        return CMS_UNABLE_TO_INITIALIZE_HASH_FUNCTIONS;
    }

    for (i = 0; i < num_arrays; i++)
    {
        (*count_min_sketch)->hash_funcs_arr[i].hash_func = murmur3_hash;
    }

    return CMS_INITIALIZE_HASH_FUNCTIONS_OK;

}


static status_count_min_sketch_op_type initialize_data(count_min_sketch_type** count_min_sketch, count_min_sketch_data_p_type data_type)
{

    switch (data_type)
    {
    case BYTE_TYPE:
        (*count_min_sketch)->data_p = malloc(sizeof(byte_type));
        break;
    case WORD_TYPE:
        (*count_min_sketch)->data_p = malloc(sizeof(word_type));
        break;
    case DWORD_TYPE:
        (*count_min_sketch)->data_p = malloc(sizeof(dword_type));
        break;
    case QWORD_TYPE:
        (*count_min_sketch)->data_p = malloc(sizeof(qword_type));
        break;
    default:
        return CMS_INVALID_DATA_TYPE;

    }

    if (NULL == (*count_min_sketch)->data_p)
    {
        return CMS_UNABLE_TO_INITIALIZE_DATA;
    }

    return CMS_INITIALIZE_DATA_OK;

}


status_count_min_sketch_op_type initialize(count_min_sketch_type** count_min_sketch, uint32 num_arrays, uint32 num_counters_per_array, count_min_sketch_data_p_type data_type)
{
    /*---------------------------
          MACROS
    ---------------------------*/
#define free_cms( cms )\
        free( cms ); \
        cms = NULL;

#define free_counters_matrix( cms )      \
        free( ( cms )->counters_matrix );\
        ( cms )->counters_matrix = NULL;

#define free_hash_funcs_array( cms )    \
        free( ( cms )->hash_funcs_arr );\
        ( cms )->hash_funcs_arr = NULL;

#define free_data( cms )        \
        free( ( cms )->data_p );\
        ( cms )->data_p = NULL;

    /*---------------------------
           VARIABLES
    ---------------------------*/
    uint32                                    i = 0;
    status_count_min_sketch_op_type status_init = CMS_INITIALIZE_COUNTERS_OK;

    /* Check if not already initialized */
    if (NULL != (*count_min_sketch))
    {
        status_init = CMS_ALREADY_INITIALIZED;
    }

    /* Structure initialization */
    if (status_init < CMS_MAX_STATES_INIT_OK)
    {
        (*count_min_sketch) = (count_min_sketch_type*)malloc(sizeof(count_min_sketch_type));

        if (NULL == (*count_min_sketch))
        {
            status_init = CMS_UNABLE_TO_INITIALIZE;
        }
        else
        {
            /* Set fields */
            (*count_min_sketch)->data_type = data_type;
            (*count_min_sketch)->num_arrays = num_arrays;
            (*count_min_sketch)->num_counters_per_array = num_counters_per_array;
        }
    }

    /* Counter arrays initialization */
    if (status_init < CMS_MAX_STATES_INIT_OK)
    {
        status_init = initialize_counters(count_min_sketch, num_arrays, num_counters_per_array);
    }

    /* Hash functions initialization */
    if (status_init < CMS_MAX_STATES_INIT_OK)
    {
        status_init = initialize_hash_funcs_array(count_min_sketch, num_arrays);
    }

    /* Data initialization */
    if (status_init < CMS_MAX_STATES_INIT_OK)
    {
        status_init = initialize_data(count_min_sketch, data_type);
    }

    switch (status_init)
    {
    case CMS_INVALID_DATA_TYPE:
    case CMS_UNABLE_TO_INITIALIZE_DATA:
        free_hash_funcs_array((*count_min_sketch));
    case CMS_UNABLE_TO_INITIALIZE_HASH_FUNCTIONS:
        free_counters_matrix((*count_min_sketch));
    case CMS_UNABLE_TO_INITIALIZE_COUNTER_MATRIX:
        free_cms((*count_min_sketch));
        break;
    case CMS_UNABLE_TO_INITIALIZE:
    case CMS_ALREADY_INITIALIZED:
        break;
    }

    /*---------------------------
         UNDEFINE MACROS
    ---------------------------*/
#undef free_cms
#undef free_counters_matrix
#undef free_counters_arrays
#undef free_hash_funcs_array
#undef free_data

    return ((status_init < CMS_MAX_STATES_INIT_OK) ? CMS_SUCCESSFUL_INITIALIZATION : status_init);

}

status_count_min_sketch_op_type free_count_min_sketch(count_min_sketch_type* count_min_sketch)
{
    /*---------------------------
              MACROS
    ---------------------------*/
#define free_cms( cms )\
        free( cms ); \
        cms = NULL;

#define free_counters_matrix( cms )      \
        free( ( cms )->counters_matrix );\
        ( cms )->counters_matrix = NULL;

#define free_hash_funcs_array( cms )    \
        free( ( cms )->hash_funcs_arr );\
        ( cms )->hash_funcs_arr = NULL;

#define free_data( cms )        \
        free( ( cms )->data_p );\
        ( cms )->data_p = NULL;

    /*---------------------------
           VARIABLES
    ---------------------------*/
    uint32           i = 0;
    uint32  num_arrays = (count_min_sketch)->num_arrays;

    free_data(count_min_sketch);
    free_hash_funcs_array(count_min_sketch);
    free_counters_matrix(count_min_sketch);
    free_cms(count_min_sketch);

    /*---------------------------
         UNDEFINE MACROS
    ---------------------------*/
#undef free_cms
#undef free_counters_matrix
#undef free_counters_arrays
#undef free_hash_funcs_array
#undef free_data

    return CMS_FREE_OK;

}


void get_hash_funcs_seeds(count_min_sketch_type* cms, uint32* seeds_arr)
{
    /*---------------------------
              VARIBLES
    ---------------------------*/
    uint32 i = 0;

    for (i = 0; i < cms->num_arrays; i++)
    {
        seeds_arr[i] = cms->hash_funcs_arr[i].seed;
    }
}


void initialize_hash_funcs_seed(count_min_sketch_type** cms)
{
    /*---------------------------
              VARIBLES
    ---------------------------*/
    uint32 i = 0;

    for (i = 0; i < (*cms)->num_arrays; i++)
    {
        (*cms)->hash_funcs_arr[i].seed = (uint32)rand();
    }

}


status_count_min_sketch_op_type insert(count_min_sketch_type* count_min_sketch, uint32 x)
{
    /*---------------------------
              VARIBLES
    ---------------------------*/
    uint32 hash_rez = 0;
    uint32 i = 0;
    uint32 num_counters = count_min_sketch->num_counters_per_array;
    uint32 seed = 0;

    for (i = 0; i < count_min_sketch->num_arrays; i++)
    {
        seed = count_min_sketch->hash_funcs_arr[i].seed;
        hash_rez = count_min_sketch->hash_funcs_arr[i].hash_func(x, seed);
        hash_rez %= count_min_sketch->num_counters_per_array;
        count_min_sketch->counters_matrix[i * num_counters + hash_rez]++;
    }

    return CMS_INSERT_SUCCESS;
}


uint32 estimate_count(count_min_sketch_type* count_min_sketch, uint32 x)
{
    /*---------------------------
              VARIBLES
    ---------------------------*/
    uint32    count = 0;
    uint32 hash_rez = 0;
    uint32        i = 0;
    uint32 estimate = MAX_UINT32;
    uint32 num_counters = count_min_sketch->num_counters_per_array;
    uint32     seed = 0;

    for (i = 0; i < count_min_sketch->num_arrays; i++)
    {
        seed = count_min_sketch->hash_funcs_arr[i].seed;
        hash_rez = count_min_sketch->hash_funcs_arr[i].hash_func(x, seed);
        hash_rez %= count_min_sketch->num_counters_per_array;
        count = count_min_sketch->counters_matrix[i * num_counters + hash_rez];
        estimate = (estimate > count) ? count : estimate;
    }

    return estimate;

}


uint32* Master_Read_Input(int argc, char* argv[], uint32* numbs)
{
    /*---------------------------
              VARIABLES
    ---------------------------*/
    uint32 *arr = NULL;
    int32 fd_inp = 0;
    uint32 indx = 0;
    status_file_op_type status = FILE_OP_OK;

    printf("Master: File path to read %s\n", argv[3]);
    status = open_file(argv[3], &fd_inp, _O_RDONLY, _S_IREAD);

    if (FILE_OPEN_OK == status)
    {
        status = read_number(fd_inp, numbs);
    }

    if (FILE_READ_OK == status)
    {
        printf("Master: Numbers to read %u\n", (*numbs) );
        arr = (uint32*)malloc(sizeof(uint32) * (*numbs));

        if (NULL != arr)
        {
            status = read_numbers(fd_inp, arr, (*numbs));
        }
    }

    status = close_file(fd_inp);

    if (FILE_CLOSE_OK != status)
    {
        return NULL;
    }

    return arr;

}


void Master_Build_Displs_and_Send_Counts_Arrs(uint32 numbs, uint32 size, int32* displs_arr, int32* send_counts_arr)
{
    /*---------------------------
              VARIABLES
    ---------------------------*/
    uint32 indx = 0;
    uint32 base_numbs_per_proc = numbs / size;
    uint32 rem_numbs = numbs % size;

    for (indx = 0; indx < size; indx++)
    {
        send_counts_arr[indx] = (MASTER == indx) ? (base_numbs_per_proc + rem_numbs) : base_numbs_per_proc;
        displs_arr[indx] = ( (indx == 0) ? 0 : (displs_arr[indx - 1] ) + send_counts_arr[indx - 1]);
    }
}


void set_hash_funcs_seed(count_min_sketch_type* cms, uint32* seeds)
{
    /*---------------------------
              VARIBLES
    ---------------------------*/
    uint32 i = 0;
    for (i = 0; i < cms->num_arrays; i++)
    {
        cms->hash_funcs_arr[i].seed = seeds[i];
    }
}


void Insert_Numbers_Into_CMS(count_min_sketch_type* cms, uint32* arr, uint32 numbs)
{
    /*---------------------------
              VARIABLES
    ---------------------------*/
    uint32 indx = 0;

    for (indx = 0; indx < numbs; indx++)
    {
        insert(cms, arr[indx]);
    }
}

void Count_Min_Sketch_Lab(int argc, char* argv[])
{
    /*---------------------------
            VARIABLES
    ---------------------------*/
    uint32 *arr = NULL;
    uint32 C = 0;
    int32  *displs_arr = NULL;
    uint32 k = 0;
    uint32 *local_arr = NULL;
    uint32 numbs = 0;
    uint32 recv_counts = 0;
    uint32* seeds = NULL;
    int32  *send_counts_arr = NULL;
    count_min_sketch_type *cms = NULL;
    count_min_sketch_type *global_cms = NULL;
    status_count_min_sketch_op_type status_init = CMS_SUCCESSFUL_INITIALIZATION;
    double tStart = 0.0;
    double tEnd = 0.0;

    get_rank(rank);
    get_size(size);

    /* MASTER PART */
    if (MASTER == rank)
    {
        arr = Master_Read_Input(argc, argv, &numbs);

        if (NULL == arr)
        {
            MPI_Abort(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);
            return;
        }

        displs_arr = (int32*)malloc(sizeof(int32) * size);

        if( NULL == displs_arr )
        {
            free(arr);
            MPI_Abort(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);
            return;
        }

        send_counts_arr = (int32*)malloc(sizeof(int32) * size);

        if( NULL == send_counts_arr )
        {
            free(arr);
            free(displs_arr);
            MPI_Abort(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);
            return;
        }

        Master_Build_Displs_and_Send_Counts_Arrs(numbs, size, displs_arr, send_counts_arr);

        k = (uint32)atoi(argv[1]);
        C = (uint32)atoi(argv[2]);

        status_init = initialize(&global_cms, k, C, DWORD_TYPE);

        if (CMS_SUCCESSFUL_INITIALIZATION != status_init)
        {
            free(arr);
            free(displs_arr);
            free(send_counts_arr);

            MPI_Abort(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);
            return;
        }

        initialize_hash_funcs_seed(&global_cms);
    }

    /* COMMON PART */
    MPI_Barrier(MPI_COMM_WORLD);
    {
        tStart = MPI_Wtime();
        // 0 -> Broadcast number of elements in input file
        MPI_Bcast(&numbs, 1, MPI_UINT32_T, MASTER, MPI_COMM_WORLD);

        // 1 -> Adjust receive elements count for each process
        if( MASTER == rank)
        {
            recv_counts = send_counts_arr[rank];
        }
        else
        {
            recv_counts = numbs / size;
        }

        // 2 -> Allocate local array for each process
        local_arr = (uint32*)malloc(sizeof(uint32) * recv_counts);
        if( NULL == local_arr )
        {
            if( MASTER == rank )
            {
                free(arr);
                free(displs_arr);
                free(send_counts_arr);
            }
            MPI_Abort(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);
            return;
        }

        // 3 -> Scatter input numbers to all processes
        MPI_Scatterv(arr, send_counts_arr, displs_arr, MPI_UINT32_T, local_arr, recv_counts, MPI_UINT32_T, MASTER, MPI_COMM_WORLD);
        
        // 4 -> Bcast CMS params to all processes
        MPI_Bcast(&k, 1, MPI_UINT32_T, MASTER, MPI_COMM_WORLD);
        MPI_Bcast(&C, 1, MPI_UINT32_T, MASTER, MPI_COMM_WORLD);        

        // 4.1.1 -> Init CMS structure on each process
        status_init = initialize(&cms, k, C, DWORD_TYPE);

        if (CMS_SUCCESSFUL_INITIALIZATION != status_init)
        {
            if (MASTER == rank)
            {
                free(arr);
                free(displs_arr);
                free(send_counts_arr);
            }
            free(seeds);
            MPI_Abort(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);
            return;
        }

        // 4.2 -> Bcast hash functions seeds to all processes
        seeds = (uint32*)malloc(sizeof(uint32) * k);

        if (NULL == seeds)
        {
            if (MASTER == rank)
            {
                free(arr);
                free(displs_arr);
                free(send_counts_arr);
            }
            MPI_Abort(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);
            return;
        }

        if (MASTER == rank)
        {
            get_hash_funcs_seeds(global_cms, seeds);
        }
    
        MPI_Bcast(seeds, k, MPI_UINT32_T, MASTER, MPI_COMM_WORLD);
        
        // 4.3 -> Set hash functions seeds for each process CMS
        set_hash_funcs_seed(cms, seeds);
        
        // 5 -> Insert numbers into CMS
        Insert_Numbers_Into_CMS(cms, local_arr, recv_counts);
        
        // 6 -> Gather CMS from all processes to master
        MPI_Reduce( cms->counters_matrix, 
                    (rank == MASTER) ? global_cms->counters_matrix : NULL, 
                    (cms->num_arrays) * (cms->num_counters_per_array), 
                    MPI_UINT32_T, 
                    MPI_SUM, 
                    MASTER, 
                    MPI_COMM_WORLD);

        // 7 -> Free resources
        if( MASTER == rank )
        {
            free(arr);
            free(displs_arr);
            free(send_counts_arr);
            free_count_min_sketch(global_cms);
        }

        free(seeds);
        free(local_arr);
        free_count_min_sketch(cms);

    }
    MPI_Barrier(MPI_COMM_WORLD);
    tEnd = MPI_Wtime();

    if(MASTER == rank)
    {
        printf("Execution time: %f seconds\n", tEnd - tStart);
    }

}

int main(int argc, char* argv[])
{
    srand((unsigned)time(NULL));

    /* START MPI Program */
    double tStart = 0.0;
    double tEnd = 0.0;

    MPI_Init(&argc, &argv);
    {
       Count_Min_Sketch_Lab(argc, argv);
    }
    MPI_Finalize();
}