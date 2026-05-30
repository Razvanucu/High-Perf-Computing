/*-------------------------------------------
          PROJECT SIDE DEFINITIONS
-------------------------------------------*/
//#define DEBUG
//#define TEST

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
#include "time.h"
#include "Windows.h"

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

typedef struct Matrix_type
{
    uint16 rows;
    uint16 cols;
    int32** data;
}Matrix_type;

/*-------------------------------------------
             MEMORY CONSTANTS
-------------------------------------------*/

/*-------------------------------------------
                PROTOTYPES
-------------------------------------------*/
void test_matrix_read(int argc, char* argv[]);
void test_sqrt();

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
                VARIABLES
-------------------------------------------*/
int32 size = 0;
int32 rank = 0;

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
        // printf("%d\n", (*num));
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


status_file_op_type read_numbers(int32 fd, int32* arr, int32 nums)
{
    /*---------------------------
            VARIABLES
    ---------------------------*/
    int32              indx = 0;
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

/*---------------------------
*
*   MATRIX OPERATIONS
*
---------------------------*/
void matrix_ctor(Matrix_type* const mat, uint16 rows, uint16 cols)
{
    /*---------------------------
            VARIABLES
    ---------------------------*/
    int32* mem_block = NULL;
    uint16 row = 0;
    uint16 row_free = 0;

    mat->rows = rows;
    mat->cols = cols;
    mat->data = (int32**)malloc(rows * sizeof(int32*));

    if (NULL == mat->data)
    {
        perror("Memory allocation failed for matrix data\n");
        exit(EXIT_FAILURE);
    }

    mem_block = (int32*)malloc(rows * cols * sizeof(int32));
    if (NULL == mem_block)
    {
        perror("Memory allocation failed for matrix data block\n");
        free(mat->data);
        mat->data = NULL;
        exit(EXIT_FAILURE);
    }

    for (row = 0; row < rows; row++)
    {
        mat->data[row] = mem_block + row * cols;
    }

}


void matrix_dtor(Matrix_type** const mat)
{
    /*---------------------------
            VARIABLES
    ---------------------------*/
    uint16 row = 0;

    if (NULL == (*mat))
    {
        return;
    }

    if (NULL != (*mat)->data)
    {
        free((*mat)->data[0]); /* free the contiguous block of memory allocated for matrix data */
        (*mat)->data[0] = NULL;

        free((*mat)->data);
        (*mat)->data = NULL;
    }
    free(*mat);
    *mat = NULL;
}


void matrix_populate(Matrix_type* const mat, const int32* const arr, const int32 arr_size)
{
    /*---------------------------
            VARIABLES
    ---------------------------*/
    uint16 col = 0;
    uint16 row = 0;
    uint32 indx = 0;

    assert(arr_size == (mat->rows * mat->cols));

    for (row = 0; row < mat->rows; row++)
    {
        for (col = 0; col < mat->cols; col++)
        {
            mat->data[row][col] = arr[indx++];
        }
    }
}


void matrix_zero(Matrix_type* const mat)
{
    memset(mat->data[0], 0, (mat->rows) * (mat->cols) * (sizeof(int32)));
}


void matrix_add(const Matrix_type* const A, const Matrix_type* const B, Matrix_type* const rez)
{
    /*---------------------------
            VARIABLES
    ---------------------------*/
    uint16 col = 0;
    uint16 row = 0;

    assert(A->rows == B->rows);
    assert(A->cols == B->cols);

    assert(rez->rows == A->rows);
    assert(rez->cols == A->cols);

    for (row = 0; row < A->rows; row++)
    {
        for (col = 0; col < A->cols; col++)
        {
            rez->data[row][col] = A->data[row][col] + B->data[row][col];
        }
    }

}


void matrix_multiply(const Matrix_type* const A, const Matrix_type* const B, Matrix_type* const rez)
{
    /*---------------------------
            VARIABLES
    ---------------------------*/
    uint16 col = 0;
    uint16 k = 0;
    uint16 row = 0;

    assert(A->cols == B->rows);
    assert(rez->rows == A->rows);
    assert(rez->cols == B->cols);

    for (row = 0; row < A->rows; row++)
    {
        for (col = 0; col < B->cols; col++)
        {
            rez->data[row][col] = 0;
            for (k = 0; k < A->cols; k++)
            {
                rez->data[row][col] += A->data[row][k] * B->data[k][col];
            }
        }
    }

}


void matrix_multiply_accumulate(const Matrix_type* const A, const Matrix_type* const B, Matrix_type* const rez)
{
    /*---------------------------
            VARIABLES
    ---------------------------*/
    uint16 col = 0;
    uint16 k = 0;
    uint16 row = 0;

    assert(A->cols == B->rows);
    assert(rez->rows == A->rows);
    assert(rez->cols == B->cols);

    for (row = 0; row < A->rows; row++)
    {
        for (col = 0; col < B->cols; col++)
        {
            for (k = 0; k < A->cols; k++)
            {
                rez->data[row][col] += A->data[row][k] * B->data[k][col];
            }
        }
    }

}


boolean matrix_eq(const Matrix_type* const A, const Matrix_type* const B)
{
    /*---------------------------
            VARIABLES
    ---------------------------*/
    uint16 col = 0;
    uint16 row = 0;

    assert(A->rows == B->rows);
    assert(A->cols == B->cols);

    for (row = 0; row < A->rows; row++)
    {
        for (col = 0; col < A->cols; col++)
        {
            if (A->data[row][col] != B->data[row][col])
            {
                return FALSE;
            }
        }
    }
    return TRUE;

}


boolean matrix_check_split(const Matrix_type* const A, const uint32 block_num_rows, const uint32 block_num_cols)
{
    return (((uint32)A->rows) % block_num_rows == 0) && (((uint32)A->cols) % block_num_cols == 0);
}


void display_matrix(const Matrix_type* const mat, const char* matrix_name)
{
    /*---------------------------
            VARIABLES
    ---------------------------*/
    uint16 col = 0;
    uint16 row = 0;
    printf("\nMatrix %s (%ux%u):\n[",
        ((NULL == matrix_name) ? "" : matrix_name),
        mat->rows, mat->cols);

    for (row = 0; row < mat->rows; row++)
    {
        for (col = 0; col < mat->cols; col++)
        {
            printf("%d ", mat->data[row][col]);
        }
        printf((row == mat->rows - 1) ? "]\n\n" : "\n");
    }

}

/*---------------------------
*
*   MASTER OPERATIONS
*
---------------------------*/
boolean validate_matrix_multiplication(const Matrix_type* const A, const Matrix_type* const B)
{
    return matrix_eq(A, B);
}


void Master_Read_Input(int argc, char* argv[], int32** arrA, int32* numbsA, int32** arrB, int32* numbsB)
{
    int32 fdA = 0, fdB = 0;

    if (EXP_NUM_ARGS == argc)
    {
        fprintf(stderr, "Usage: %s <block_size> <first_matrix_file> <second_matrix_file>\n", argv[0]);
        return;
    }

    open_file(argv[2], &fdA, O_RDONLY, NONE);
    open_file(argv[3], &fdB, O_RDONLY, NONE);

    read_number(fdA, numbsA);
    read_number(fdB, numbsB);

    if (((*numbsA) != (*numbsB))
        || ((*numbsA) <= 0)
        || ((*numbsB) <= 0))
    {
        close_file(fdA);
        close_file(fdB);

        perror("The number of elements in the first matrix is not equal to the number of elements in the second matrix\n");
        return;
    }
    (*arrA) = (int32*)calloc(square((*numbsA)), sizeof(int32));

    if (NULL == arrA)
    {
        close_file(fdA);
        close_file(fdB);
        perror("Could not allocate memory for numbers in the first file");
        return;
    }

    (*arrB) = (int32*)calloc(square((*numbsB)), sizeof(int32));

    if (NULL == arrB)
    {
        free((*arrA));
        (*arrA) = NULL;

        close_file(fdA);
        close_file(fdB);

        perror("Could not allocate memory for number in the second file");
        return;
    }

    read_numbers(fdA, (*arrA), square((*numbsA)));
    read_numbers(fdB, (*arrB), square((*numbsB)));

    close_file(fdA);
    close_file(fdB);

}


void Master_Build_Matrices(const int32* const arrA, const int32 numbsA, const int32* const arrB, const int32 numbsB, Matrix_type** A, Matrix_type** B, Matrix_type** C)
{
    matrix_ctor((*A), numbsA, numbsA);
    matrix_ctor((*B), numbsB, numbsB);
    matrix_ctor((*C), numbsA, numbsB);

    matrix_populate((*A), arrA, square(numbsA));
    matrix_populate((*B), arrB, square(numbsB));
    matrix_zero((*C));

}


boolean Master_Check_Split(const Matrix_type* const A, const Matrix_type* const B, const uint32 A_block_num_rows, const uint32 A_block_num_cols, const uint32 B_block_num_rows, const uint32 B_block_num_cols)
{
    return ((matrix_check_split(A, A_block_num_rows, A_block_num_cols))
        && (matrix_check_split(B, B_block_num_rows, B_block_num_cols)));
}

/*---------------------------
*
*     COMMON OPERATIONS
*
---------------------------*/

uint32 get_sqrt(uint32 num)
{
    /*---------------------------
            VARIABLES
    ---------------------------*/
    uint32 low = 0;
    uint32 high = num / 2;
    uint32 mid = 0;

    if (1 == num)
    {
        return num;
    }

    while (low < high)
    {
        mid = (low + high) / 2;
        if (mid == num / mid)
        {
            return mid;
        }
        else if (mid < num / mid)
        {
            low = mid + 1;
        }
        else
        {
            high = mid - 1;
        }
    }
    return high;
}


void create_2D_cartesian_topology_square(MPI_Comm* comm_cart, const uint32 num_processes, const boolean row_periodic, const boolean col_periodic, const boolean reorder)
{
    /*---------------------------
            VARIABLES
    ---------------------------*/
    int32 dims[NUM_DIMS] = { NONE, NONE };
    uint32 indx = 0;
    int32 periods[NUM_DIMS] = { row_periodic, col_periodic };
    uint32 sqrt_num_processes = get_sqrt(num_processes);

    for (indx = sqrt_num_processes; indx >= 1; indx--)
    {
        if (num_processes % indx == 0)
        {
            dims[0] = (int32)indx;
            dims[1] = (int32)(num_processes / indx);
            break;
        }
    }

    MPI_Cart_create(MPI_COMM_WORLD, NUM_DIMS, dims, periods, reorder, comm_cart);

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

void MPI_CANNON_LAB(int argc, char* argv[])
{
    /*---------------------------
       LOCAL LITERAL CONSTANTS
    ---------------------------*/
#define UPPER_NEIGHBOR  ( 0  )
#define RIGHT_NEIGHBOR  ( 1  )
#define DOWN_NEIGHBOR   ( 2  )
#define LEFT_NEIGHBOR   ( 3  )
#define CEVA_TAG_NEBUN  ( 67 )
#define CEMIL_TAG_NEBUN ( 68 )
#define HAIDE_HAIDE     ( 69 )
#define BAAAAAAAAAA     ( 70 )

    /*---------------------------
            VARIABLES
    ---------------------------*/
    int32 iter = 0;

    /* Global Matrices */
    Matrix_type* A_global = NULL;
    Matrix_type* B_global = NULL;
    Matrix_type* C_global = NULL;

    /* Local Matrices */
    Matrix_type* A_local = NULL;
    Matrix_type* B_local = NULL;
    Matrix_type* C_local = NULL;

    /* Sequential Result Matrix */
    Matrix_type* Rez_seq = NULL;

    /* input files variables */
    int32* arrA = NULL;
    int32* arrB = NULL;
    int32 numbsA = 0;
    int32 numbsB = 0;

    /* Full matrix size */
    int32 N = 0;

    /* Block size for splitting the input matrices into blocks */
    int32  block_size = 0;

    /* Cartesian Communicator */
    MPI_Comm comm_cart;

    /* Node coordinates */
    int32 proc_coord[2] = { NONE, NONE };
    int32 other_proc_coord[2] = { NONE, NONE };

    /* For positioning part */
    rank_type upper_neighbor_rank = 0;
    rank_type right_neighbor_rank = 0;
    rank_type down_neighbor_rank = 0;
    rank_type left_neighbor_rank = 0;

    /* Neigbours in topology */
    rank_type neighbors_ranks[4] = { NONE, NONE, NONE, NONE };

    /* Send counts and displacements for scattering */
    int32* sendcounts = NULL;
    int32* displacements = NULL;

    /* Variables for MPI derived datatypes for matrix A */
    int32 sizes_A[NUM_DIMS] = { NONE, NONE };
    int32 subsizes_A[NUM_DIMS] = { NONE, NONE };
    int32 starts_A[NUM_DIMS] = { NONE, NONE };
    MPI_Datatype A_block_type;
    MPI_Datatype A_temp_type;

    /* Variables for MPI derived datatypes for matrix B */
    int32 sizes_B[NUM_DIMS] = { NONE, NONE };
    int32 subsizes_B[NUM_DIMS] = { NONE, NONE };
    int32 starts_B[NUM_DIMS] = { NONE, NONE };
    MPI_Datatype B_block_type;
    MPI_Datatype B_temp_type;

    /* Variables for MPI derived datatypes for matrix C */
    int32 sizes_C[NUM_DIMS] = { NONE, NONE };
    int32 subsizes_C[NUM_DIMS] = { NONE, NONE };
    int32 starts_C[NUM_DIMS] = { NONE, NONE };
    MPI_Datatype C_block_type;
    MPI_Datatype C_temp_type;

    /* Timers */
    double tStart = 0.0;
    double tEnd = 0.0;

    MPI_Status status;

    MPI_Barrier(MPI_COMM_WORLD);
    {
        /* MASTER Initial part */
        if (MASTER == rank)
        {
            /* Build Matrices */
            Master_Read_Input(argc, argv, &arrA, &numbsA, &arrB, &numbsB);
            N = numbsA;

            A_global = (Matrix_type*)malloc(sizeof(Matrix_type));
            B_global = (Matrix_type*)malloc(sizeof(Matrix_type));
            C_global = (Matrix_type*)malloc(sizeof(Matrix_type));

            Master_Build_Matrices(arrA, numbsA, arrB, numbsB, &A_global, &B_global, &C_global);

#ifdef DEBUG
            display_matrix(A_global, "A_global");
            display_matrix(B_global, "B_global");
            display_matrix(C_global, "C_global");
#endif

            free(arrA);
            arrA = NULL;
            free(arrB);
            arrB = NULL;

            /* Check block size can be used */
            block_size = atoi(argv[1]);
            if ((0 == block_size)
                || (!Master_Check_Split(A_global, B_global, block_size, block_size, block_size, block_size))
                || (N * N !=  size * block_size * block_size))
            {
                matrix_dtor(&A_global);
                matrix_dtor(&B_global);
                matrix_dtor(&C_global);

                printf("Block size: %d\n", block_size);
                perror("Block size cannot be used to split the input matrices into equal blocks\n");

                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

                exit(BAAAAAAAAAA);
            }

            /* Build send counts and displacements */
            sendcounts = (int32*)malloc(sizeof(int32) * size);
            if (NULL == sendcounts)
            {
                matrix_dtor(&A_global);
                matrix_dtor(&B_global);
                matrix_dtor(&C_global);

                perror("Sendcounts could not be allocated\n");

                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

                exit(BAAAAAAAAAA);
            }

            displacements = (int32*)malloc(sizeof(int32) * size);
            if (NULL == displacements)
            {
                matrix_dtor(&A_global);
                matrix_dtor(&B_global);
                matrix_dtor(&C_global);

                free(sendcounts);
                sendcounts = NULL;

                perror("Displacements could not be allocated\n");

                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

                exit(BAAAAAAAAAA);
            }

            /* Calculate sendcounts and displacements */
            for (int32 i = 0; i < N / block_size; i++)
            {
                for (int32 j = 0; j < N / block_size; j++)
                {
                    sendcounts[i * N / block_size + j] = 1;
                    displacements[i * N / block_size + j] = i * N + j;
                }
            }

            /* Calculate Matrix Multiplication using sequential algorithm */
            Rez_seq = (Matrix_type*)malloc(sizeof(Matrix_type));
            matrix_ctor(Rez_seq, numbsA, numbsB);
            matrix_multiply(A_global, B_global, Rez_seq);
        }

        /* Common Part */
        MPI_Barrier(MPI_COMM_WORLD);
        {
            tStart = MPI_Wtime();

            /* Create cartesian topology */
            create_2D_cartesian_topology_square(&comm_cart, size, TRUE, TRUE, TRUE);

            /* Get neighbors ranks in the topology */
            get_neighbors_processes_ranks(comm_cart, 1, &neighbors_ranks[0], &neighbors_ranks[1], &neighbors_ranks[2], &neighbors_ranks[3]);

            /* Bcast block size */
            MPI_Bcast(&block_size, 1, MPI_INT32_T, MASTER, comm_cart);

            /* Bcast full matrix size */
            MPI_Bcast(&N, 1, MPI_INT32_T, MASTER, comm_cart);

            /* Size of the full matrix (we assume that the matrices are square for this lab)*/
            sizes_A[0] = sizes_A[1] = sizes_B[0] = sizes_B[1] = sizes_C[0] = sizes_C[1] = N;

            /* Size of the block */
            subsizes_A[0] = subsizes_A[1] = subsizes_B[0] = subsizes_B[1] = subsizes_C[0] = subsizes_C[1] = block_size;

            /* Starts of the subarray */
            starts_A[0] = starts_A[1] = starts_B[0] = starts_B[1] = starts_C[0] = starts_C[1] = 0;

            /* Create block types */
            MPI_Type_create_subarray(NUM_DIMS, sizes_A, subsizes_A, starts_A, MPI_ORDER_C, MPI_INT32_T, &A_temp_type);
            MPI_Type_create_subarray(NUM_DIMS, sizes_B, subsizes_B, starts_B, MPI_ORDER_C, MPI_INT32_T, &B_temp_type);
            MPI_Type_create_subarray(NUM_DIMS, sizes_C, subsizes_C, starts_C, MPI_ORDER_C, MPI_INT32_T, &C_temp_type);

            MPI_Type_create_resized(A_temp_type, 0, block_size * sizeof(int32), &A_block_type);
            MPI_Type_create_resized(B_temp_type, 0, block_size * sizeof(int32), &B_block_type);
            MPI_Type_create_resized(C_temp_type, 0, block_size * sizeof(int32), &C_block_type);

            MPI_Type_commit(&A_block_type);
            MPI_Type_commit(&B_block_type);
            MPI_Type_commit(&C_block_type);

            /* Initialize local matrices */
            A_local = (Matrix_type*)malloc(sizeof(Matrix_type));
            B_local = (Matrix_type*)malloc(sizeof(Matrix_type));
            C_local = (Matrix_type*)malloc(sizeof(Matrix_type));

            matrix_ctor(A_local, block_size, block_size);
            matrix_ctor(B_local, block_size, block_size);
            matrix_ctor(C_local, block_size, block_size);

            /* Scatter submatrices */
            MPI_Scatterv((MASTER == rank) ? A_global->data[0] : NULL
                , sendcounts, displacements, A_block_type, A_local->data[0], block_size * block_size, MPI_INT32_T, MASTER, comm_cart);

            MPI_Scatterv((MASTER == rank) ? B_global->data[0] : NULL
                , sendcounts, displacements, B_block_type, B_local->data[0], block_size * block_size, MPI_INT32_T, MASTER, comm_cart);

            MPI_Scatterv((MASTER == rank) ? C_global->data[0] : NULL
                , sendcounts, displacements, C_block_type, C_local->data[0], block_size * block_size, MPI_INT32_T, MASTER, comm_cart);

#ifdef DEBUG
            display_matrix(A_local, "A_local");
            display_matrix(B_local, "B_local");
            display_matrix(C_local, "C_local");
#endif

            /* Get process coordinates */
            MPI_Cart_coords(comm_cart, rank, NUM_DIMS, proc_coord);

            /* Get rank of process to send blocks for positioning */
            /* Send up rank */
            other_proc_coord[0] = (proc_coord[0] - proc_coord[1] + (N / block_size)) % (N / block_size);
            other_proc_coord[1] = proc_coord[1];

            MPI_Cart_rank(comm_cart, other_proc_coord, &upper_neighbor_rank);

            /* Receive down rank */
            other_proc_coord[0] = (proc_coord[0] + proc_coord[1]) % (N / block_size);
            other_proc_coord[1] = proc_coord[1];

            MPI_Cart_rank(comm_cart, other_proc_coord, &down_neighbor_rank);

            /* Send left rank */
            other_proc_coord[0] = proc_coord[0];
            other_proc_coord[1] = (proc_coord[1] - proc_coord[0] + (N / block_size)) % (N / block_size);

            MPI_Cart_rank(comm_cart, other_proc_coord, &left_neighbor_rank);

            /* Receive right rank */
            other_proc_coord[0] = proc_coord[0];
            other_proc_coord[1] = (proc_coord[1] + proc_coord[0]) % (N / block_size);

            MPI_Cart_rank(comm_cart, other_proc_coord, &right_neighbor_rank);

            /*----------------------------
                       CANNON
            ----------------------------*/
            /* 1 Positioning */
            /* 1.0 Shift matrix Aij i times left */
            MPI_Sendrecv_replace(A_local->data[0], block_size * block_size, MPI_INT32_T, left_neighbor_rank, CEVA_TAG_NEBUN, right_neighbor_rank, CEVA_TAG_NEBUN, comm_cart, &status);

            /* 1.1 Shift matrix Bij j times up */
            MPI_Sendrecv_replace(B_local->data[0], block_size * block_size, MPI_INT32_T, upper_neighbor_rank, CEMIL_TAG_NEBUN, down_neighbor_rank, CEMIL_TAG_NEBUN, comm_cart, &status);

            /* 2 Iterate */
            for (iter = 0; iter < N / block_size; iter++)
            {
                /* 2.0 Multiplication */
                matrix_multiply_accumulate(A_local, B_local, C_local);

                /* 2.1 Movement       */
                /* 2.1.0 Shift matrix Aik left */
                MPI_Sendrecv_replace(A_local->data[0], block_size * block_size, MPI_INT32_T, neighbors_ranks[LEFT_NEIGHBOR], BAAAAAAAAAA, neighbors_ranks[RIGHT_NEIGHBOR], BAAAAAAAAAA, comm_cart, &status);

                /* 2.1.1 Shift matrix Bkj up   */
                MPI_Sendrecv_replace(B_local->data[0], block_size * block_size, MPI_INT32_T, neighbors_ranks[UPPER_NEIGHBOR], HAIDE_HAIDE, neighbors_ranks[DOWN_NEIGHBOR], HAIDE_HAIDE, comm_cart, &status);
            }

            /* Gather partial results */
            MPI_Gatherv(C_local->data[0], block_size * block_size, MPI_INT32_T,
                (MASTER == rank) ? C_global->data[0] : NULL
                , sendcounts, displacements, C_block_type, MASTER, comm_cart);

            /* Destroy block type */
            MPI_Type_free(&A_temp_type);
            MPI_Type_free(&B_temp_type);
            MPI_Type_free(&C_temp_type);

            MPI_Type_free(&A_block_type);
            MPI_Type_free(&B_block_type);
            MPI_Type_free(&C_block_type);


            /* Free local resources */
            matrix_dtor(&A_local);
            matrix_dtor(&B_local);
            matrix_dtor(&C_local);

        }
        MPI_Barrier(MPI_COMM_WORLD);
        tEnd = MPI_Wtime();

        /* MASTER free global resources */
        if (MASTER == rank)
        {
            printf("Sequential and Cannon's algorithm matrix multiplication results %s\n",
                (validate_matrix_multiplication(Rez_seq, C_global) ? ("match") : ("do not match")));

#ifdef DEBUG
            display_matrix(Rez_seq, "Rez_seq");
            display_matrix(C_global, "C_global");
#endif
            printf("Execution time: %f seconds\n", tEnd - tStart);

            free(sendcounts);
            sendcounts = NULL;

            free(displacements);
            displacements = NULL;

            matrix_dtor(&A_global);
            matrix_dtor(&B_global);
            matrix_dtor(&C_global);
            matrix_dtor(&Rez_seq);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);


    if (MASTER == rank)
    {

    }

    /*----------------------------------
      Undefine local literal constants
    ----------------------------------*/
#undef UPPER_NEIGHBOR
#undef RIGHT_NEIGHBOR
#undef DOWN_NEIGHBOR
#undef LEFT_NEIGHBOR
#undef CEVA_TAG_NEBUN 
#undef CEMIL_TAG_NEBUN
#undef HAIDE_HAIDE 
#undef BAAAAAAAAAA 
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    {
        get_size(size);
        get_rank(rank);

        MPI_CANNON_LAB(argc, argv);

    }
    MPI_Finalize();
}

void test_sqrt()
{
    for (uint32 num = 0; num <= 38; num++)
    {
        uint32 sqrt_num = get_sqrt(num);
        printf("Found the sqrt of %d: %d\n", num, sqrt_num);
        if ((sqrt_num * sqrt_num != num)
            && ((sqrt_num + 1) * (sqrt_num + 1) < num))
        {
            printf("Error: %d is not the floor of the square root of %d\n", sqrt_num, num);
        }
    }
}

void test_matrix_read(int argc, char* argv[])
{
    if (EXP_NUM_ARGS == argc)
    {
        fprintf(stderr, "Usage: %s <block_size> <first_matrix_file> <second_matrix_file>\n", argv[0]);
        return;
    }

    printf("%s\n", argv[2]);
    printf("%s\n", argv[3]);

    int32 fdA, fdB;
    open_file(argv[2], &fdA, O_RDONLY, NONE);
    open_file(argv[3], &fdB, O_RDONLY, NONE);

    int32 numbsA, numbsB;
    int32* arrA, * arrB;

    read_number(fdA, &numbsA);
    read_number(fdB, &numbsB);

    arrA = (int32*)calloc(square(numbsA), sizeof(int32));

    if (NULL == arrA)
    {
        perror("Could not allocate memory for numbers in the first file");
        return;
    }

    arrB = (int32*)calloc(square(numbsB), sizeof(int32));

    if (NULL == arrB)
    {
        perror("Could not allocate memory for number in the second file");
        return;
    }

    read_numbers(fdA, arrA, square(numbsA));
    read_numbers(fdB, arrB, square(numbsB));

    Matrix_type* A = (Matrix_type*)malloc(sizeof(Matrix_type));
    Matrix_type* B = (Matrix_type*)malloc(sizeof(Matrix_type));
    Matrix_type* C = (Matrix_type*)malloc(sizeof(Matrix_type));

    matrix_ctor(A, numbsA, numbsA);
    matrix_ctor(B, numbsB, numbsB);
    matrix_ctor(C, numbsA, numbsB);

    matrix_populate(A, arrA, square(numbsA));
    matrix_populate(B, arrB, square(numbsB));

    display_matrix(A, NULL);
    display_matrix(B, NULL);

    free(arrA);
    arrA = NULL;
    free(arrB);
    arrB = NULL;

    matrix_add(A, B, C);
    display_matrix(C, NULL);
    matrix_multiply(A, B, C);
    display_matrix(C, NULL);

    matrix_dtor(&A);
    matrix_dtor(&B);
    matrix_dtor(&C);

}