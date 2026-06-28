export const TYPE_SIZE_MAP: Record<number, number> = {
    0: 0, // UNDEFINED
    1: 4, // FLOAT
    2: 1, // UINT8
    3: 1, // INT8
    4: 2, // UINT16
    5: 2, // INT16
    6: 4, // INT32
    7: 8, // INT64
    8: -1, // STRING
    9: 1, // BOOL
    10: 2, // FLOAT16
    11: 8, // DOUBLE
    12: 4, // UINT32
    13: 8, // UINT64
    14: 8, // COMPLEX64
    15: 16, // COMPLEX128
    16: 2, // BFLOAT16
    17: 1, // FLOAT8E4M3FN
    18: 1, // FLOAT8E4M3FNUZ
    19: 2, // FLOAT8E5M2
    20: 2, // FLOAT8E5M2FNUZ
    21: 1, // UINT4
    22: 1, // INT4
};

export const SCALAR_SHAPE: number[] = [];
export const UNKNOWN_SHAPE = [-1];

export const BASE_TEN = 10;
