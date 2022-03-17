#include <stdint.h>
#include <cmath>
#include <algorithm>


#define THR11 19.0 / 219.0
#define THR1 20 << 8
#define THR2 30 << 8
#define THR3 18 << 8


float_t limitDenoise(float_t denoise, float_t ref) {
    if (denoise < THR11) {
        return ref;
    } else {
        return denoise;
    }
}


uint16_t fixBorder(uint16_t x) {
    if (((x >> 8) - 16) <= 0) {
        return 65535;
    } else {
        return 0;
    }
}

uint16_t debandMask(float_t dbmask_y, float_t dbmask_u, float_t dbmask_v, uint16_t ref) {
    if (ref > THR3) {
        return std::max(dbmask_y, std::max(dbmask_u, dbmask_v));
    } else {
        return 0;
    }
}


uint16_t debandLuma(uint16_t ref, uint16_t db_verydark, uint16_t db_dark, uint16_t db_bright) {
    if (ref < THR1) {
        return db_verydark;
    } else if (ref < THR2) {
        return db_dark;
    } else {
        return db_bright;
    }
}


uint16_t debandChroma(uint16_t ref, uint16_t db_verydark, uint16_t db_dark, uint16_t db_bright) {
    return db_dark;
}
