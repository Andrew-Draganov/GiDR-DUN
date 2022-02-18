// https://martin.ankerl.com/2007/10/04/optimized-pow-approximation-for-java-and-c-c/
#include <stdio.h>
#include <math.h>
double fastPow(double a, double b) {
    union {
        double d;
        int x[2];
    } u = { a };
    if(b == 1.0){
        return a;
    }
    u.x[1] = (int)(b * (u.x[1] - 1072632447) + 1072632447);
    u.x[0] = 0;
    return u.d;
}

float my_fmin(float a, float b) {
    return fmin(a, b);
}
float my_fmax(float a, float b) {
    return fmax(a, b);
}
