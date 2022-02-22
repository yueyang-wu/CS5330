#include <iostream>
#include <sys/time.h>
#include <math.h>
#include <stdlib.h>

using namespace std;

double getTime() {
    struct timeval cur;

    gettimeofday( &cur, NULL );
    return( cur.tv_sec + cur.tv_usec / 1000000.0 );
}

int main() {
    double pStart, pEnd, mStart, mEnd;
    int result;

    // initialize random seed
    srand(time(NULL));

    // generate 10000 numbers
    for (int i = 0; i < 10; i++) {
        int num = rand() % INT_MAX + 1;
        cout << "number: " << num << endl;

        // square 500 times using pow
        double pTime;
        pStart = getTime();
        for (int j = 0; j < 500; j++) {
            pow(num, 2);
        }
        pEnd = getTime();
        pTime = pEnd - pStart;
//        cout << "pTime: " << pTime << endl;

        // square 500 times using multiplication
        double mTime;
        mStart = getTime();
        for (int j = 0; j < 500; j++) {
            pow(num, 2);
        }
        mEnd = getTime();
        mTime = mEnd - mStart;
//        cout << "mTime: " << mTime << endl;

        // compare the results
        // if pow takes longer time, result--
        // if multiplication takes longer time, result++
        // if result is negative, multiplication is better
//        if (pTime > mTime) {
//            result--;
//        } else if (pTime < mTime) {
//            result++;
//        }
        cout << "pTime - mTime: " << pTime - mTime << endl;
    }
//    cout << "result: " << result << endl;
    return 0;
}
