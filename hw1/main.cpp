#include <iostream>
#include <sys/time.h>
#include <math.h>
#include <stdlib.h>

using namespace std;

/*
 * capture the current time
 */
double getTime() {
    struct timeval cur;

    gettimeofday( &cur, NULL );
    return( cur.tv_sec + cur.tv_usec / 1000000.0 );
}

/*
 * Randomly generate 1,000 integer between 1 and 2,147,483,647.
 * Calculate the square of each integer using pow and multiplication for 10,000 times each.
 * Record the time of doing each 10,000 operations and compare
 * Print out the result
 */
int compareInteger() {
    cout << "*** compare integer ***" << endl;
    int result = 0;

    // initialize random seed
    srand(time(NULL));

    // generate 1000 numbers between 1 and 2147483647
    for (int i = 1; i <= 1000; i++) {
        int num = rand() % INT_MAX + 1;
        cout << i << ". number: " << num << endl;

        // square 10,000 times using pow
        double pStart = getTime();
        for (int j = 0; j < 10000; j++) {
            pow(num, 2);
        }
        double pEnd = getTime();
        double pTime = pEnd - pStart;

        // square 10,000 times using multiplication
        double mStart = getTime();
        for (int j = 0; j < 10000; j++) {
            int res = num * num;
        }
        double mEnd = getTime();
        double mTime = mEnd - mStart;

        // display the difference between the time using pow and time using multiplication
        cout << "pTime - mTime: " << pTime - mTime << endl;
        cout << endl;

        // count the results
        // if pow takes longer time, result--
        // if multiplication takes longer time, result++
        // if result is negative, multiplication is better
        if (pTime > mTime) {
            result--;
        } else if (pTime < mTime) {
            result++;
        }
    }
    cout << "result: " << result << endl;
    return 0;
}

/*
 * Randomly generate 1,000 number between 0.0001 and 9999.9999.
 * Calculate the square of each number using pow and multiplication for 10,000 times each.
 * Record the time of doing each 10,000 operations and compare
 * Print out the result
 */
int compareFloat() {
    cout << "*** compare float number ***" << endl;
    int result = 0;

    // initialize random seed
    srand(time(NULL));

    // generate 1,000 number between 0.0001 and 9999.9999
    for (int i = 1; i <= 1000; i++) {
        float num = static_cast <float> (rand()) / static_cast <float> (RAND_MAX / (9999.9999 - 0.0001));
        cout << i << ". number: " << num << endl;

        // square 10,000 times using pow
        double pStart = getTime();
        for (int j = 0; j < 10000; j++) {
            pow(num, 2);
        }
        double pEnd = getTime();
        double pTime = pEnd - pStart;

        // square 10,000 times using multiplication
        double mStart = getTime();
        for (int j = 0; j < 10000; j++) {
            float res = num * num;
        }
        double mEnd = getTime();
        double mTime = mEnd - mStart;

        // display the difference between the time using pow and time using multiplication
        cout << "pTime - mTime: " << pTime - mTime << endl;
        cout << endl;

        // count the results
        // if pow takes longer time, result--
        // if multiplication takes longer time, result++
        // if result is negative, multiplication is better
        if (pTime > mTime) {
            result--;
        } else if (pTime < mTime) {
            result++;
        }
    }
    cout << "result: " << result << endl;
    return 0;
}

int main() {
    compareInteger();
    compareFloat();
    return 0;
}
