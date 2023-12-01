#include <bits/stdc++.h>

using namespace std;

namespace KopeliovichStream {

/**
 * Author: Sergey Kopeliovich (Burunduk30@gmail.com)
 */

#define VERSION "0.1.8"

#include <algorithm>
#include <cassert>
#include <cstdio>

/** Fast allocation */

#ifdef FAST_ALLOCATOR_MEMORY
    int allocator_pos = 0;
    char allocator_memory[(int)FAST_ALLOCATOR_MEMORY];

    inline void *operator new(size_t n) {
        // fprintf(stderr, "n=%ld\n", n);
        char *res = allocator_memory + allocator_pos;
        assert(n <= (size_t)((int)FAST_ALLOCATOR_MEMORY - allocator_pos));
        allocator_pos += n;
        return (void *)res;
    }

    inline void operator delete(void *) noexcept {
    }

    inline void operator delete(void *, size_t) noexcept {
    }

    // inline void * operator new [] (size_t) { assert(0); }
    // inline void operator delete [] (void *) { assert(0); }
#endif

/** Fast input-output */

    template<class T = int>
    inline T readInt();

    inline double readDouble();

    inline int readUInt();

    inline int readChar();  // first non-blank character
    inline void readWord(char *s);

    inline bool readLine(char *s);  // do not save '\n'
    inline bool isEof();

    inline int getChar();

    inline int peekChar();

    inline bool seekEof();

    inline void skipBlanks();

    template<class T>
    inline void writeInt(T x, char end = 0, int len = -1);

    inline void writeChar(int x);

    inline void writeWord(const char *s);

    inline void
    writeDouble(double x, int len = 10);  // works correct only for |x| < 2^{63}
    inline void flush();

    static struct buffer_flusher_t {
        ~buffer_flusher_t() {
            flush();
        }
    } buffer_flusher;

/** Read */

    static const int buf_size = 4096 * 4;

    static unsigned char buf[buf_size];
    static int buf_len = 0, buf_pos = 0;

    inline bool isEof() {
        if (buf_pos == buf_len) {
            buf_pos = 0, buf_len = fread(buf, 1, buf_size, stdin);
            if (buf_pos == buf_len)
                return 1;
        }
        return 0;
    }

    inline int getChar() {
        return isEof() ? -1 : buf[buf_pos++];
    }

    inline int peekChar() {
        return isEof() ? -1 : buf[buf_pos];
    }

    inline bool seekEof() {
        int c;
        while ((c = peekChar()) != -1 && c <= 32)
            buf_pos++;
        return c == -1;
    }

    inline void skipBlanks() {
        while (!isEof() && buf[buf_pos] <= 32U)
            buf_pos++;
    }

    inline int readChar() {
        int c = getChar();
        while (c != -1 && c <= 32)
            c = getChar();
        return c;
    }

    inline int readUInt() {
        int c = readChar(), x = 0;
        while ('0' <= c && c <= '9')
            x = x * 10 + (c - '0'), c = getChar();
        return x;
    }

    template<class T>
    inline T readInt() {
        int minus = 0, c = readChar();
        T x = 0;
        if (c == '-')
            minus = 1, c = getChar();
        else if (c == '+')
            c = getChar();
        for (; '0' <= c && c <= '9'; c = getChar())
            if (minus)
                x = x * 10 - (c - '0');  // take care about -2^{31}
            else
                x = x * 10 + (c - '0');
        return x;
    }

    inline double readDouble() {
        int s = 1, c = readChar();
        double x = 0;
        if (c == '-')
            s = -1, c = getChar();
        while ('0' <= c && c <= '9')
            x = x * 10 + (c - '0'), c = getChar();
        if (c == '.') {
            c = getChar();
            double coef = 1;
            while ('0' <= c && c <= '9')
                x += (c - '0') * (coef *= 1e-1), c = getChar();
        }
        return s == 1 ? x : -x;
    }

    inline void readWord(char *s) {
        int c = readChar();
        while (c > 32)
            *s++ = c, c = getChar();
        *s = 0;
    }

    inline bool readLine(char *s) {
        int c = getChar();
        while (c != '\n' && c != -1)
            *s++ = c, c = getChar();
        *s = 0;
        return c != -1;
    }

/** Write */

    static int write_buf_pos = 0;
    static char write_buf[buf_size];

    inline void writeChar(int x) {
        if (write_buf_pos == buf_size)
            fwrite(write_buf, 1, buf_size, stdout), write_buf_pos = 0;
        write_buf[write_buf_pos++] = x;
    }

    inline void flush() {
        if (write_buf_pos) {
            fwrite(write_buf, 1, write_buf_pos, stdout), write_buf_pos = 0;
            fflush(stdout);
        }
    }

    template<class T>
    inline void writeInt(T x, char end, int output_len) {
        if (x < 0)
            writeChar('-');

        char s[24];
        int n = 0;
        while (x || !n)
            s[n++] = '0' + abs((int) (x % 10)),
                    x /= 10;  // `abs`: take care about -2^{31}
        while (n < output_len)
            s[n++] = '0';
        while (n--)
            writeChar(s[n]);
        if (end)
            writeChar(end);
    }

    inline void writeWord(const char *s) {
        while (*s)
            writeChar(*s++);
    }

    inline void writeDouble(double x, int output_len) {
        if (x < 0)
            writeChar('-'), x = -x;
        assert(x <= (1LLU << 63) - 1);
        long long t = (long long) x;
        writeInt(t), x -= t;
        writeChar('.');
        for (int i = output_len - 1; i > 0; i--) {
            x *= 10;
            t = std::min(9, (int) x);
            writeChar('0' + t), x -= t;
        }
        x *= 10;
        t = std::min(9, (int) (x + 0.5));
        writeChar('0' + t);
    }

}

/// test J
/// 1 2
/// 2 266
/// 3 497
/// 4 800
/// 5 356
/// 6 150
/// 7 120
/// 8 160
/// 9 709
/// 10 180
/// 11 255
/// 12 380
/// 13 80
/// 14 821
/// 15 185
/// 16 571
/// 17 354
/// 18 150
/// 19 120
/// 20 160
/// 21 829
/// 22 181
/// 23 127
/// 24 255
/// 25 80
/// 26 837
/// 27 184
/// 28 282
/// 29 355
/// 30 150
/// 31 120
/// 32 160
/// 33 1654
/// 34 181
/// 35 254
/// 36 250
/// 37 80
/// 38 1657
/// 39 185
/// 40 805
/// 41 358
/// 42 150
/// 43 120
/// 44 160
/// 45 834
/// 46 182
/// 47 255
/// 48 380
/// 49 80
/// 50 825
/// 51 184

using namespace KopeliovichStream;

#include <chrono>

using namespace std::chrono;

bool is_spoiled(double num) {
    return std::isnan(num) || std::isinf(num);
}

bool high_equal(double x, double y) {
    if (is_spoiled(x) || is_spoiled(y)) {
        exit(1);
    }
    return abs(x - y) <= 1e-9 * max({1.0, abs(x), abs(y)});
}

//#define FAST_STREAM

//#define PRINT_DEBUG_INFO

//#define PRINT_SEARCH_INFO

//#define VERIFY_DP

//#define DEBUG_MODE

#ifdef DEBUG_MODE

#define FAILED_ASSERT(message)                                                \
    {                                                                         \
        std::cout << "assert failed at " __FILE__ << ":" << __LINE__ << '\n'; \
        std::cout << "message: \"" << (message) << "\"" << endl;                    \
        std::exit(1);                                                         \
    }

#define ASSERT(condition, message) \
    if (!(condition))              \
    FAILED_ASSERT(message)

//_STL_VERIFY(condition, message) // don't work on CLion

#else

#define ASSERT(condition, message)// condition

#endif// DEBUG_MODE

double TOTAL_TIME = 0;
time_point<steady_clock> time_start;

void my_time_start() {
#ifndef FAST_STREAM
    time_start = steady_clock::now();
#endif
}

void my_time_end() {
#ifndef FAST_STREAM
    auto time_stop = steady_clock::now();
    auto duration = time_stop - time_start;
    double time = duration_cast<nanoseconds>(duration).count() / 1e9;
    TOTAL_TIME += time;
#endif
}

struct request_t {
    int TBS;
    int n;
    int t0;
    int t1;
    int ost_len;
};

struct Solution {
    int N;
    int K;
    int T;
    int R;
    int J;

    // s0[t][n][k][r]
    vector<vector<vector<vector<double>>>> s0;

    // s0[t][k][r][n]
    vector<vector<vector<vector<double>>>> s0_tkrn;

    // d[n][m][k][r]
    vector<vector<vector<vector<double>>>> d;

    // exp_d[n][m][k][r]
    vector<vector<vector<vector<double>>>> exp_d;

    // exp_d_2[n][k][r][m]
    vector<vector<vector<vector<double>>>> exp_d_2;

    // exp_d_pow[n][m][k][r]
    vector<vector<vector<vector<double>>>> exp_d_pow;

    vector<request_t> requests;

    // p[t][n][k][r]
    vector<vector<vector<vector<double>>>> p;

    // total_g[j]
    vector<double> total_g;

    // add_g[t][n]
    vector<vector<double>> add_g;

    void read(
#ifndef FAST_STREAM
            std::istream &input
#endif
    ) {

#ifdef FAST_STREAM
        N = readInt();
        K = readInt();
        T = readInt();
        R = readInt();
#else
        input >> N >> K >> T >> R;
#endif

        s0.assign(T, vector(N, vector(K, vector<double>(R))));
        s0_tkrn.assign(T, vector(K, vector(R, vector<double>(N))));
        for (int t = 0; t < T; t++) {
            for (int k = 0; k < K; k++) {
                for (int r = 0; r < R; r++) {
                    for (int n = 0; n < N; n++) {
#ifdef FAST_STREAM
                        s0[t][n][k][r] = readDouble();
#else
                        input >> s0[t][n][k][r];
#endif
                        s0_tkrn[t][k][r][n] = s0[t][n][k][r];
                    }
                }
            }
        }

        d.assign(N, vector(N, vector(K, vector<double>(R))));
        exp_d.assign(N, vector(N, vector(K, vector<double>(R))));
        exp_d_2.assign(N, vector(K, vector(R, vector<double>(N))));
        exp_d_pow.assign(N, vector(N, vector(K, vector<double>(R))));
        for (int k = 0; k < K; k++) {
            for (int r = 0; r < R; r++) {
                for (int m = 0; m < N; m++) {
                    for (int n = 0; n < N; n++) {
#ifdef FAST_STREAM
                        d[n][m][k][r] = readDouble();
#else
                        input >> d[n][m][k][r];
#endif
                        exp_d[n][m][k][r] = exp(d[n][m][k][r]);

                        exp_d_2[n][k][r][m] = exp_d[n][m][k][r];
                        exp_d_pow[n][m][k][r] = pow(exp_d[n][m][k][r], 3);
                    }
                }
            }
        }


#ifdef FAST_STREAM
        J = readInt();
#else
        input >> J;
#endif

        requests.assign(J, {});
        for (int i = 0; i < J; i++) {
            int j;
            int t0, td;

#ifdef FAST_STREAM
            j = readInt();
            requests[j].TBS = readInt();
            requests[j].n = readInt();
            t0 = readInt();
            td = readInt();
#else
            input >> j;
            input >> requests[j].TBS;
            input >> requests[j].n;
            input >> t0 >> td;
#endif
            int t1 = t0 + td - 1;

            requests[j].t0 = t0;
            requests[j].t1 = t1;
        }
    }

    void print() {
        for (int t = 0; t < T; t++) {
            for (int k = 0; k < K; k++) {
                for (int r = 0; r < R; r++) {
                    for (int n = 0; n < N; n++) {
#ifdef FAST_STREAM
                        writeDouble(p[t][n][k][r]);
                        writeChar(' ');
#else
                        cout << p[t][n][k][r] << ' ';
#endif
                    }
#ifdef FAST_STREAM
                    writeChar('\n');
#else
                    cout << '\n';
#endif
                }
            }
        }
    }

    bool verify_power(int t) {
        vector<double> sum(K);
        for (int n = 0; n < N; n++) {
            for (int k = 0; k < K; k++) {
                for (int r = 0; r < R; r++) {
                    sum[k] += p[t][n][k][r];
                }
            }
        }
        for (int k = 0; k < K; k++) {
            if (sum[k] > R + 1e-9) {
                return false;
            }
        }
        for (int k = 0; k < K; k++) {
            for (int r = 0; r < R; r++) {
                double sum = 0;
                for (int n = 0; n < N; n++) {
                    sum += p[t][n][k][r];
                }
                if (sum > 4 + 1e-9) {
                    return false;
                }
            }
        }
        return true;
    }

    double sum_power(int t, int n) {
        double sum = 0;
        for (int k = 0; k < K; k++) {
            for (int r = 0; r < R; r++) {
                sum += p[t][n][k][r];
            }
        }
        return sum;
    }

    void deterministic_descent(int t, vector<int> js) {
        map<int, int> n_to_j;
        for (int j: js) {
            n_to_j[requests[j].n] = j;
        }

        auto correct_build_dp_sum = [&]() {
            // dp_sum[n][k][r]
            vector<vector<vector<double>>> dp_sum(N, vector(K, vector<double>(R)));
            for (int n = 0; n < N; n++) {
                for (int m = 0; m < N; m++) {
                    if (m != n) {
                        for (int k = 0; k < K; k++) {
                            for (int r = 0; r < R; r++) {
                                dp_sum[n][k][r] += s0[t][n][k][r] * p[t][m][k][r] / exp_d[n][m][k][r];
                            }
                        }
                    }
                }
            }
            return dp_sum;
        };

        auto correct_build_dp_sum_2 = [&]() {
            vector<vector<double>> dp_sum_2(N, vector<double>(R));

            auto dp_sum = correct_build_dp_sum();
            for (int n = 0; n < N; n++) {
                for (int k = 0; k < K; k++) {
                    for (int r = 0; r < R; r++) {
                        dp_sum_2[n][r] += dp_sum[n][k][r];
                    }
                }
            }
            return dp_sum_2;
        };

        auto correct_build_dp_exp_d_prod = [&]() {
            vector<vector<vector<double>>> dp_exp_d_prod(N, vector(K, vector<double>(R, 1)));
            for (int n = 0; n < N; n++) {
                for (int k = 0; k < K; k++) {
                    for (int r = 0; r < R; r++) {
                        for (int m = 0; m < N; m++) {
                            if (n != m) {
                                if (p[t][m][k][r] > 0) {
                                    dp_exp_d_prod[n][k][r] *= exp_d_2[n][k][r][m];
                                }
                            }
                        }
                    }
                }
            }
            return dp_exp_d_prod;
        };

        auto correct_build_dp_prod = [&]() {
            vector<vector<vector<double>>> dp_prod(N, vector(K, vector<double>(R, 1)));
            auto dp_sum = correct_build_dp_sum();
            auto dp_sum_2 = correct_build_dp_sum_2();
            auto dp_exp_d_prod = correct_build_dp_exp_d_prod();
            for (int n = 0; n < N; n++) {
                for (int k = 0; k < K; k++) {
                    for (int r = 0; r < R; r++) {
                        dp_prod[n][k][r] *= p[t][n][k][r];
                        dp_prod[n][k][r] *= s0[t][n][k][r];
                        dp_prod[n][k][r] /= 1 + dp_sum_2[n][r] - dp_sum[n][k][r];
                        dp_prod[n][k][r] *= dp_exp_d_prod[n][k][r];
                    }
                }
            }
            return dp_prod;
        };

        auto correct_build_dp_accum_prod = [&]() {
            vector<vector<double>> dp_accum_prod(N, vector<double>(K));
            auto dp_prod = correct_build_dp_prod();
            for (int n = 0; n < N; n++) {
                for (int k = 0; k < K; k++) {
                    dp_accum_prod[n][k] = 1;
                    for (int r = 0; r < R; r++) {
                        if (p[t][n][k][r] > 0) {
                            dp_accum_prod[n][k] *= dp_prod[n][k][r];
                        }
                    }
                }
            }
            return dp_accum_prod;
        };

        auto correct_build_dp_count = [&]() {
            vector<vector<int>> dp_count(N, vector<int>(K));
            for (int n = 0; n < N; n++) {
                for (int k = 0; k < K; k++) {
                    for (int r = 0; r < R; r++) {
                        if (p[t][n][k][r] > 0) {
                            dp_count[n][k]++;
                        }
                    }
                }
            }
            return dp_count;
        };

        auto correct_build_dp_denom_sum = [&]() {
            vector<vector<vector<double>>> dp_denom_sum(N, vector(K, vector<double>(R)));
            auto dp_sum = correct_build_dp_sum();
            auto dp_sum_2 = correct_build_dp_sum_2();
            for (int n = 0; n < N; n++) {
                for (int k = 0; k < K; k++) {
                    for (int r = 0; r < R; r++) {
                        dp_denom_sum[n][k][r] = 1 + dp_sum_2[n][r] - dp_sum[n][k][r];
                    }
                }
            }
            return dp_denom_sum;
        };

        // dp_exp_d_prod[n][k][r]
        vector<vector<vector<double>>> dp_exp_d_prod(N, vector(K, vector<double>(R, 1)));

        // dp_prod[n][k][r]
        vector<vector<vector<double>>> dp_prod(N, vector(K, vector<double>(R)));

        // dp_accum_prod[n][k]
        vector<vector<double>> dp_accum_prod(N, vector<double>(K, 1));

        // dp_count[n][k]
        vector<vector<int>> dp_count(N, vector<int>(K));

        // dp_denom_sum[n][k]
        // (1 + dp_sum_2[m][r] - dp_sum[m][k][r])
        vector<vector<vector<double>>> dp_denom_sum(N, vector(K, vector<double>(R, 1)));

        // dp_denom_sum_global_add[n][r]
        vector<vector<double>> dp_denom_sum_global_add(N, vector<double>(R));

        // dp_power_sum[k][r]
        vector<vector<double>> dp_power_sum(K, vector<double>(R));

        // dp_power_sum2[k]
        vector<double> dp_power_sum2(K);

        auto build_nms = [&]() {
            vector<int> nms;
            for (int j: js) {
                nms.push_back(requests[j].n);
            }
            sort(nms.begin(), nms.end());
            return nms;
        };
        vector<int> nms = build_nms(); // для быстрого прохода по N (без лишних)

        auto initialize_dp_from_power = [&]() {
            dp_count = correct_build_dp_count();
            dp_prod = correct_build_dp_prod();
            dp_accum_prod = correct_build_dp_accum_prod();
            dp_exp_d_prod = correct_build_dp_exp_d_prod();
            dp_denom_sum = correct_build_dp_denom_sum();
            dp_denom_sum_global_add.assign(N, vector<double>(R));

            dp_power_sum.assign(K, vector<double>(R));
            dp_power_sum2.assign(K, 0);
            for (int k = 0; k < K; k++) {
                for (int r = 0; r < R; r++) {
                    for (int n: nms) {
                        dp_power_sum[k][r] += p[t][n][k][r];
                    }
                    dp_power_sum2[k] += dp_power_sum[k][r];
                }
            }
        };

        auto update_dynamics = [&](int n, int k, int r, double change) { // NOLINT
            // TODO: порядок очень важен

            // TODO: ОЧЕНЬ ВАЖНО
            if (change == 0) {
                return;
            }

            //my_time_start();

#ifdef DEBUG_MODE
            if (high_equal(change, 0)) {
                cout << fixed << setprecision(50);
                cout << "lol:\n" << change << endl;
                ASSERT(false, "kek");
            }
#endif

            double old_p = p[t][n][k][r];

            for (int m: nms) {
                for (int k = 0; k < K; k++) {
                    if (p[t][m][k][r] > 0) {
                        dp_accum_prod[m][k] /= dp_prod[m][k][r];
                    }
                }
            }

            ASSERT(!(p[t][n][k][r] != 0 && high_equal(p[t][n][k][r], 0)), ":_(");
            // ====================

            if (high_equal(p[t][n][k][r] + change, 0)) {
                change = -p[t][n][k][r];
                p[t][n][k][r] = 0;
            } else {
                p[t][n][k][r] += change;
            }
            dp_power_sum[k][r] += change;
            dp_power_sum2[k] += change;

            double new_p = p[t][n][k][r];

            ASSERT(!(p[t][n][k][r] != 0 && high_equal(p[t][n][k][r], 0)), ":_(");
            // ====================

            for (int m: nms) {
                if (m != n) {
                    double x = change * s0_tkrn[t][k][r][m] / exp_d_2[n][k][r][m];
                    dp_denom_sum_global_add[m][r] += x;
                    dp_denom_sum[m][k][r] -= x;
                }
            }

            ASSERT((p[t][n][k][r] == change) == high_equal(p[t][n][k][r], change), "kek");

            // TODO: тут очень опасно делать сравнения по eps
            // можно получить неправильное обновление
            if (p[t][n][k][r] > 0 && p[t][n][k][r] == change) {
                // было ноль, стало не ноль
                dp_count[n][k]++;
                for (int m: nms) {
                    if (n != m) {
                        dp_exp_d_prod[m][k][r] *= exp_d_2[n][k][r][m];
                    }
                }
            } else if (p[t][n][k][r] == 0) {
                // было не ноль, стало 0
                dp_count[n][k]--;
                for (int m: nms) {
                    if (n != m) {
                        dp_exp_d_prod[m][k][r] /= exp_d_2[n][k][r][m];
                    }
                }
            }

            for (int m: nms) {
                for (int k = 0; k < K; k++) {
                    dp_prod[m][k][r] = p[t][m][k][r] * s0[t][m][k][r]
                                       / (dp_denom_sum[m][k][r] + dp_denom_sum_global_add[m][r])
                                       * dp_exp_d_prod[m][k][r];
                }
            }

            for (int m: nms) {
                for (int k = 0; k < K; k++) {
                    if (p[t][m][k][r] > 0) {
                        dp_accum_prod[m][k] *= dp_prod[m][k][r];
                    }
                }
            }

            // ====================
            // VERIFY
#ifdef VERIFY_DP
            /*auto correct_dp_sum = correct_build_dp_sum();
            for (int n: nms) {
                for (int k = 0; k < K; k++) {
                    for (int r = 0; r < R; r++) {
                        ASSERT(!is_spoiled(dp_sum[n][k][r]), "fatal");
                        if (!high_equal(dp_sum[n][k][r], correct_dp_sum[n][k][r])) {
                            //cout << fixed << setprecision(50);
                            cout << "\n\nerror:\n" << abs(dp_sum[n][k][r] - correct_dp_sum[n][k][r]) << '\n'
                                 << dp_sum[n][k][r] << '\n'
                                 << correct_dp_sum[n][k][r] << endl;
                            ASSERT(false, "failed");
                        }
                    }
                }
            }

            auto correct_dp_sum_2 = correct_build_dp_sum_2();
            for (int n: nms) {
                for (int r = 0; r < R; r++) {
                    ASSERT(!is_spoiled(dp_sum_2[n][r]), "fatal");
                    if (!high_equal(dp_sum_2[n][r], correct_dp_sum_2[n][r])) {
                        //cout << fixed << setprecision(50);
                        cout << "\n\nerror:\n" << abs(dp_sum_2[n][r] - correct_dp_sum_2[n][r]) << '\n'
                             << dp_sum_2[n][r] << '\n'
                             << correct_dp_sum_2[n][r] << endl;
                        ASSERT(false, "failed");
                    }
                }
            }*/

            auto correct_dp_exp_d_prod = correct_build_dp_exp_d_prod();
            for (int n: nms) {
                for (int k = 0; k < K; k++) {
                    for (int r = 0; r < R; r++) {
                        ASSERT(!is_spoiled(dp_exp_d_prod[n][k][r]), "fatal");
                        if (!high_equal(dp_exp_d_prod[n][k][r], correct_dp_exp_d_prod[n][k][r])) {
                            cout << fixed << setprecision(50);
                            cout << "\n\nerror:\n" << abs(dp_exp_d_prod[n][k][r] - correct_dp_exp_d_prod[n][k][r])
                                 << '\n'
                                 << dp_exp_d_prod[n][k][r] << '\n'
                                 << correct_dp_exp_d_prod[n][k][r] << endl;

                            cout << "change:\n" << change << '\n' << new_p << '\n' << old_p << '\n';

                            ASSERT(false, "failed");
                        }
                    }
                }
            }

            auto correct_dp_count = correct_build_dp_count();
            for (int n: nms) {
                for (int k = 0; k < K; k++) {
                    if (dp_count[n][k] != correct_dp_count[n][k]) {
                        ASSERT(false, "failed");
                    }
                }
            }

            auto correct_dp_denom_sum = correct_build_dp_denom_sum();
            // (dp_denom_sum[m][k][r] + dp_denom_sum_global_add[m][r])
            for (int n: nms) {
                for (int k = 0; k < K; k++) {
                    for (int r = 0; r < R; r++) {
                        ASSERT(!is_spoiled(dp_prod[n][k][r]), "fatal");
                        double my_denom_sum = (dp_denom_sum[n][k][r] + dp_denom_sum_global_add[n][r]);
                        if (!high_equal(my_denom_sum, correct_dp_denom_sum[n][k][r])) {
                            //cout << fixed << setprecision(50);
                            cout << "\n\nerror:\n" << abs(my_denom_sum - correct_dp_denom_sum[n][k][r])
                                 << '\n'
                                 << my_denom_sum << '\n'
                                 << correct_dp_denom_sum[n][k][r] << endl;
                            ASSERT(false, "failed");
                        }
                    }
                }
            }

            auto correct_dp_prod = correct_build_dp_prod();
            for (int n: nms) {
                for (int k = 0; k < K; k++) {
                    for (int r = 0; r < R; r++) {
                        ASSERT(!is_spoiled(dp_prod[n][k][r]), "fatal");
                        if (!high_equal(dp_prod[n][k][r], correct_dp_prod[n][k][r])) {
                            //cout << fixed << setprecision(50);
                            cout << "\n\nerror:\n" << abs(dp_prod[n][k][r] - correct_dp_prod[n][k][r])
                                 << '\n'
                                 << dp_prod[n][k][r] << '\n'
                                 << correct_dp_prod[n][k][r] << endl;
                            ASSERT(false, "failed");
                        }
                    }
                }
            }

            auto correct_dp_accum_prod = correct_build_dp_accum_prod();
            for (int n: nms) {
                for (int k = 0; k < K; k++) {
                    ASSERT(!is_spoiled(dp_accum_prod[n][k]), "fatal");
                    //cout << dp_accum_prod[n][k] << '\n' << correct_dp_accum_prod[n][k] << "\n\n";
                    if (!high_equal(dp_accum_prod[n][k], correct_dp_accum_prod[n][k])) {
                        //cout << fixed << setprecision(50);
                        cout << "\n\nerror:\n" << abs(dp_accum_prod[n][k] - correct_dp_accum_prod[n][k])
                             << '\n'
                             << dp_accum_prod[n][k] << '\n'
                             << correct_dp_accum_prod[n][k] << endl;
                        ASSERT(false, "failed");
                    }

                }
            }
#endif

            //my_time_end();
        };

        auto correct_f = [&]() { // NOLINT
            double result = 0;
            for (int j: js) {
                auto [TBS, n, t0, t1, ost_len] = requests[j];
                add_g[t][n] = get_g(t, n);
                double x = 0;
                if (add_g[t][n] + total_g[j] >= TBS) {
                    x += 1e6;
                } else {
                    x += add_g[t][n] - TBS;
                }
                ASSERT(ost_len != 0, "ost_len is zero, why?");
                result += x;
            }
            return result;
        };

        auto update_add_g = [&](int t, int n) {
            double sum = 0;
            for (int k = 0; k < K; k++) {
                if (dp_count[n][k] != 0) {
                    ASSERT(dp_accum_prod[n][k] > 0, "what?");
                    sum += dp_count[n][k] * log2(1 + pow(dp_accum_prod[n][k], 1.0 / dp_count[n][k]));
                }
            }
            ASSERT(sum >= 0 && !is_spoiled(sum), "invalid g");
            add_g[t][n] = 192 * sum;
        };

        auto fast_f = [&]() { // NOLINT
            my_time_start();
            double result = 0;
            for (int j: js) {
                auto [TBS, n, t0, t1, ost_len] = requests[j];
                update_add_g(t, n);

                double x = 0;

                // TODO: улучшить эту метрику
                if (add_g[t][n] + total_g[j] >= TBS) {
                    x += 1e6;
                } else {
                    x += add_g[t][n] - TBS;
                }

                ASSERT(ost_len != 0, "ost_len is zero, why?");
                result += x;
            }

#ifdef DEBUG_MODE
            if (!high_equal(result, correct_f())) {
                cout << "\n\nerror\n";
                cout << result << endl;
                cout << correct_f() << endl;
                cout << abs(result - correct_f()) << endl;
            }
#endif
            ASSERT(high_equal(result, correct_f()), "fatal");
            my_time_end();
            return result;
        };

        double deltas = 1.0;// / js.size();

        // было
        //TEST CASE==============
        //2/2 4.9136e-05s
        //TEST CASE==============
        //145.993/150 0.899706s
        //TEST CASE==============
        //685.999/829 0.245169s
        //TEST CASE==============
        //184/184 0.0307541s

        double add_power_value = deltas;

        double remove_power_value = 0.3;

        auto calc_add_power = [&](int k, int r) {
            double add = add_power_value;
            {
#ifdef DEBUG_MODE
                double sum = 0;
                for (int n: nms) {
                    sum += p[t][n][k][r];
                }
                ASSERT(high_equal(dp_power_sum[k][r], sum), "kek");
                ASSERT(sum <= 4 + 1e-9, "fatal");
#endif
                add = min(add, 4 + 1e-12 - dp_power_sum[k][r]);
            }

            {
#ifdef DEBUG_MODE
                double sum = 0;
                for (int r = 0; r < R; r++) {
                    sum += dp_power_sum[k][r];
                }
                ASSERT(high_equal(sum, dp_power_sum2[k]), "kek");
                ASSERT(sum <= R + 1e-9, "fatal");
#endif
                add = min(add, R + 1e-12 - dp_power_sum2[k]);
            }
            if (add < 1e-8) {
                add = 0;
            }
            return add;
        };

        double absolute_best_f = -1e300;
        vector<vector<vector<double>>> absolute_best_power(N, vector(K, vector<double>(R)));

        auto update_absolute_best = [&]() {
            double f = fast_f();
            if (f > absolute_best_f) {
                absolute_best_f = f;
                for (int n = 0; n < N; n++) {
                    for (int k = 0; k < K; k++) {
                        for (int r = 0; r < R; r++) {
                            absolute_best_power[n][k][r] = p[t][n][k][r];
                        }
                    }
                }
            }
        };

        update_absolute_best();

        // 689.999/829 12.4852s -> 9.40219s

        int k = 0;
        int r = 0;
        auto do_step_add = [&]() { // NOLINT
            double best_f = -1e300;
            int best_n = -1;
            int best_k = -1;
            int best_r = -1;
            double best_add = 0;

            for (int kek = 0; kek < 3; kek++) {
                for (int n: nms) {
                    //for (int k = 0; k < K; k++) {
                    //for (int r = 0; r < R; r++) {
                    // add
                    {
                        double add = calc_add_power(k, r);
                        if (add != 0) {
                            update_dynamics(n, k, r, +add);
                            double new_f = fast_f();
                            update_dynamics(n, k, r, -add);

                            if (best_f < new_f) {
                                best_f = new_f;
                                best_n = n;
                                best_k = k;
                                best_r = r;
                                best_add = add;
                            }
                        }
                    }

                    // set zero
                    {
                        if (p[t][n][k][r] != 0) {
                            double x = p[t][n][k][r];
                            update_dynamics(n, k, r, -x);
                            double new_f = fast_f();
                            update_dynamics(n, k, r, +x);

                            if (best_f < new_f) {
                                best_f = new_f;
                                best_n = n;
                                best_k = k;
                                best_r = r;
                                best_add = -x;
                            }
                        }
                    }

                    {
                        double sub = max(0.1, p[t][n][k][r] / 2);
                        if (sub < p[t][n][k][r]) {
                            update_dynamics(n, k, r, -sub);
                            double new_f = fast_f();
                            update_dynamics(n, k, r, +sub);

                            if (best_f < new_f) {
                                best_add = -sub;
                                best_n = n;
                                best_f = new_f;
                                best_k = k;
                                best_r = r;
                            }

                        }
                    }
                    //}
                    //}
                }

                k++;
                if (k == K) {
                    k = 0;
                    r++;
                    if (r == R) {
                        r = 0;
                    }
                }

                /*r++;
                if (r == R) {
                    r = 0;
                    k++;
                    if (k == K) {
                        k = 0;
                    }
                }*/
            }

            if (best_n == -1) {
                //k++;
                return false;
            }
            update_dynamics(best_n, best_k, best_r, best_add);
            //cout << "add: " << best_n << ' ' << best_k << ' ' << best_r << ' ' << best_add << ' ' << fast_f() << endl;
            update_absolute_best();
            return true;
        };

        auto do_step_remove = [&]() {
            double best_f = -1e300;
            int best_n = -1;
            int best_k = -1;
            int best_r = -1;
            double best_add = 0;
            for (int n: nms) {
                for (int k = 0; k < K; k++) {
                    for (int r = 0; r < R; r++) {
                        if (p[t][n][k][r] > 0) {
                            double x = p[t][n][k][r];
                            update_dynamics(n, k, r, -x);
                            double new_f = fast_f();
                            update_dynamics(n, k, r, +x);

                            if (best_f < new_f) {
                                best_f = new_f;
                                best_n = n;
                                best_k = k;
                                best_r = r;
                                best_add = -x;
                            }
                        }
                    }
                }
            }

            if (best_n == -1) {
                return false;
            }
            update_dynamics(best_n, best_k, best_r, best_add);
            update_absolute_best();
            return true;
        };

        auto do_step_RobinHood = [&]() {
            double best_f = -1e300;
            int best_n = -1;
            int best_m = -1;
            int best_k = -1;
            int best_r = -1;
            double best_p = 0;
            for (int n: nms) {
                for (int m: nms) {
                    if (m == n) {
                        continue;
                    }
                    for (int k = 0; k < K; k++) {
                        for (int r = 0; r < R; r++) {
                            if (p[t][m][k][r] > deltas) {
                                double x = deltas;
                                update_dynamics(m, k, r, -x);
                                update_dynamics(n, k, r, +x);
                                double new_f = fast_f();
                                update_dynamics(n, k, r, -x);
                                update_dynamics(m, k, r, +x);

                                if (best_f < new_f) {
                                    best_f = new_f;
                                    best_n = n;
                                    best_m = m;
                                    best_k = k;
                                    best_r = r;
                                    best_p = x;
                                }
                            }
                        }
                    }
                }
            }

            if (best_n == -1) {
                return false;
            }
            update_dynamics(best_m, best_k, best_r, -best_p);
            update_absolute_best();
            update_dynamics(best_n, best_k, best_r, +best_p);
            update_absolute_best();
            return true;
        };

        //1 96 8/13
        //2 0 6/14
        //3 3 7/14
        //4 81 9/16

        //add: 49 2 0 1 983842
        //add: 43 0 1 1 1.98426e+06
        //add: 13 1 3 1 2.9846e+06
        //add: 19 0 2 1 3.98485e+06
        //add: 31 0 1 1 4.98486e+06
        //add: 11 0 1 1 5.98487e+06
        //add: 46 0 4 1 5.98678e+06
        //add: 46 1 4 1 6.98696e+06
        //add: 16 1 3 1 6.98767e+06
        //add: 27 2 0 1 6.98822e+06
        //add: 16 1 3 1 6.9884e+06
        //add: 27 2 0 1 6.98857e+06
        //add: 16 1 3 1 6.98868e+06
        //add: 16 2 0 1 6.98889e+06
        //add: 49 2 2 1 7.98906e+06
        //add: 46 3 4 1 7.98906e+06
        //add: 23 3 4 1 7.98908e+06
        //add: 46 3 4 1 7.98908e+06
        //add: 23 3 4 1 7.98911e+06
        //add: 43 3 1 1 6.98909e+06
        //1 96 8/13
        //add: 0 0 4 1 975158
        //add: 45 0 0 1 1.97579e+06
        //add: 42 0 1 1 2.97628e+06
        //add: 20 0 1 1 3.97672e+06
        //add: 4 0 2 1 4.97704e+06
        //add: 9 1 3 1 5.97706e+06
        //add: 8 1 3 1 5.97825e+06
        //add: 19 1 3 1 5.9792e+06
        //add: 10 1 3 1 5.97967e+06
        //add: 0 1 4 1 5.97967e+06
        //add: 0 2 4 1 5.97967e+06
        //add: 19 2 4 1 5.97978e+06
        //add: 41 2 4 1 5.97981e+06
        //add: 0 2 4 1 5.97981e+06
        //add: 4 2 2 1 5.97981e+06
        //add: 4 3 2 1 5.97981e+06
        //add: 34 3 2 1 5.97983e+06
        //add: 34 3 2 1 5.97984e+06
        //add: 34 3 2 1 5.97985e+06
        //add: 45 3 0 1 5.97985e+06
        //2 0 6/14
        //add: 49 0 0 1 984125
        //add: 36 0 0 1 1.98534e+06
        //add: 25 0 1 1 2.98646e+06
        //add: 23 0 2 1 3.98721e+06
        //add: 18 0 3 1 4.98773e+06
        //add: 22 1 4 1 5.98823e+06
        //add: 29 1 4 1 6.98833e+06
        //add: 28 1 4 1 6.98882e+06
        //add: 28 1 4 1 6.98899e+06
        //add: 18 1 3 1 6.98899e+06
        //add: 18 2 3 1 6.98899e+06
        //add: 28 2 3 1 6.98904e+06
        //add: 28 2 3 1 6.98907e+06
        //add: 28 2 3 1 6.98911e+06
        //add: 23 2 2 1 6.98911e+06
        //add: 23 3 2 1 6.98911e+06
        //add: 28 3 2 1 6.98916e+06
        //add: 28 3 2 1 6.9892e+06
        //add: 28 3 2 1 6.98924e+06
        //add: 25 3 1 1 6.98924e+06
        //3 3 7/14
        //add: 38 0 0 1 985447
        //add: 4 0 1 1 1.9859e+06
        //add: 35 0 1 1 2.98632e+06
        //add: 22 0 2 1 3.98672e+06
        //add: 41 0 3 1 4.98695e+06
        //add: 9 1 4 1 5.98718e+06
        //add: 2 1 4 1 6.98735e+06
        //add: 40 1 4 1 7.98746e+06
        //add: 28 1 4 1 7.98801e+06
        //add: 6 2 3 1 7.98804e+06
        //add: 41 1 3 1 8.98807e+06
        //add: 22 2 2 1 8.98807e+06
        //add: 28 2 2 1 8.9883e+06
        //add: 28 2 2 1 8.98842e+06
        //add: 28 2 2 1 8.98851e+06
        //add: 38 3 0 1 8.98851e+06
        //add: 3 3 0 1 8.98889e+06
        //add: 3 3 0 1 8.98904e+06
        //add: 3 3 0 1 8.98914e+06
        //add: 41 3 3 1 8.98914e+06
        //4 81 9/16
        //29.9999/829 0.0441239s
        //TIME SAMPLE: 0.0125003s

        int count = 2;
        for (int step = 0; step < 300; step++) {
            //cout << fast_f() << "->";
            //do_step_RobinHood();
            //do_step_remove();
            do_step_add();
            if (false) {
                break;
                count--;
                if (count <= 0) {
                    break;
                }
                //break; // 621.999/829

                for (int kek = 0; kek < 3; kek++) {
                    do_step_RobinHood();
                }

                //689.999/829
                /*for (int i = 0; i <= js.size() * 0.8; i++) {
                    do_step_remove();
                }*/
            }
            //do_step_add();
            //do_step_remove();
            //do_step_RobinHood();

            //do_step_RobinHood();
        }
        //cout << '\n' << endl;

        // accept best power
        for (int n = 0; n < N; n++) {
            for (int k = 0; k < K; k++) {
                for (int r = 0; r < R; r++) {
                    p[t][n][k][r] = absolute_best_power[n][k][r];
                }
            }
        }

#ifndef FAST_STREAM
        {
            int count = 0;
            for (int j: js) {
                count += total_g[j] + get_g(t, requests[j].n) >= requests[j].TBS;
            }
            static int kek = 0;
            kek++;
            //cout << kek << ' ' << t << ' ' << count << '/' << js.size() << endl;
        }
#endif

        //1 50 2/2
        //2 98 3/3
        //3 72 4/4
        //4 77 4/4
        //5 2 5/5
        //6 5 4/5
        //7 9 4/5
        //8 39 5/5
        //9 67 5/5
        //10 82 5/5
        //11 90 5/5
        //12 4 5/6
        //13 25 4/6
        //14 31 6/6
        //15 33 6/6
        //16 37 5/6
        //17 44 6/6
        //18 55 5/6
        //19 70 4/6
        //20 83 6/6
        //21 86 6/6
        //22 89 5/6
        //23 95 6/6
        //24 6 7/7
        //25 19 7/7
        //26 27 7/7
        //27 35 6/7
        //28 40 7/7
        //29 59 7/7
        //30 60 7/7
        //31 62 7/7
        //32 64 6/7
        //33 80 5/7
        //34 84 4/7
        //35 92 6/7
        //36 97 7/7
        //37 11 7/8
        //38 12 5/8
        //39 14 6/8
        //40 16 6/8
        //41 17 8/8
        //42 18 7/8
        //43 21 8/8
        //44 24 8/8
        //45 28 7/8
        //46 42 6/8
        //47 45 7/8
        //48 47 8/8
        //49 54 8/8
        //50 63 8/8
        //51 69 8/8
        //52 73 8/8
        //53 79 8/8
        //54 94 8/8
        //55 13 6/9
        //56 15 8/9
        //57 22 6/9
        //58 23 8/9
        //59 30 5/9
        //60 32 6/9
        //61 51 9/9
        //62 53 9/9
        //63 56 9/9
        //64 57 8/9
        //65 65 9/9
        //66 75 5/9
        //67 76 8/9
        //68 78 7/9
        //69 8 8/10
        //70 20 7/10
        //71 26 10/10
        //72 34 9/10
        //73 36 8/10
        //74 43 8/10
        //75 46 10/10
        //76 48 7/10
        //77 58 9/10
        //78 66 9/10
        //79 87 9/10
        //80 93 8/10
        //81 1 5/11
        //82 7 8/11
        //83 38 8/11
        //84 49 6/11
        //85 52 9/11
        //86 61 10/11
        //87 68 9/11
        //88 85 9/11
        //89 91 9/11
        //90 10 9/12
        //91 29 8/12
        //92 41 10/12
        //93 71 11/12
        //94 74 10/12
        //95 88 9/12
        //96 96 8/13
        //97 0 7/14
        //98 3 6/14
        //99 81 10/16
        //689.999/829 14.1578s
    }

    void set_nice_power(int t, vector<int> js) {
        if (js.empty()) {
            return;
        }

        // наиболее оптимально расставить силу так
        // что это значит? наверное мы хотим как можно больший прирост g
        // а также чтобы доотправлять сообщения

        for (int n = 0; n < N; n++) {
            for (int k = 0; k < K; k++) {
                for (int r = 0; r < R; r++) {
                    p[t][n][k][r] = 0;
                }
            }
        }

        deterministic_descent(t, js);

        for (int j: js) {
            int n = requests[j].n;
            add_g[t][n] = get_g(t, n);
            total_g[j] += add_g[t][n];
        }
    }

    void solve() {
        p.assign(T, vector(N, vector(K, vector<double>(R))));
        total_g.assign(J, 0);
        add_g.assign(T, vector<double>(N));
        for (int j = 0; j < J; j++) {
            requests[j].ost_len = requests[j].t1 - requests[j].t0 + 1;
        }

        vector<vector<int>> js(T);
        for (int j = 0; j < J; j++) {
            auto [TBS, n, t0, t1, ost_len] = requests[j];
            for (int t = t0; t <= t1; t++) {
                js[t].push_back(j);
            }
        }

        auto verify_ost_len = [&](int j) {
#ifdef DEBUG_MODE
            auto [TBS, n, t0, t1, ost_len] = requests[j];
            int count = 0;
            for (int t = t0; t <= t1; t++) {
                count += find(js[t].begin(), js[t].end(), j) != js[t].end();
            }
            //cout << count << ' ' << ost_len << endl;
            ASSERT(count == ost_len, "failed calculating ost_len");
#endif
        };

        for (int step = 0; step < T; step++) {
            // выберем самое лучшее время, куда наиболее оптимально поставим силу

            //96 96 8/13
            //97 0 7/14
            //98 3 6/14
            //99 81 10/16
            //for (int t: {96, 0, 3, 81}) {
            //    set_nice_power(t, js[t]);
            //}
            //return;

            int best_time = -1;
            /// оптимальный выбор времени
            /// немного улучшает score
            {
                // TODO: выбирать при помощи суммы TBS и уже набранной total_g
                // а не при помощи размера, это треш

                auto metric = [&](int t) {
                    double result = 0;
                    for (int j: js[t]) {
                        result += 1.0 / pow(requests[j].ost_len, 2);
                    }
                    result += js[t].size() * 0.3;
                    return result;
                };

                auto compare = [&](int lhs, int rhs) {
                    return metric(lhs) < metric(rhs);
                };

                for (int t = 0; t < T; t++) {
                    if (!js[t].empty()) {
                        if (best_time == -1 || compare(t, best_time)) {
                            best_time = t;
                        }
                    }
                }

                if (best_time == -1) {
                    break;
                }
            }

            // наиболее оптимально расставим силу в момент времени best_time
            set_nice_power(best_time, js[best_time]);

            // remove accepted
            for (int j: js[best_time]) {
                auto [TBS, n, t0, t1, ost_len] = requests[j];

                if (total_g[j] >= TBS) {
                    // удалим из других
                    for (int t = t0; t <= t1; t++) {
                        if (!js[t].empty() && t != best_time) {
                            requests[j].ost_len--;
                            auto it = find(js[t].begin(), js[t].end(), j);
                            ASSERT(it != js[t].end(), "fatal");
                            js[t].erase(it);
                        }
                    }
                }
            }

            // clear
            {
                for (int j: js[best_time]) {
                    verify_ost_len(j);
                    requests[j].ost_len--;
                }

                js[best_time].clear();
            }
        }
    }

    double get_score() {
        double power_sum = 0;
        for (int t = 0; t < T; t++) {
            for (int n = 0; n < N; n++) {
                for (int k = 0; k < K; k++) {
                    for (int r = 0; r < R; r++) {
                        power_sum += p[t][n][k][r];
                    }
                }
            }
        }
        int X = 0;
        for (int j = 0; j < J; j++) {
            X += total_g[j] >= requests[j].TBS;
        }
        return X - 1e-6 * power_sum;
    }

    double correct_get_g(int t, int n) {
        vector<vector<double>> dp_s0_p_d(K, vector<double>(R));

        // update dp_s0_p_d
        {
            // dp_sum[k][r]
            vector<vector<double>> dp_sum(K, vector<double>(R));
            for (int m = 0; m < N; m++) {
                if (m != n) {
                    for (int k = 0; k < K; k++) {
                        for (int r = 0; r < R; r++) {
                            dp_sum[k][r] += s0[t][n][k][r] * p[t][m][k][r] / exp_d[n][m][k][r];
                        }
                    }
                }
            }

            vector<double> dp_sum_2(R);
            for (int k = 0; k < K; k++) {
                for (int r = 0; r < R; r++) {
                    dp_sum_2[r] += dp_sum[k][r];
                }
            }

            for (int k = 0; k < K; k++) {
                for (int r = 0; r < R; r++) {
                    dp_s0_p_d[k][r] = 1 + dp_sum_2[r] - dp_sum[k][r];
                }
            }
        }

        double sum = 0;
        for (int k = 0; k < K; k++) {
            double accum_prod = 1;
            int count = 0;
            for (int r = 0; r < R; r++) {
                if (p[t][n][k][r] > 0) {
                    count++;
                    accum_prod *= p[t][n][k][r];
                    accum_prod *= s0[t][n][k][r];
                    accum_prod /= dp_s0_p_d[k][r];

                    for (int m = 0; m < N; m++) {
                        if (n != m) {
                            if (p[t][m][k][r] > 0) {
                                accum_prod *= exp_d_2[n][k][r][m];
                            }
                        }
                    }
                }
            }

            if (count != 0) {
                sum += count * log2(1 + pow(accum_prod, 1.0 / count));
            }
        }
        ASSERT(sum >= 0 && !is_spoiled(sum), "invalid g");
        return 192 * sum;
    }

    double get_g(int t, int n) {
        return correct_get_g(t, n);

        double sum = 0;
        for (int k = 0; k < K; k++) {
            double accum_prod = 1;
            int count = 0;
            for (int r = 0; r < R; r++) {
                if (p[t][n][k][r] > 0) {
                    count++;
                    accum_prod *= p[t][n][k][r];
                    accum_prod *= s0[t][n][k][r];
                }
            }
            if (count != 0) {
                sum += count * log2(1 + pow(accum_prod, 1.0 / count));
            }
        }
        ASSERT(sum >= 0 && !is_spoiled(sum), "invalid g");
        ASSERT(correct_get_g(t, n) == 192 * sum, "failed calc");
        return 192 * sum;
    }
};

int main() {
#ifndef FAST_STREAM
    for (int test_case = 0; test_case <= 3; test_case++) {

        std::ifstream input("input.txt");
        if (test_case == 0) {
            input = std::ifstream("input.txt");
        } else if (test_case == 1) {
            input = std::ifstream("6");
        } else if (test_case == 2) {
            input = std::ifstream("21");
        } else if (test_case == 3) {
            input = std::ifstream("51");
        } else {
            ASSERT(false, "what is test case?");
        }
        cout << "TEST CASE==============\n";
#endif
        //std::ios::sync_with_stdio(false);
        //std::cin.tie(0);
        //std::cout.tie(0);

        Solution solution;
#ifdef FAST_STREAM
        solution.read();
#else
        solution.read(input);
        auto time_start = steady_clock::now();
#endif

        solution.solve();

#ifndef FAST_STREAM
        auto time_stop = steady_clock::now();
        auto duration = time_stop - time_start;
        double time = duration_cast<nanoseconds>(duration).count() / 1e9;
        cout << solution.get_score() << '/' << solution.J << ' ' << time << "s" << endl;
        cout << "TIME SAMPLE: " << TOTAL_TIME << "s\n";
#endif

#ifdef FAST_STREAM
        solution.print();
#endif

#ifndef FAST_STREAM
    }
#endif
}
