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
    return abs(x - y) <= 1e-9 * max(abs(x), abs(y));
}

#define FAST_STREAM

//#define PRINT_DEBUG_INFO

//#define PRINT_SEARCH_INFO

//#define VERIFY_DP

//#define DEBUG_MODE

#ifdef DEBUG_MODE

#define FAILED_ASSERT(message)                                                \
    {                                                                         \
        std::cerr << "assert failed at " __FILE__ << ":" << __LINE__ << '\n'; \
        std::cerr << "message: \"" << (message) << "\"\n";                    \
        std::exit(1);                                                         \
    }

#define ASSERT(condition, message) \
    if (!(condition))              \
    FAILED_ASSERT(message)

//_STL_VERIFY(condition, message) // don't work on CLion

#else

#define ASSERT(condition, message)// condition

#endif// DEBUG_MODE

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

    enum build_weights_version {
        ALL,
        ONCE_IN_R,
    } use_build_weight_version = ONCE_IN_R;

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

    vector<vector<double>> power_snapshot(int t, int n) {
        vector<vector<double>> save_p(K, vector<double>(R));
        for (int k = 0; k < K; k++) {
            for (int r = 0; r < R; r++) {
                save_p[k][r] = p[t][n][k][r];
            }
        }
        return save_p;
    }

    vector<vector<vector<double>>> power_snapshot(int t) {
        vector<vector<vector<double>>> save_p(N, vector(K, vector<double>(R)));
        for (int n = 0; n < N; n++) {
            for (int k = 0; k < K; k++) {
                for (int r = 0; r < R; r++) {
                    save_p[n][k][r] = p[t][n][k][r];
                }
            }
        }
        return save_p;
    }

    void set_power(int t, int n, const vector<vector<double>> &power) {
        for (int k = 0; k < K; k++) {
            for (int r = 0; r < R; r++) {
                p[t][n][k][r] = power[k][r];
            }
        }
    }

    void set_power(int t, const vector<vector<vector<double>>> &power) {
        for (int n = 0; n < N; n++) {
            for (int k = 0; k < K; k++) {
                for (int r = 0; r < R; r++) {
                    p[t][n][k][r] = power[n][k][r];
                }
            }
        }
    }

    void mult_power(int t, int n, double factor) {
        for (int k = 0; k < K; k++) {
            for (int r = 0; r < R; r++) {
                p[t][n][k][r] *= factor;
            }
        }
    }

    double calc_g_with_power_factor(int t, int n, double factor) {
        auto old_power = power_snapshot(t, n);
        mult_power(t, n, factor);
        double g = get_g(t, n);
        set_power(t, n, old_power);
        return g;
    }

    bool verify_mult_power(int t, int n, double factor) {
        vector<double> sum(K);
        for (int m = 0; m < N; m++) {
            for (int k = 0; k < K; k++) {
                for (int r = 0; r < R; r++) {
                    if (m != n) {
                        sum[k] += p[t][m][k][r];
                    } else {
                        sum[k] += factor * p[t][m][k][r];
                    }
                }
            }
        }
        for (int k = 0; k < K; k++) {
            if (sum[k] > R) {
                return false;
            }
        }
        for (int k = 0; k < K; k++) {
            for (int r = 0; r < R; r++) {
                double sum = 0;
                for (int m = 0; m < N; m++) {
                    if (m != n) {
                        sum += p[t][m][k][r];
                    } else {
                        sum += factor * p[t][m][k][r];
                    }
                }
                if (sum > 4) {
                    return false;
                }
            }
        }
        return true;
    }

    // если этот запрос отправлен, то эта функция минимизирует количество затраченной силы для этого,
    // чтобы дать возможность другим запросам пользоваться большей силой
    /// !вызывать до обновления total_g[j]!
    void minimize_power(int t, int j) {
        auto [TBS, n, t0, t1, ost_len] = requests[j];
        double add_g = get_g(t, n);
        if (add_g + total_g[j] >= TBS + 1e-1) {
#ifdef PRINT_DEBUG_INFO
            cout << "minimize_power(" << TBS << ") " << add_g + total_g[j] << "->";
#endif

            double tl = 0, tr = 1;
            while (tl < tr - 1e-3) {
                double tm = (tl + tr) / 2;
                if (calc_g_with_power_factor(t, n, tm) + total_g[j] >= TBS) {
                    tr = tm;
                } else {
                    tl = tm;
                }
            }
            double good_factor = tr;
            mult_power(t, n, tr);
#ifdef PRINT_DEBUG_INFO
            cout << total_g[j] + get_g(t, n) << endl;
#endif
        }
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

    // если сообщение не отправлено
    // то вернет множитель силы, на который нужно домножить
    // чтобы отправить это сообщение
    // либо 0
    double calc_power_factor_for_accept_request(int t, int j) {
        auto [TBS, n, t0, t1, ost_len] = requests[j];

        // этот запрос не используется
        if (sum_power(t, n) == 0) {
            return 0;
        }

#ifdef PRINT_DEBUG_INFO
        cout << "more_power(" << TBS << ") " << get_g(t, n) + total_g[j] << "->";
#endif

        double tl = 0, tr = 1e9;
        while (tl < tr - 1e-3) {
            double tm = (tl + tr) / 2;
            // TODO: это сделает неточным good_factor=tr
            /*if (!verify_mult_power(t, n, tm)) {
                // мы домножили на слишком большое число
                tr = tm;
            }*/
            if (calc_g_with_power_factor(t, n, tm) + total_g[j] >= TBS) {
                tr = tm;
            } else {
                tl = tm;
            }
        }

        double good_factor = tr;
        ASSERT(good_factor > 1, "why not?");
#ifdef PRINT_DEBUG_INFO
        cout << calc_g_with_power_factor(t, n, good_factor) + total_g[j] << ", good_factor: " << good_factor
             << ", verify: " << verify_mult_power(t, n, good_factor) << endl;
#endif
        if (verify_mult_power(t, n, good_factor)) {
            return good_factor;
        } else {
            return 0;
        }
    }

    void set_power(int t, vector<tuple<double, int, int, int>> set_of_weights) {
        if (use_build_weight_version == ONCE_IN_R) {
            for (auto [weight, n, k, r]: set_of_weights) {
                p[t][n][k][r] = min(4.0, weight * R);
            }
        } else {
            for (auto [weight, n, k, r]: set_of_weights) {
                p[t][n][k][r] = 4 * weight;
            }
            vector<double> sum(K);
            for (auto [weight, n, k, r]: set_of_weights) {
                sum[k] += p[t][n][k][r];
            }
            for (auto [weight, n, k, r]: set_of_weights) {
                if (sum[k] > 1e-9) {
                    p[t][n][k][r] *= min(1.0, R / sum[k]);
                }
            }
        }
    }

    vector<tuple<double, int, int, int>> build_weights_of_power(int t, const vector<int> &js) {
        // (weight, n, k, r)
        vector<tuple<double, int, int, int>> set_of_weights;
        if (use_build_weight_version == ONCE_IN_R) {
            // requests_weights[r] = {}
            vector<vector<tuple<double, int>>> requests_weights(R);
            for (int j: js) {
                auto [TBS, n, t0, t1, ost_len] = requests[j];
                for (int r = 0; r < R; r++) {
                    double weight = 1;
                    ASSERT(total_g[j] < TBS, "failed");

                    weight *= exp(-cbrt(TBS - total_g[j]));
                    for (int k = 0; k < K; k++) {
                        weight *= s0[t][n][k][r];
                    }

                    ASSERT(weight >= 0 && !is_spoiled(weight), "invalid weight");

                    requests_weights[r].emplace_back(weight, j);
                }
            }

            // (weight, r, j)
            set<tuple<double, int, int>, greater<>> S;
            for (int r = 0; r < R; r++) {
                for (auto [weight, j]: requests_weights[r]) {
                    S.insert({weight, r, j});
                }
            }
            vector<bool> used(R, false);
            while (!S.empty()) {
                //cout << S.size() << endl;
                auto [weight, r, j] = *S.begin();
                S.erase(S.begin());
                if (used[r]) {
                    continue;
                }
                used[r] = true;

                // relax others
                {
                    set<tuple<double, int, int>, greater<>> new_S;
                    for (auto [weight2, r2, j2]: S) {
                        if (j == j2) {
                            new_S.insert({weight2 / 1e40, r2, j2});
                        } else {
                            new_S.insert({weight2, r2, j2});
                        }
                    }
                    S = new_S;
                }

                auto [TBS, n, t0, t1, ost_len] = requests[j];

                for (int k = 0; k < K; k++) {
                    double weight = 1;

                    ASSERT(total_g[j] < TBS, "failed");

                    //weight = 30 - log(1 + TBS);//+ log(1 + total_g[j]);

                    //cout << weight << endl;
                    ASSERT(weight >= 0 && !is_spoiled(weight), "invalid weight");

                    set_of_weights.emplace_back(weight, n, k, r);
                }
            }
        } else if (use_build_weight_version == ALL) {
            for (int j: js) {
                auto [TBS, n, t0, t1, ost_len] = requests[j];

                for (int k = 0; k < K; k++) {
                    for (int r = 0; r < R; r++) {
                        double weight = 1;
                        for (int j2: js) {
                            int m = requests[j2].n;
                            if (n != m) {
                                weight *= pow(exp_d[n][m][k][r], 3);
                            }
                        }

                        ASSERT(weight >= 0 && !is_spoiled(weight), "invalid weight");

                        set_of_weights.emplace_back(weight, n, k, r);
                    }
                }
            }
        } else {
            ASSERT(false, "invalid build weights version");
        }

        auto calc_sum = [&]() {
            vector<double> sum_weight(K);
            for (auto [weight, n, k, r]: set_of_weights) {
                ASSERT(weight >= 0, "invalid weight");
                sum_weight[k] += weight;
            }
            return sum_weight;
        };

        auto calc_sum2 = [&]() {
            vector<vector<double>> sum_weight(K, vector<double>(R));
            for (auto [weight, n, k, r]: set_of_weights) {
                ASSERT(weight >= 0, "invalid weight");
                sum_weight[k][r] += weight;
            }
            return sum_weight;
        };

        auto fix_sum = [&]() {
            if (use_build_weight_version == ONCE_IN_R) {
                auto sum_weight = calc_sum();

                for (auto &[weight, n, k, r]: set_of_weights) {
                    if (sum_weight[k] != 0) {
                        weight = weight / sum_weight[k];
                    } else {
                        weight = 0;
                    }
                }
            } else {
                auto sum_weight = calc_sum2();

                for (auto &[weight, n, k, r]: set_of_weights) {
                    if (sum_weight[k][r] != 0) {
                        weight = weight / sum_weight[k][r];
                    } else {
                        weight = 0;
                    }
                }
            }
        };

        auto threshold = [&]() {
            for (int i = 0; i < set_of_weights.size(); i++) {
                auto [weight, n, k, r] = set_of_weights[i];
                if (weight < 0.02) {
                    swap(set_of_weights[i], set_of_weights.back());
                    set_of_weights.pop_back();
                    i--;
                }
            }
        };

        auto verify = [&]() {
            for (auto [weight, n, k, r]: set_of_weights) {
                ASSERT(0 <= weight && weight <= 1, "invalid weight");
            }
            auto sum_weight = calc_sum();
            for (int k = 0; k < K; k++) {
                ASSERT(sum_weight[k] == 0 || abs(sum_weight[k] - 1) < 1e-9, "invalid sum_weight");

            }
        };

        fix_sum();
        threshold();
        fix_sum();

        return set_of_weights;
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

        auto build_nms = [&]() {
            vector<int> nms;
            for (int j: js) {
                nms.push_back(requests[j].n);
            }
            sort(nms.begin(), nms.end());
            return nms;
        };
        vector<int> nms = build_nms(); // для быстрого прохода по N (без лишних)

        dp_count = correct_build_dp_count();
        dp_prod = correct_build_dp_prod();
        dp_accum_prod = correct_build_dp_accum_prod();
        dp_exp_d_prod = correct_build_dp_exp_d_prod();
        dp_denom_sum = correct_build_dp_denom_sum();
        dp_denom_sum_global_add.assign(N, vector<double>(R));

        auto update_dynamics = [&](int n, int k, int r, double change) { // NOLINT
            // TODO: порядок очень важен

            // TODO: ОЧЕНЬ ВАЖНО
            if (change == 0) {
                return;
            }

            for (int m: nms) {
                for (int k = 0; k < K; k++) {
                    if (p[t][m][k][r] > 0) {
                        ASSERT(!high_equal(0, dp_prod[m][k][r]), "dividing by zero");
                        dp_accum_prod[m][k] /= dp_prod[m][k][r];
                    }
                }
            }

            // ====================
            p[t][n][k][r] += change;
            // ====================

            for (int m: nms) {
                if (m != n) {
                    double x = change * s0_tkrn[t][k][r][m] / exp_d_2[n][k][r][m];
                    dp_denom_sum_global_add[m][r] += x;
                    dp_denom_sum[m][k][r] -= x;
                }
            }

            // TODO: тут очень опасно делать сравнения по eps
            // можно получить неправильное обновление
            if (p[t][n][k][r] > 0 && (p[t][n][k][r] - change) == 0) {
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
                            //cout << fixed << setprecision(50);
                            cout << "\n\nerror:\n" << abs(dp_exp_d_prod[n][k][r] - correct_dp_exp_d_prod[n][k][r])
                                 << '\n'
                                 << dp_exp_d_prod[n][k][r] << '\n'
                                 << correct_dp_exp_d_prod[n][k][r] << endl;
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
                    x += add_g[t][n] - TBS * 1.1;
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
                    sum += dp_count[n][k] * log2(1 + pow(dp_accum_prod[n][k], 1.0 / dp_count[n][k]));
                }
            }
            ASSERT(sum >= 0 && !is_spoiled(sum), "invalid g");
            add_g[t][n] = 192 * sum;
        };

        auto fast_f = [&]() { // NOLINT
            double result = 0;
            for (int j: js) {
                auto [TBS, n, t0, t1, ost_len] = requests[j];
                update_add_g(t, n);

                double x = 0;

                //TEST CASE==============
                //2/2 0.000111581s
                //TEST CASE==============
                //145.993/150 0.266032s
                //TEST CASE==============
                //683.999/829 0.326419s
                //TEST CASE==============
                //184/184 0.0688514s

                // TODO: улучшить эту метрику
                if (add_g[t][n] + total_g[j] >= TBS) {
                    x += 1e6;
                } else {
                    x += add_g[t][n] - TBS * 1.1;
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
            return result;
        };

        double add_power_value = 1;
        double take_power_value = 0.3;

        auto calc_add_power = [&](int k, int r) {
            double add = add_power_value;
            {
                double sum = 0;
                for (int n: nms) {
                    sum += p[t][n][k][r];
                }
                add = min(add, 4 - sum);
                ASSERT(sum <= 4 + 1e-9, "fatal");
            }

            {
                double sum = 0;
                for (int n: nms) {
                    for (int r = 0; r < R; r++) {
                        sum += p[t][n][k][r];
                    }
                }
                add = min(add, R - sum);
                ASSERT(sum <= R + 1e-9, "fatal");
            }
            if (add < 1e-9) {
                add = 0;
            }
            return add;
        };

        auto RobinHood = [&](int n, int k, int r) { // NOLINT
            // посмотрим у кого мы можем отобрать силу, чтобы взять ее себе

            double best_f = -1e300;
            int best_m = -1;

            vector<tuple<double, int>> kek;
            for (int m: nms) {
                if (n != m) {
                    if (p[t][m][k][r] > take_power_value) {
                        int j = n_to_j[m];
                        // kek.emplace_back((total_g[j] + add_g[t][m]) - requests[j].TBS, m); // 16067.854 points
                        // TODO: рассматривать add_g[t][n] / sum_power
                        // типа то, сколько мы получаем g за единицу вложенной силы

                        //kek.emplace_back(0.5 * (total_g[j] + add_g[t][m]) / requests[j].TBS +
                        //                 add_g[t][m] / sum_power, m);

                        //kek.emplace_back(2 * (total_g[j] + add_g[t][m]) / requests[j].TBS +
                        //                 add_g[t][m] / sum_power, m); // 16027.854 points

                        //kek.emplace_back((total_g[j] + add_g[t][m]) / requests[j].TBS +
                        //add_g[t][m] / sum_power, m); // 16025.854 points

                        //kek.emplace_back((total_g[j] + add_g[t][m]) / requests[j].TBS, m);// 16064.854 points
                        //kek.emplace_back(add_g[t][m] - requests[j].TBS, m); // 16067.854 points
                        //kek.emplace_back(add_g[t][m], m); // 16006.854 points
                        // kek.emplace_back(-add_g[t][m] / sum_power, m); // 15957.855 points
                        // kek.emplace_back(add_g[t][m] / sum_power, m); // 16024.854 points

                        kek.emplace_back(add_g[t][m] - requests[j].TBS, m);
                    }
                }
            }
            sort(kek.begin(), kek.end(), greater<>());

            for (auto [weight, m]: kek) {
                ASSERT(verify_power(t), "failed power");
                update_dynamics(m, k, r, -take_power_value);
                ASSERT(verify_power(t), "failed power");
                update_dynamics(n, k, r, +take_power_value);
                ASSERT(verify_power(t), "failed power");

                double new_f = fast_f();

                if (new_f > best_f) {
                    best_f = new_f;
                    best_m = m;
                }

                update_dynamics(n, k, r, -take_power_value);
                ASSERT(verify_power(t), "failed power");
                update_dynamics(m, k, r, +take_power_value);
                ASSERT(verify_power(t), "failed power");
                // fast_f();
                break;
            }
            fast_f();

            return tuple{best_f, take_power_value, best_m};
        };

        double DIVINE_f = fast_f();
#ifdef VERIFY_DP
        ASSERT(high_equal(fast_f(), correct_f()), "fatal");
#endif
        const int STEPS = 1e9;

        auto find_best_step = [&](int j) { // NOLINT
            if (j == -1) {
                ASSERT(false, "out of range");
            }

            auto [TBS, n, t0, t1, ost_len] = requests[j];

            double best_f = -1e300;
            int best_m = -1;
            int best_k = -1;
            int best_r = -1;
            double best_add = 0;
            for (int k = 0; k < K; k++) {
                for (int r = 0; r < R; r++) {
                    // try to add
                    {
                        double add = calc_add_power(k, r);

                        if (add != 0) {
                            update_dynamics(n, k, r, +add);
                            double new_f = fast_f();
                            update_dynamics(n, k, r, -add);

                            if (best_f < new_f) {
                                best_add = add;
                                best_m = -1;
                                best_f = new_f;
                                best_k = k;
                                best_r = r;
                            }
                        }
                    }

                    // try to sub
                    {
                        double sub = max(0.1, p[t][n][k][r] / 2);
                        if (sub < p[t][n][k][r]) {
                            update_dynamics(n, k, r, -sub);
                            double new_f = fast_f();
                            update_dynamics(n, k, r, +sub);

                            if (best_f < new_f) {
                                best_add = -sub;
                                best_m = -1;
                                best_f = new_f;
                                best_k = k;
                                best_r = r;
                            }

                        }
                    }

                    // try to take away
                    {
                        auto [new_f, take_power, m] = RobinHood(n, k, r);
                        if (m != -1) {
                            if (best_f < new_f) {
                                best_add = take_power;
                                best_m = m;
                                best_f = new_f;
                                best_k = k;
                                best_r = r;
                            }
                        }
                    }

                    // try to set zero
                    {
                        if (p[t][n][k][r] > 0) {
                            double x = p[t][n][k][r];
                            update_dynamics(n, k, r, -x);
                            double new_f = fast_f();
                            update_dynamics(n, k, r, +x);

                            if (best_f < new_f) {
                                best_add = -x;
                                best_m = -1;
                                best_f = new_f;
                                best_k = k;
                                best_r = r;
                            }
                        }
                    }
                }
            }

            return tuple{best_f, j, best_m, best_k, best_r, best_add};
        };

        int loller = 0;
        for (int step = 0; step < STEPS && !js.empty(); step++) {
            //cout << fast_f() << "->";

            //TEST CASE==============
            //2/2 4.3538e-05s
            //TEST CASE==============
            //145.993/150 0.263592s
            //TEST CASE==============
            //680.999/829 0.314781s
            //TEST CASE==============
            //184/184 0.0654606s

            auto foo = [&](int j) {
                auto [TBS, n, t0, t1, ost_len] = requests[j];
                //return TBS - (total_g[j] + add_g[t][n]); // 16200.849 points
                //return add_g[t][n]; // 14852.862 points
                //return add_g[t][n] + total_g[j]; // 14816.857 points
                //return add_g[t][n] - total_g[j]; // 15001.858 points
                //return add_g[t][n] - total_g[j] - TBS; // 15785.851 points
                // return add_g[t][n] - total_g[j] + TBS; // 15785.851 points
                // +total_g[j] очень плохо
                // return -add_g[t][n] - total_g[j] - TBS; // 13007.855 points
                // return -add_g[t][n] - total_g[j] + TBS; // 16200.849 points
                //return -add_g[t][n] + TBS; // 16171.849 points

                // return -add_g[t][n]; // 14922.852 points

                // return sum_power(t, n); // 14922.865 points
                // return -sum_power(t, n); // 14917.852 points
                //return -add_g[t][n] + sum_power(t, n); // 14923.852 points
                // return -add_g[t][n] - sum_power(t, n); // 14593.855 points
                //return -add_g[t][n] + 30 * sum_power(t, n); // 14942.852 points
                //return -add_g[t][n] + 60 * sum_power(t, n); // 14948.852 points
                // return -add_g[t][n] + 1000 * sum_power(t, n); // 15203.85 points
                // return -add_g[t][n] + 10000 * sum_power(t, n); // 15079.861 points
                // return -add_g[t][n] + 5000 * sum_power(t, n); // Partial result: 14958.866 points
                // return -add_g[t][n] + 1100 * sum_power(t, n); // 15211.849 points
                // return -add_g[t][n] + 900 * sum_power(t, n); // 15189.85 points
                //return -add_g[t][n] + 1200 * sum_power(t, n); // 14989.852 points
                //return -add_g[t][n] + 1050 * sum_power(t, n); // 15217.85 points

                // return -add_g[t][n] + 1050 * sum_power(t, n) + TBS; // 16191.85 points
                // return -add_g[t][n] - total_g[j] + 1050 * sum_power(t, n) + TBS; // 16228.85 points
                // return -add_g[t][n] - total_g[j] + 950 * sum_power(t, n) + TBS; // 16232.85 points
                // return -add_g[t][n] - total_g[j] + 900 * sum_power(t, n) + TBS; // 16228.85 points

                // return ost_len * 900 -add_g[t][n] - total_g[j] + 950 * sum_power(t, n) + TBS; // 16238.85 points

                return ost_len * 900 - add_g[t][n] - total_g[j] + 950 * sum_power(t, n) + TBS; // 16238.85 points
            };

            sort(js.begin(), js.end(), [&](int lhs, int rhs) {
                return foo(lhs) < foo(rhs);
            });

            double best_f = -1e200;
            int best_j = -1;
            int best_m = -1;
            int best_k = -1;
            int best_r = -1;
            double best_add = 0;

            auto update_best = [&](tuple<double, int, int, int, int, double> step) {
                auto [f, j, m, k, r, add] = step;
                if (f > best_f + 1e-9) {
                    best_f = f;
                    best_j = j;
                    best_m = m;
                    best_k = k;
                    best_r = r;
                    best_add = add;
                }
            };

            int count_visited = 0;
            for (int j: js) {
                auto [TBS, n, t0, t1, ost_len] = requests[j];
                if ((total_g[j] + add_g[t][n]) < TBS) {
                    update_best(find_best_step(j));
                    count_visited++;
                    if (count_visited == 2) {
                        break;
                    }
                }
            }

            //TEST CASE==============
            //2/2 0.00018763s
            //TEST CASE==============
            //143.994/150 0.120328s
            //TEST CASE==============
            //685.999/829 0.133603s
            //TEST CASE==============
            //184/184 0.050675s

            /*reverse(js.begin(), js.end());

            count_visited = 0;
            for (int j: js) {
                auto [TBS, n, t0, t1, ost_len] = requests[j];
                if ((total_g[j] + add_g[t][n]) > TBS) {
                    update_best(find_best_step(j));
                    count_visited++;
                    if (count_visited == 2) {
                        break;
                    }
                }
            }*/

            if (best_j == -1) {
                break;
            }

            if (best_f <= DIVINE_f) {
                loller++;
                if (loller == 5) {
                    break;
                }
                // сделаем какой-то шаг
            }
            DIVINE_f = best_f;

            auto [TBS, n, t0, t1, ost_len] = requests[best_j];

            if (best_m != -1) {
                // Robin Hood
                update_dynamics(best_m, best_k, best_r, -best_add);
                update_dynamics(n, best_k, best_r, +best_add);
                ASSERT(verify_power(t), "failed power");
            } else if (best_k != -1) {
                // add power
                update_dynamics(n, best_k, best_r, best_add);
                ASSERT(verify_power(t), "failed power");
            }
            fast_f();

            //cout << abs(fast_f() - DIVINE_f) << endl;
            ASSERT(high_equal(fast_f(), DIVINE_f), "failed");
        }
    }

    void set_nice_power(int t, vector<int> js) {
        if (js.empty()) {
            return;
        }

        // наиболее оптимально расставить силу так
        // что это значит? наверное мы хотим как можно больший прирост g
        // а также чтобы доотправлять сообщения

        auto get_score_in_time = [&]() {
            double score = 0;
            for (int j: js) {
                auto [TBS, n, t0, t1, ost_len] = requests[j];
                add_g[t][n] = get_g(t, n);
                if (total_g[j] + add_g[t][n] >= TBS) {
                    score += 10;
                } else {
                    score += add_g[t][n] / TBS;
                }
            }
            return score;
        };

        for (int n = 0; n < N; n++) {
            for (int k = 0; k < K; k++) {
                for (int r = 0; r < R; r++) {
                    p[t][n][k][r] = 0;
                }
            }
        }

        deterministic_descent(t, js);

        /*set_power(t, build_weights_of_power(t, js));
        deterministic_descent(t, js);
        double scoreA = get_score_in_time();
        auto powerA = power_snapshot(t);

        for (int n = 0; n < N; n++) {
            for (int k = 0; k < K; k++) {
                for (int r = 0; r < R; r++) {
                    p[t][n][k][r] = 0;
                }
            }
        }

        deterministic_descent(t, js);
        double scoreB = get_score_in_time();
        //auto powerB = power_snapshot(t);

        if(scoreA > scoreB){
            set_power(t, powerA);
        }*/

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
            //cout << "lol: ";
            for (int r = 0; r < R; r++) {
                if (p[t][n][k][r] > 0) {
                    //cout << p[t][n][k][r] << ' ';
                    count++;
                    accum_prod *= p[t][n][k][r];
                    accum_prod *= s0[t][n][k][r];
                    //ASSERT(p[t][n][k][r] == 1, "failed");
                }
            }
            //cout << endl;

            if (count != 0) {
                sum += count * log2(1 + pow(accum_prod, 1.0 / count));
            }
            //cout << "kek: " << accum_prod << ' ' << count << endl;
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
#endif

#ifdef FAST_STREAM
    solution.print();
#endif

#ifndef FAST_STREAM
    }
#endif
}
