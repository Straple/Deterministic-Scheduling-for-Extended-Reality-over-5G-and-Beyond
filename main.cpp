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

}  // namespace KopeliovichStream

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
    if (y == 0) {
        swap(x, y);
    }
    if (x == 0) {
        return abs(y) < 1e-9;
    }
    return abs(x - y) <= 1e-9 * max({abs(x), abs(y)});
}

#define FAST_STREAM

//#define PRINT_DEBUG_INFO

//#define PRINT_SEARCH_INFO

//#define VERIFY_DP

//#define DEBUG_MODE

#ifdef DEBUG_MODE

#define FAILED_ASSERT(message)                                                \
    {                                                                         \
        std::cout << "assert failed at " __FILE__ << ":" << __LINE__ << '\n'; \
        std::cout << "message: \"" << (message) << "\"" << endl;              \
        std::exit(1);                                                         \
    }

#define ASSERT(condition, message) \
    if (!(condition))              \
    FAILED_ASSERT(message)

//_STL_VERIFY(condition, message) // don't work on CLion

#else

#define ASSERT(condition, message)  // condition

#endif  // DEBUG_MODE

time_point<steady_clock> global_time_start;

time_point<steady_clock> global_time_finish;

time_point<steady_clock> accum_time_start;

double accum_time = 0;

void my_time_start() {
#ifndef FAST_STREAM
    accum_time_start = steady_clock::now();
#endif
}

void my_time_end() {
#ifndef FAST_STREAM
    auto time_stop = steady_clock::now();
    auto duration = time_stop - accum_time_start;
    double time = duration_cast<nanoseconds>(duration).count() / 1e9;
    accum_time += time;
#endif
}

struct request_t {
    double TBS;
    int n;
    int t0;
    int t1;
};

constexpr uint32_t MAX_N = 100;
constexpr uint32_t MAX_K = 10;
constexpr uint32_t MAX_T = 1000;
constexpr uint32_t MAX_R = 10;
constexpr uint32_t MAX_J = 5000;

mt19937 rnd(42);

double get_rnd() {
    return rnd() * 1.0 / UINT_MAX;
}

struct Solution {
    int N;
    int K;
    int T;
    int R;
    int J;

    // s0[t][n][k][r]
    double s0[MAX_T][MAX_N][MAX_K][MAX_R];

    // s0[t][k][r][n]
    double s0_tkrn[MAX_T][MAX_K][MAX_R][MAX_N];

    // d[n][m][k][r]
    double d[MAX_N][MAX_N][MAX_K][MAX_R];

    // exp_d[n][m][k][r]
    double exp_d[MAX_N][MAX_N][MAX_K][MAX_R];

    // exp_d_2[n][k][r][m]
    double exp_d_2[MAX_N][MAX_K][MAX_R][MAX_N];

    // exp_d_pow[n][m][k][r]
    double exp_d_pow[MAX_N][MAX_N][MAX_K][MAX_R];

    vector<request_t> requests;

    // p[t][n][k][r]
    double p[MAX_T][MAX_N][MAX_K][MAX_R];

    double main_p[MAX_T][MAX_N][MAX_K][MAX_R];

    // main_total_g[j]
    double main_total_g[MAX_J];

    // add_g[t][n]
    double add_g[MAX_T][MAX_N];

    // main_add_g[t][n]
    double main_add_g[MAX_T][MAX_N];

    // main_best_f[t]
    double main_best_f[MAX_T];

    // dp_exp_d_prod[t][r][n][k]
    double dp_exp_d_prod[MAX_T][MAX_R][MAX_N][MAX_K];

    // dp_s[t][r][n][k]
    double dp_s[MAX_T][MAX_R][MAX_N][MAX_K];

    // dp_prod_s[t][n][k]
    double dp_prod_s[MAX_T][MAX_N][MAX_K];

    // dp_count[t][n][k]
    int dp_count[MAX_T][MAX_N][MAX_K];

    // dp_denom_sum[t][r][n][k]
    double dp_denom_sum[MAX_T][MAX_R][MAX_N][MAX_K];

    // dp_denom_sum_global_add[t][r][n]
    double dp_denom_sum_global_add[MAX_T][MAX_R][MAX_N];

    // dp_power_sum[t][k][r]
    double dp_power_sum[MAX_T][MAX_K][MAX_R];

    // dp_power_sum2[t][k]
    double dp_power_sum2[MAX_T][MAX_K];

    // js[t]
    vector<int> js[MAX_T];

    vector<int> nms[MAX_T];

    map<int, int> cast_n_to_j;

    // save_kr[t][n] = (k, r)
    int save_kr_index[MAX_T][MAX_N];

    // permute_kr[t][n]
    vector<pair<int, int>> permute_kr[MAX_T][MAX_N];

    int count_visited[MAX_T];

    set<pair<double, int>> Q;
    double value_in_Q[MAX_T];

    // changes_stack[t] = { (n, k, r, change) }
    vector<tuple<int, int, int, double>> changes_stack[MAX_T];

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
            requests[j].TBS += 1e-9;
            int t1 = t0 + td - 1;

            requests[j].t0 = t0;
            requests[j].t1 = t1;
        }

        for (int t = 0; t < T; t++) {
            for (int n = 0; n < N; n++) {
                for (int k = 0; k < K; k++) {
                    dp_prod_s[t][n][k] = 1;
                }
            }
        }

        for (int t = 0; t < T; t++) {
            for (int r = 0; r < R; r++) {
                for (int n = 0; n < N; n++) {
                    for (int k = 0; k < K; k++) {
                        dp_exp_d_prod[t][r][n][k] = 1;
                        dp_denom_sum[t][r][n][k] = 1;
                    }
                }
            }
        }

        for (int j = 0; j < J; j++) {
            auto [TBS, n, t0, t1] = requests[j];
            cast_n_to_j[n] = j;
            for (int t = t0; t <= t1; t++) {
                js[t].push_back(j);
                nms[t].push_back(n);
            }
        }
        for (int t = 0; t < T; t++) {
            sort(nms[t].begin(), nms[t].end());
        }

        for (int t = 0; t < T; t++) {
            main_best_f[t] = -1e300;
        }

        for (int t = 0; t < T; t++) {
            for (int n = 0; n < N; n++) {
                permute_kr[t][n].reserve(K * R);
                for (int k = 0; k < K; k++) {
                    for (int r = 0; r < R; r++) {
                        permute_kr[t][n].emplace_back(k, r);
                    }
                }

                sort(permute_kr[t][n].begin(), permute_kr[t][n].end(), [&](const auto &lhs, const auto &rhs) {
                    return s0[t][n][lhs.first][lhs.second] > s0[t][n][rhs.first][rhs.second];
                });
            }
        }
    }

    void print() {
        //ofstream cout("output.txt");
        for (int t = 0; t < T; t++) {
            for (int k = 0; k < K; k++) {
                for (int r = 0; r < R; r++) {
                    for (int n = 0; n < N; n++) {
#ifdef FAST_STREAM
                        writeDouble(main_p[t][n][k][r]);
                        writeChar(' ');
#else
                        cout << main_p[t][n][k][r] << ' ';
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

    // p[t][n][k][r] += change
    // with update dp for fast_f
    void change_power(int t, int n, int k, int r, double change) {
        // TODO: порядок очень важен

        // TODO: ОЧЕНЬ ВАЖНО
        if (change == 0) {
            return;
        }

#ifdef DEBUG_MODE
        if (high_equal(change, 0)) {
            cout << fixed << setprecision(50);
            cout << "lol:\n" << change << endl;
            ASSERT(false, "kek");
        }
#endif

        for (int m: nms[t]) {
            for (int k = 0; k < K; k++) {
                if (p[t][m][k][r] > 0) {
                    dp_prod_s[t][m][k] /= dp_s[t][r][m][k];
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
        dp_power_sum[t][k][r] += change;
        dp_power_sum2[t][k] += change;

        //double new_p = p[t][n][k][r];

        ASSERT(!(p[t][n][k][r] != 0 && high_equal(p[t][n][k][r], 0)), ":_(");
        // ====================

        for (int m: nms[t]) {
            if (m != n) {
                double x = change * s0_tkrn[t][k][r][m] / exp_d_2[n][k][r][m];
                dp_denom_sum_global_add[t][r][m] += x;
                dp_denom_sum[t][r][m][k] -= x;
            }
        }

        ASSERT((p[t][n][k][r] == change) == high_equal(p[t][n][k][r], change), "kek");

        // TODO: тут очень опасно делать сравнения по eps
        // можно получить неправильное обновление
        if (p[t][n][k][r] > 0 && p[t][n][k][r] == change) {
            // было ноль, стало не ноль
            dp_count[t][n][k]++;
            for (int m: nms[t]) {
                if (n != m) {
                    dp_exp_d_prod[t][r][m][k] *= exp_d_2[n][k][r][m];
                }
            }
        } else if (p[t][n][k][r] == 0) {
            // было не ноль, стало 0
            dp_count[t][n][k]--;
            for (int m: nms[t]) {
                if (n != m) {
                    dp_exp_d_prod[t][r][m][k] /= exp_d_2[n][k][r][m];
                }
            }
        }

        for (int m: nms[t]) {
            for (int k = 0; k < K; k++) {
                dp_s[t][r][m][k] = p[t][m][k][r] * s0[t][m][k][r] /
                                   (dp_denom_sum[t][r][m][k] +
                                    dp_denom_sum_global_add[t][r][m]) *
                                   dp_exp_d_prod[t][r][m][k];
            }
        }

        for (int m: nms[t]) {
            for (int k = 0; k < K; k++) {
                if (p[t][m][k][r] > 0) {
                    dp_prod_s[t][m][k] *= dp_s[t][r][m][k];
                }
            }
        }
    }

    double correct_f(int t) {
        double result = 0;
        for (int j: js[t]) {
            auto [TBS, n, t0, t1] = requests[j];
            double g = get_g(t, n);
            double x = 0;
            if (g + main_total_g[j] - main_add_g[t][n] > TBS) {
                x += 1e6;
            } else {
                x += g - TBS;
            }
            result += x;
        }
        return result;
    }

    void update_add_g(int t, int n) {
        double sum = 0;
        for (int k = 0; k < K; k++) {
            if (dp_count[t][n][k] != 0) {
                ASSERT(dp_prod_s[t][n][k] > 0, "what?");
                sum += dp_count[t][n][k] * log2(1 + pow(dp_prod_s[t][n][k], 1.0 / dp_count[t][n][k]));
            }
        }
        ASSERT(sum >= 0 && !is_spoiled(sum), "invalid g");
        add_g[t][n] = 192 * sum;
    }

    void update_add_g(int t) {
        for (int j: js[t]) {
            auto [TBS, n, t0, t1] = requests[j];
            update_add_g(t, n);
        }
    }

    double fast_f(int t) {
        double result = 0;
        for (int j: js[t]) {
            auto [TBS, n, t0, t1] = requests[j];
            double x = 0;
            // TODO: улучшить эту метрику
            if (add_g[t][n] + main_total_g[j] - main_add_g[t][n] > TBS) {
                x += 1e6;
            } else {
                x += add_g[t][n] - TBS;
            }
            result += x;
        }

#ifdef DEBUG_MODE
        if (!high_equal(result, correct_f(t))) {
            cout << "\n\nerror\n";
            cout << result << endl;
            cout << correct_f(t) << endl;
            cout << abs(result - correct_f(t)) << endl;
        }
#endif
        ASSERT(high_equal(result, correct_f(t)), "fatal");
        return result;
    }

    double calc_may_add_power(int t, int k, int r, double add) {
        {
#ifdef DEBUG_MODE
            double sum = 0;
            for (int n: nms[t]) {
                sum += p[t][n][k][r];
            }
            ASSERT(high_equal(dp_power_sum[t][k][r], sum), "kek");
            ASSERT(sum <= 4 + 1e-9, "fatal");
#endif
            add = min(add, 4 - 1e-9 - dp_power_sum[t][k][r]);
        }

        {
#ifdef DEBUG_MODE
            double sum = 0;
            for (int r = 0; r < R; r++) {
                sum += dp_power_sum[t][k][r];
            }
            ASSERT(high_equal(sum, dp_power_sum2[t][k]), "kek");
            ASSERT(sum <= R + 1e-9, "fatal");
#endif
            add = min(add, R - 1e-9 - dp_power_sum2[t][k]);
        }
        if (add < 1e-7) {
            add = 0;
        }
        return add;
    }

    double calc_value_in_Q(int t) {
        return count_visited[t] + 10 * main_best_f[t] / 1e6 / max(1, (int) js[t].size());
    }

    bool relax_main_version(int t) {
        double f = fast_f(t);
        bool relaxed = false;
        if (f > main_best_f[t]) {
            relaxed = true;
            main_best_f[t] = f;

            // update g
            for (int j: js[t]) {
                int n = requests[j].n;
                main_total_g[j] -= main_add_g[t][n];
                main_add_g[t][n] = add_g[t][n];
                main_total_g[j] += main_add_g[t][n];
            }

            // copy power
            for (int n = 0; n < N; n++) {
                for (int k = 0; k < K; k++) {
                    for (int r = 0; r < R; r++) {
                        main_p[t][n][k][r] = p[t][n][k][r];
                    }
                }
            }

            // relax main_best_f
            for (int time = max(0, t - 100); time < min(T, t + 100); time++) {
                if (time != t) {
                    main_best_f[time] = 0;
                    for (int j: js[time]) {
                        auto [TBS, n, t0, t1] = requests[j];
                        double x = 0;
                        if (main_total_g[j] > TBS) {
                            x += 1e6;
                        } else {
                            x += main_add_g[time][n] - TBS;
                        }
                        main_best_f[time] += x;
                    }

                    // update Q
                    {
                        Q.erase({value_in_Q[time], time});
                        value_in_Q[time] = calc_value_in_Q(time);
                        Q.insert({value_in_Q[time], time});
                    }
                }
            }
        }

#ifdef DEBUG_MODE
        for (int t = 0; t < T; t++) {
            for (int n = 0; n < N; n++) {
                for (int k = 0; k < K; k++) {
                    for (int r = 0; r < R; r++) {
                        swap(p[t][n][k][r], main_p[t][n][k][r]);
                    }
                }
            }
        }
        for (int t = 0; t < T; t++) {
            ASSERT(main_best_f[t] < -1e200 || high_equal(main_best_f[t], correct_f(t)), "kek");
        }

        for (int j = 0; j < J; j++) {
            auto [TBS, n, t0, t1] = requests[j];
            double g = 0;
            for (int t = t0; t <= t1; t++) {
                g += get_g(t, n);
                ASSERT(high_equal(main_add_g[t][n], get_g(t, n)), "kek");
            }
            ASSERT(high_equal(g, main_total_g[j]), "kek");
        }

        for (int t = 0; t < T; t++) {
            for (int n = 0; n < N; n++) {
                for (int k = 0; k < K; k++) {
                    for (int r = 0; r < R; r++) {
                        swap(p[t][n][k][r], main_p[t][n][k][r]);
                    }
                }
            }
        }
#endif
        return relaxed;
    }

    void deterministic_descent(int t, const vector<int> &js) {
        update_add_g(t);
        if (relax_main_version(t)) {
            changes_stack[t].clear();
        }

        vector<bool> view(N);
        vector<bool> dont_touch(N);

        for (int j: js) {
            auto [TBS, n, t0, t1] = requests[j];
            if (add_g[t][n] + main_total_g[j] - main_add_g[t][n] <= TBS) {
                view[n] = true;
            } else {
                view[n] = get_rnd() < 0.5;
                dont_touch[n] = !view[n];
            }
        }

        auto my_f = [&]() {
            double result = 0;
            for (int j: js) {
                auto [TBS, n, t0, t1] = requests[j];
                if (view[n]) {
                    double x = 0;
                    // TODO: улучшить эту метрику
                    if (add_g[t][n] + main_total_g[j] - main_add_g[t][n] > TBS) {
                        x += 1e6;
                    } else {
                        x += add_g[t][n] - TBS;
                    }
                    result += x;
                }
            }
            return result;
        };

        auto do_step = [&]() {  // NOLINT
            double best_f = -1e300;
            int best_n = -1;
            int best_k = -1;
            int best_r = -1;
            double best_add = 0;

            for (int n: nms[t]) {
                if (dont_touch[n]) {
                    continue;
                }
                int &i = save_kr_index[t][n];
                for (int step = 0; step < 4; step++) {
                    auto [k, r] = permute_kr[t][n][i];
                    i++;
                    if (i == permute_kr[t][n].size()) {
                        i = 0;
                    }

                    // add
                    {
                        double add = calc_may_add_power(t, k, r, 1.0);
                        if (add != 0) {
                            change_power(t, n, k, r, +add);
                            update_add_g(t);
                            double new_f = my_f();
                            change_power(t, n, k, r, -add);

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
                    if (p[t][n][k][r] != 0) {
                        double x = p[t][n][k][r];
                        change_power(t, n, k, r, -x);
                        update_add_g(t);
                        double new_f = my_f();
                        change_power(t, n, k, r, +x);

                        if (best_f < new_f) {
                            best_f = new_f;
                            best_n = n;
                            best_k = k;
                            best_r = r;
                            best_add = -x;
                        }
                    }

                    // sub
                    if (p[t][n][k][r] != 0) {
                        double sub = p[t][n][k][r] / 2;
                        if (sub > 0.1) {
                            change_power(t, n, k, r, -sub);
                            update_add_g(t);
                            double new_f = my_f();
                            change_power(t, n, k, r, +sub);

                            if (best_f < new_f) {
                                best_add = -sub;
                                best_n = n;
                                best_f = new_f;
                                best_k = k;
                                best_r = r;
                            }
                        }
                    }
                }
            }

            if (best_n == -1) {
                return;
            }

            change_power(t, best_n, best_k, best_r, best_add);
            changes_stack[t].emplace_back(best_n, best_k, best_r, best_add);
            update_add_g(t);
            if (relax_main_version(t)) {
                changes_stack[t].clear();
            }
        };

        for (int step = 0; step < 20; step++) {
            do_step();
        }

        // возвращение к main
        /*if (changes_stack[t].size() > 50) {
            while (!changes_stack[t].empty()) {
                auto [n, k, r, change] = changes_stack[t].back();
                changes_stack[t].pop_back();
                change_power(t, n, k, r, -change);
            }
        }*/

        if (changes_stack[t].size() > 20) {
            changes_stack[t].pop_back();

            vector<pair<double, int>> kek;
            for (int j: js) {
                auto [TBS, n, t0, t1] = requests[j];
                if (add_g[t][n] + main_total_g[j] - main_add_g[t][n] < TBS) {
                    kek.emplace_back(add_g[t][n] + main_total_g[j] - main_add_g[t][n] - TBS, n);
                }
            }
            sort(kek.begin(), kek.end(), greater<>());
            int threshold = kek.size() / 2;
            while (kek.size() > threshold) {
                auto [weight, n] = kek.back();
                kek.pop_back();


                for (int k = 0; k < K; k++) {
                    for (int r = 0; r < R; r++) {
                        if (p[t][n][k][r] != 0) {
                            change_power(t, n, k, r, -p[t][n][k][r]);
                            update_add_g(t);
                            relax_main_version(t);
                        }
                    }
                }
            }
        }
    }

    void set_nice_power(int t, const vector<int> &js) {
        if (js.empty()) {
            return;
        }

        // наиболее оптимально расставить силу так
        // что это значит? наверное мы хотим как можно больший прирост g
        // а также чтобы доотправлять сообщения

        deterministic_descent(t, js);
    }

    void solve() {
        for (int t = 0; t < T; t++) {
            if (!js[t].empty()) {
                value_in_Q[t] = js[t].size() - 1e5; // TODO: может менее строже?
                Q.insert({value_in_Q[t], t});
            }
        }

        if (Q.empty()) {
            return;
        }

        while (true) {
            // выберем самое лучшее время, куда наиболее оптимально поставим
            // силу

            {
                auto time_stop = steady_clock::now();
                if (time_stop > global_time_finish) {
                    break;
                }
            }

            int best_time = -1;

            {
                for (auto it = Q.begin(); it != Q.end(); it++) {
                    int t = it->second;
                    int count_accepted = 0;
                    for (int j: js[t]) {
                        auto [TBS, n, t0, t1] = requests[j];
                        count_accepted += main_total_g[j] > TBS;
                    }

                    if (count_accepted != js[t].size()) {
                        best_time = t;
                        Q.erase(it);
                        break;
                    }
                }
                //best_time = Q.begin()->second;
                //Q.erase(Q.begin());
            }

            if (best_time == -1) {
                break; // full score
            }

            // наиболее оптимально расставим силу в момент времени best_time
            count_visited[best_time]++;
            set_nice_power(best_time, js[best_time]);

            {
                value_in_Q[best_time] = calc_value_in_Q(best_time);
                /*weight += count_visited[best_time] * count_visited[best_time];
                int count_accepted = 0;
                for (int j: js[best_time]) {
                    auto [TBS, n, t0, t1] = requests[j];
                    count_accepted += main_total_g[j] > TBS;
                }
                weight += count_accepted * 1.0 / (int) js[best_time].size() * 5;*/

                Q.insert({value_in_Q[best_time], best_time});
            }
        }
#ifndef FAST_STREAM
        int sum_count = 0;
        for (int t = 0; t < T; t++) {
            sum_count += count_visited[t];
        }
        cout << "total count visited: " << sum_count << ' ' << sum_count * 1.0 / T << endl;

        for (int t = 0; t < T; t++) {
            int count_accepted = 0;
            for (int j: js[t]) {
                auto [TBS, n, t0, t1] = requests[j];
                count_accepted += main_total_g[j] > TBS;
            }
            cout << t << ' ' << count_visited[t] << ' ' << count_accepted << '/' << js[t].size() << endl;
        }
#endif
    }

    double get_score() {
        double power_sum = 0;
        for (int t = 0; t < T; t++) {
            for (int n = 0; n < N; n++) {
                for (int k = 0; k < K; k++) {
                    for (int r = 0; r < R; r++) {
                        power_sum += main_p[t][n][k][r];
                        p[t][n][k][r] = main_p[t][n][k][r];
                    }
                }
            }
        }
        int X = 0;
        for (int j = 0; j < J; j++) {
            auto [TBS, n, t0, t1] = requests[j];
#ifdef DEBUG_MODE
            double g = 0;
            for (int t = t0; t <= t1; t++) {
                g += get_g(t, n);
            }
            ASSERT(high_equal(g, main_total_g[j]), "oh ho :_(");
#endif
            X += main_total_g[j] > requests[j].TBS;
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
                            dp_sum[k][r] += s0[t][n][k][r] * p[t][m][k][r] /
                                            exp_d[n][m][k][r];
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
} solution;

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(0);
    std::cout.tie(0);

#ifndef FAST_STREAM
    for (int test_case = 2; test_case <= 2; test_case++) {
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

        global_time_start = steady_clock::now();
        global_time_finish = global_time_start + nanoseconds(1'900 * 1'000'000);

#ifdef FAST_STREAM
        solution.read();
#else
        solution.read(input);
#endif

        solution.solve();

#ifndef FAST_STREAM
        auto time_stop = steady_clock::now();
        auto duration = time_stop - global_time_start;
        double time = duration_cast<nanoseconds>(duration).count() / 1e9;
        cout << solution.get_score() << '/' << solution.J << ' ' << time << "s"
             << endl;
        cout << "ACCUM TIME: " << accum_time << "s" << endl;
        accum_time = 0;
#endif


#ifdef FAST_STREAM
        solution.print();
#endif

#ifndef FAST_STREAM
    }
#endif
}
