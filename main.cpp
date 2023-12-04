#pragma GCC optimization ("O3")
#pragma GCC target("avx,avx2,fma")
#pragma GCC optimize("Ofast")
#pragma GCC optimization ("unroll-loops")

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
    /*if (is_spoiled(x) || is_spoiled(y)) {
        exit(1);
    }*/
    if (x == 0 || y == 0) {
        return abs(x - y) < 1e-9;
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
    uint32_t n;
    uint32_t t0;
    uint32_t t1;
};

constexpr uint32_t MAX_N = 100;
constexpr uint32_t MAX_K = 10;
constexpr uint32_t MAX_T = 1000;
constexpr uint32_t MAX_R = 10;
constexpr uint32_t MAX_J = 5000;

mt19937 gen(33);
static std::uniform_int_distribution<> dis(0, 100);

double get_rnd() {
    return dis(gen) / 100.0;
}

struct Solution {
    uint32_t N;
    uint32_t K;
    uint32_t T;
    uint32_t R;
    uint32_t J;

    // s0[t][r][k][n]
    double s0[MAX_T][MAX_R][MAX_K][MAX_N];

    // s0_tkrn[t][k][r][n]
    double s0_tkrn[MAX_T][MAX_K][MAX_R][MAX_N];

    // d[n][m][k][r]
    double d[MAX_N][MAX_N][MAX_K][MAX_R];

    // exp_d[n][m][k][r]
    double exp_d[MAX_N][MAX_N][MAX_K][MAX_R];

    // exp_d_2[n][k][r][m]
    double exp_d_2[MAX_N][MAX_K][MAX_R][MAX_N];

    // p[t][r][n][k]
    double p[MAX_T][MAX_R][MAX_N][MAX_K];

    double main_p[MAX_T][MAX_R][MAX_N][MAX_K];

    // main_total_g[j]
    double main_total_g[MAX_J];

    // add_g[t][n]
    double add_g[MAX_T][MAX_N];

    // main_add_g[t][n]
    double main_add_g[MAX_T][MAX_N];

    // main_best_f[t]
    double main_best_f[MAX_T];

    // dp_exp_d_prod[t][r][k][n]
    double dp_exp_d_prod[MAX_T][MAX_R][MAX_K][MAX_N];

    // dp_s[t][r][n][k]
    double dp_s[MAX_T][MAX_R][MAX_N][MAX_K];

    // dp_prod_s[t][n][k]
    double dp_prod_s[MAX_T][MAX_N][MAX_K];

    // dp_count[t][n][k]
    uint32_t dp_count[MAX_T][MAX_N][MAX_K];

    // dp_denom_sum[t][r][k][n]
    double dp_denom_sum[MAX_T][MAX_R][MAX_K][MAX_N];

    // dp_denom_sum_global_add[t][r][n]
    double dp_denom_sum_global_add[MAX_T][MAX_R][MAX_N];

    // dp_power_sum[t][k][r]
    double dp_power_sum[MAX_T][MAX_K][MAX_R];

    // dp_power_sum2[t][k]
    double dp_power_sum2[MAX_T][MAX_K];

    // js[t]
    vector<uint32_t> js[MAX_T];

    vector<uint32_t> nms[MAX_T];

    // save_kr[t][n] = (k, r)
    uint32_t save_kr_index[MAX_T][MAX_N];

    // permute_kr[t][n]
    vector<pair<uint32_t, uint32_t>> permute_kr[MAX_T][MAX_N];

    uint32_t count_visited[MAX_T];

    set<pair<double, uint32_t>> Q;
    double value_in_Q[MAX_T];

    // changes_stack[t] = { (n, k, r, change) }
    vector<tuple<uint32_t, uint32_t, uint32_t, double>> changes_stack[MAX_T];

    vector<request_t> requests;

    // nk_with_not_zero_p[t][r]
    vector<pair<uint32_t, uint32_t>> nk_with_not_zero_p[MAX_T][MAX_R];

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
        for (uint32_t t = 0; t < T; t++) {
            for (uint32_t k = 0; k < K; k++) {
                for (uint32_t r = 0; r < R; r++) {
                    for (uint32_t n = 0; n < N; n++) {
#ifdef FAST_STREAM
                        s0[t][r][k][n] = readDouble();
#else
                        input >> s0[t][r][k][n];
#endif
                        s0_tkrn[t][k][r][n] = s0[t][r][k][n];
                    }
                }
            }
        }

        for (uint32_t k = 0; k < K; k++) {
            for (uint32_t r = 0; r < R; r++) {
                for (uint32_t m = 0; m < N; m++) {
                    for (uint32_t n = 0; n < N; n++) {
#ifdef FAST_STREAM
                        d[n][m][k][r] = readDouble();
#else
                        input >> d[n][m][k][r];
#endif
                        exp_d[n][m][k][r] = exp(d[n][m][k][r]);
                        exp_d_2[n][k][r][m] = exp_d[n][m][k][r];
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
        for (uint32_t i = 0; i < J; i++) {
            uint32_t j;
            uint32_t t0, td;

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
            uint32_t t1 = t0 + td - 1;

            requests[j].t0 = t0;
            requests[j].t1 = t1;
        }

        for (uint32_t t = 0; t < T; t++) {
            for (uint32_t n = 0; n < N; n++) {
                for (uint32_t k = 0; k < K; k++) {
                    dp_prod_s[t][n][k] = 1;
                }
            }
        }

        for (uint32_t t = 0; t < T; t++) {
            for (uint32_t r = 0; r < R; r++) {
                for (uint32_t k = 0; k < K; k++) {
                    for (uint32_t n = 0; n < N; n++) {
                        dp_exp_d_prod[t][r][k][n] = 1;
                        dp_denom_sum[t][r][k][n] = 1;
                    }
                }
            }
        }

        for (uint32_t j = 0; j < J; j++) {
            auto [TBS, n, t0, t1] = requests[j];
            for (uint32_t t = t0; t <= t1; t++) {
                js[t].push_back(j);
                nms[t].push_back(n);
            }
        }
        for (uint32_t t = 0; t < T; t++) {
            sort(nms[t].begin(), nms[t].end());
        }

        for (uint32_t t = 0; t < T; t++) {
            main_best_f[t] = -1e300;
        }

        for (uint32_t t = 0; t < T; t++) {
            for (uint32_t n = 0; n < N; n++) {
                permute_kr[t][n].reserve(K * R);
                for (uint32_t k = 0; k < K; k++) {
                    for (uint32_t r = 0; r < R; r++) {
                        permute_kr[t][n].emplace_back(k, r);
                    }
                }

                sort(permute_kr[t][n].begin(), permute_kr[t][n].end(), [&](const auto &lhs, const auto &rhs) {
                    return s0[t][lhs.second][lhs.first][n] > s0[t][rhs.second][rhs.first][n];
                });
            }
        }
    }

    void print() {
        //ofstream cout("output.txt");
        for (uint32_t t = 0; t < T; t++) {
            for (uint32_t k = 0; k < K; k++) {
                for (uint32_t r = 0; r < R; r++) {
                    for (uint32_t n = 0; n < N; n++) {
#ifdef FAST_STREAM
                        writeDouble(main_p[t][r][n][k]);
                        writeChar(' ');
#else
                        cout << main_p[t][r][n][k] << ' ';
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

    bool verify_power(uint32_t t) {
        vector<double> sum(K);
        for (uint32_t n = 0; n < N; n++) {
            for (uint32_t k = 0; k < K; k++) {
                for (uint32_t r = 0; r < R; r++) {
                    sum[k] += p[t][r][n][k];
                }
            }
        }
        for (uint32_t k = 0; k < K; k++) {
            if (sum[k] > R + 1e-9) {
                return false;
            }
        }
        for (uint32_t k = 0; k < K; k++) {
            for (uint32_t r = 0; r < R; r++) {
                double sum = 0;
                for (uint32_t n = 0; n < N; n++) {
                    sum += p[t][r][n][k];
                }
                if (sum > 4 + 1e-9) {
                    return false;
                }
            }
        }
        return true;
    }

    double sum_power(uint32_t t, uint32_t n) {
        double sum = 0;
        for (uint32_t r = 0; r < R; r++) {
            for (uint32_t k = 0; k < K; k++) {
                sum += p[t][r][n][k];
            }
        }
        return sum;
    }

    // p[t][r][n][k] += change
    // with update dp for fast_f
    void change_power(uint32_t t, uint32_t n, uint32_t k, uint32_t r, double change) {
        // TODO: порядок очень важен

        // TODO: ОЧЕНЬ ВАЖНО
        if (change == 0) {
            return;
        }

        /// МЕРИЛ ВЕЗДЕ 5s

#ifdef DEBUG_MODE
        if (high_equal(change, 0)) {
            cout << fixed << setprecision(50);
            cout << "lol:\n" << change << endl;
            ASSERT(false, "kek");
        }
#endif

        // 0.645924s -> 0.501693s -> 0.309104s
        for (auto [m, k]: nk_with_not_zero_p[t][r]) {
            dp_prod_s[t][m][k] /= dp_s[t][r][m][k];
        }

        ASSERT(!(p[t][r][n][k] != 0 && high_equal(p[t][r][n][k], 0)), ":_(");
        // ====================

        // 0.168692s
        if (high_equal(p[t][r][n][k] + change, 0)) {
            change = -p[t][r][n][k];
            p[t][r][n][k] = 0;
        } else {
            p[t][r][n][k] += change;
        }
        dp_power_sum[t][k][r] += change;
        dp_power_sum2[t][k] += change;

        ASSERT(!(p[t][r][n][k] != 0 && high_equal(p[t][r][n][k], 0)), ":_(");
        // ====================

        // 0.515646s
        for (uint32_t m: nms[t]) {
            if (m != n) {
                double x = change * s0_tkrn[t][k][r][m] / exp_d_2[n][k][r][m];
                dp_denom_sum_global_add[t][r][m] += x;
                dp_denom_sum[t][r][k][m] -= x;
            }
        }

        ASSERT((p[t][r][n][k] == change) == high_equal(p[t][r][n][k], change), "kek");

        // 0.294296s -> 0.408725s -> 0.5231s

        // TODO: тут очень опасно делать сравнения по eps
        // можно получить неправильное обновление
        if (p[t][r][n][k] > 0 && p[t][r][n][k] == change) {
            // было ноль, стало не ноль
            auto &data = nk_with_not_zero_p[t][r];
            data.emplace_back(n, k);
            for (uint32_t i = data.size() - 1; i > 0 && data[i - 1] > data[i]; i--) {
                swap(data[i - 1], data[i]);
            }

            dp_count[t][n][k]++;

            for (uint32_t m: nms[t]) {
                dp_exp_d_prod[t][r][k][m] *= exp_d_2[n][k][r][m];
            }
            dp_exp_d_prod[t][r][k][n] /= exp_d_2[n][k][r][n];
        } else if (p[t][r][n][k] == 0) {
            // было не ноль, стало 0

            auto &data = nk_with_not_zero_p[t][r];
            data.erase(find(data.begin(), data.end(), make_pair(n, k)));

            dp_count[t][n][k]--;

            for (uint32_t m: nms[t]) {
                dp_exp_d_prod[t][r][k][m] /= exp_d_2[n][k][r][m];
            }
            dp_exp_d_prod[t][r][k][n] *= exp_d_2[n][k][r][n];
        }

        //0.944143s -> 0.502808s -> 0.396134s
        for (auto [m, k]: nk_with_not_zero_p[t][r]) {
            dp_s[t][r][m][k] = p[t][r][m][k] * s0[t][r][k][m] /
                               (dp_denom_sum[t][r][k][m] +
                                dp_denom_sum_global_add[t][r][m]) *
                               dp_exp_d_prod[t][r][k][m];
        }

        // 0.54641s -> 0.379976s -> 0.259082s
        for (auto [m, k]: nk_with_not_zero_p[t][r]) {
            dp_prod_s[t][m][k] *= dp_s[t][r][m][k];
        }
    }

    double fooo(double g, double add_g, double TBS, uint32_t t, uint32_t n) {
        // TODO: улучшить эту метрику
        if (g > TBS) {
            return 1e6;// - sum_power(t, n) * 10;
        } else {
            return add_g - TBS;
        }
    }

    double correct_f(uint32_t t) {
        double result = 0;
        for (uint32_t j: js[t]) {
            auto [TBS, n, t0, t1] = requests[j];
            double g = get_g(t, n);
            result += fooo(g + main_total_g[j] - main_add_g[t][n], g, TBS, t, n);
        }
        return result;
    }

    void update_add_g(uint32_t t) {
        for (uint32_t n: nms[t]) {
            double sum = 0;
            for (uint32_t k = 0; k < K; k++) {
                if (dp_count[t][n][k] != 0) {
                    ASSERT(dp_prod_s[t][n][k] > 0, "what?");
                    sum += dp_count[t][n][k] * log2(1 + pow(dp_prod_s[t][n][k], 1.0 / dp_count[t][n][k]));
                }
            }
            ASSERT(sum >= 0 && !is_spoiled(sum), "invalid g");
            add_g[t][n] = 192 * sum;
        }
    }

    double fast_f(uint32_t t) {
        double result = 0;
        for (uint32_t j: js[t]) {
            auto [TBS, n, t0, t1] = requests[j];
            result += fooo(add_g[t][n] + main_total_g[j] - main_add_g[t][n], add_g[t][n], TBS, t, n);
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

    double calc_may_add_power(uint32_t t, uint32_t k, uint32_t r, double add) {
        {
#ifdef DEBUG_MODE
            double sum = 0;
            for (uint32_t n: nms[t]) {
                sum += p[t][r][n][k];
            }
            ASSERT(high_equal(dp_power_sum[t][k][r], sum), "kek");
            ASSERT(sum <= 4 + 1e-9, "fatal");
#endif
            add = min(add, 4 - 1e-9 - dp_power_sum[t][k][r]);
        }

        {
#ifdef DEBUG_MODE
            double sum = 0;
            for (uint32_t r = 0; r < R; r++) {
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

    double calc_value_in_Q(uint32_t t) {
        uint32_t count_accepted = 0;
        for (uint32_t j: js[t]) {
            auto [TBS, n, t0, t1] = requests[j];
            count_accepted += main_total_g[j] > TBS;
        }
        return (count_accepted == js[t].size()) * 100 * (1 + count_visited[t]) +
               count_visited[t] +
               10 * main_best_f[t] / 1e6 / (js[t].size() + 1);
    }

    bool relax_main_version(uint32_t t) {
        double f = fast_f(t);
        bool relaxed = false;
        if (f > main_best_f[t]) {
            relaxed = true;
            main_best_f[t] = f;

            // update g
            for (uint32_t j: js[t]) {
                uint32_t n = requests[j].n;
                main_total_g[j] -= main_add_g[t][n];
                main_add_g[t][n] = add_g[t][n];
                main_total_g[j] += main_add_g[t][n];
            }

            // copy power
            for (uint32_t n = 0; n < N; n++) {
                for (uint32_t k = 0; k < K; k++) {
                    for (uint32_t r = 0; r < R; r++) {
                        main_p[t][r][n][k] = p[t][r][n][k];
                    }
                }
            }

            // relax main_best_f
            uint32_t left_time = 0;
            if (t > 100) {
                left_time -= 100;
            }
            uint32_t right_time = min(T, t + 100);

            for (uint32_t time = left_time; time < right_time; time++) {
                if (time != t) {
                    main_best_f[time] = 0;
                    for (uint32_t j: js[time]) {
                        auto [TBS, n, t0, t1] = requests[j];
                        main_best_f[time] += fooo(main_total_g[j], main_add_g[time][n], TBS, time, n);
                    }

                    // update Q
                    /*{
                        Q.erase({value_in_Q[time], time});
                        value_in_Q[time] = calc_value_in_Q(time);
                        Q.insert({value_in_Q[time], time});
                    }*/
                }
            }
        }

#ifdef DEBUG_MODE
        for (uint32_t t = 0; t < T; t++) {
            for (uint32_t n = 0; n < N; n++) {
                for (uint32_t k = 0; k < K; k++) {
                    for (uint32_t r = 0; r < R; r++) {
                        swap(p[t][r][n][k], main_p[t][r][n][k]);
                    }
                }
            }
        }
        for (uint32_t t = 0; t < T; t++) {
            ASSERT(main_best_f[t] < -1e200 || high_equal(main_best_f[t], correct_f(t)), "kek");
        }

        for (uint32_t j = 0; j < J; j++) {
            auto [TBS, n, t0, t1] = requests[j];
            double g = 0;
            for (uint32_t t = t0; t <= t1; t++) {
                g += get_g(t, n);
                ASSERT(high_equal(main_add_g[t][n], get_g(t, n)), "kek");
            }
            ASSERT(high_equal(g, main_total_g[j]), "kek");
        }

        for (uint32_t t = 0; t < T; t++) {
            for (uint32_t n = 0; n < N; n++) {
                for (uint32_t k = 0; k < K; k++) {
                    for (uint32_t r = 0; r < R; r++) {
                        swap(p[t][r][n][k], main_p[t][r][n][k]);
                    }
                }
            }
        }
#endif
        return relaxed;
    }

    void deterministic_descent(uint32_t t) {
        update_add_g(t);
        if (relax_main_version(t)) {
            changes_stack[t].clear();
        }

        vector<bool> view(N);
        vector<bool> dont_touch(N);

        for (uint32_t j: js[t]) {
            auto [TBS, n, t0, t1] = requests[j];
            if (add_g[t][n] + main_total_g[j] - main_add_g[t][n] <= TBS) {
                view[n] = true;
            } else {
                view[n] = get_rnd() < 0.5;//(add_g[t][n] + main_total_g[j] - main_add_g[t][n]) / TBS - 0.9;
                dont_touch[n] = !view[n];
            }
        }

        auto my_f = [&]() {
            return fast_f(t);

            double result = 0;
            for (uint32_t j: js[t]) {
                auto [TBS, n, t0, t1] = requests[j];
                if (view[n]) {
                    result += fooo(add_g[t][n] + main_total_g[j] - main_add_g[t][n], add_g[t][n], TBS, t, n);
                }
            }
            return result;
        };

        auto do_step = [&]() { // NOLINT
            double best_f = -1e300;
            uint32_t best_n = -1;
            uint32_t best_k = -1;
            uint32_t best_r = -1;
            double best_add = 0;

            for (int step = 0; step < 4; step++) {
                int n = nms[t][gen() % nms[t].size()];
                if (dont_touch[n]) {
                    continue;
                }
                for (uint32_t k = 0; k < K; k++) {
                    for (uint32_t r = 0; r < R; r++) {
                        if (p[t][r][n][k] == 0) {
                            continue;
                        }

                        // set zero
                        {
                            double x = p[t][r][n][k];
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
                        {
                            double sub = p[t][r][n][k] * 0.5;
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

                uint32_t &i = save_kr_index[t][n];
                for (uint32_t step = 0; step < 4; step++) {
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
                }

                i++;
                if (i == permute_kr[t][n].size()) {
                    i = 0;
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

        const uint32_t steps_count = 25 - min(15U, count_visited[t]);

        for (uint32_t step = 0; step < steps_count; step++) {
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

        if (changes_stack[t].size() > 15) {
            changes_stack[t].pop_back();

            /*for (int j: js[t]) {
                auto [TBS, n, t0, t1] = requests[j];
                auto kek = permute_kr[t][n];
                reverse(kek.begin(), kek.end());
                for (auto [k, r]: kek) {
                    double g = add_g[t][n] + main_total_g[j] - main_add_g[t][n];
                    if (p[t][r][n][k] > 0) {
                        double x = p[t][r][n][k] / 2;
                        if (x < 0.1) {
                            x = p[t][r][n][k];
                        }
                        change_power(t, n, k, r, -x);
                        update_add_g(t);

                        double new_g = add_g[t][n] + main_total_g[j] - main_add_g[t][n];
                        if (new_g > g) {
                            //cout << "completed: " << g << "->" << new_g << '\n';
                        } else {
                            change_power(t, n, k, r, +x);
                        }
                    }
                }
            }*/

            vector<pair<double, uint32_t>> kek;
            for (uint32_t j: js[t]) {
                auto [TBS, n, t0, t1] = requests[j];
                if (add_g[t][n] + main_total_g[j] - main_add_g[t][n] < TBS) {
                    kek.emplace_back(add_g[t][n] + main_total_g[j] - main_add_g[t][n] - TBS, n);
                }
            }
            sort(kek.begin(), kek.end(), greater<>());
            uint32_t threshold = kek.size() / 2;
            while (kek.size() > threshold) {
                auto [weight, n] = kek.back();
                kek.pop_back();

                for (uint32_t k = 0; k < K; k++) {
                    for (uint32_t r = 0; r < R; r++) {
                        if (p[t][r][n][k] != 0) {
                            change_power(t, n, k, r, -p[t][r][n][k]);
                        }
                    }
                }

                update_add_g(t);
                relax_main_version(t);
            }
        }
    }

    void set_nice_power(uint32_t t) {
        if (js[t].empty()) {
            return;
        }

        // наиболее оптимально расставить силу так
        // что это значит? наверное мы хотим как можно больший прирост g
        // а также чтобы доотправлять сообщения

        deterministic_descent(t);
    }

    void solve() {
        for (uint32_t t = 0; t < T; t++) {
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

            /*for (int i = 0; i < 30; i++) {
                set_nice_power(0);
            }
            exit(0);*/

            {
                auto time_stop = steady_clock::now();
                if (time_stop > global_time_finish) {
                    break;
                }
            }

            uint32_t best_time = -1;

            {
                for (auto it = Q.begin(); it != Q.end(); it++) {
                    uint32_t t = it->second;
                    uint32_t count_accepted = 0;
                    for (uint32_t j: js[t]) {
                        auto [TBS, n, t0, t1] = requests[j];
                        count_accepted += main_total_g[j] > TBS;
                    }

                    if (count_accepted != js[t].size()) {
                        best_time = t;
                        Q.erase(it);
                        break;
                    }
                }
            }

            if (best_time == -1) {
                break; // full score
            }

            // наиболее оптимально расставим силу в момент времени best_time
            count_visited[best_time]++;
            set_nice_power(best_time);

            {
                value_in_Q[best_time] = calc_value_in_Q(best_time);
                Q.insert({value_in_Q[best_time], best_time});
            }
        }
#ifndef FAST_STREAM
        uint32_t sum_count = 0;
        for (uint32_t t = 0; t < T; t++) {
            sum_count += count_visited[t];
        }
        cout << "total count visited: " << sum_count << ' ' << sum_count * 1.0 / T << endl;

        for (uint32_t t = 0; t < T; t++) {
            uint32_t count_accepted = 0;
            for (uint32_t j: js[t]) {
                auto [TBS, n, t0, t1] = requests[j];
                count_accepted += main_total_g[j] > TBS;
            }
            cout << t << ' ' << count_visited[t] << ' ' << count_accepted << '/' << js[t].size() << endl;
        }
#endif
    }

    double get_score() {
        double power_sum = 0;
        for (uint32_t t = 0; t < T; t++) {
            for (uint32_t n = 0; n < N; n++) {
                for (uint32_t k = 0; k < K; k++) {
                    for (uint32_t r = 0; r < R; r++) {
                        power_sum += main_p[t][r][n][k];
                        p[t][r][n][k] = main_p[t][r][n][k];
                    }
                }
            }
        }
        uint32_t X = 0;
        for (uint32_t j = 0; j < J; j++) {
            auto [TBS, n, t0, t1] = requests[j];
#ifdef DEBUG_MODE
            double g = 0;
            for (uint32_t t = t0; t <= t1; t++) {
                g += get_g(t, n);
            }
            ASSERT(high_equal(g, main_total_g[j]), "oh ho :_(");
#endif
            X += main_total_g[j] > requests[j].TBS;
        }
        return X - 1e-6 * power_sum;
    }

    double correct_get_g(uint32_t t, uint32_t n) {
        vector<vector<double>> dp_s0_p_d(K, vector<double>(R));

        // update dp_s0_p_d
        {
            // dp_sum[k][r]
            vector<vector<double>> dp_sum(K, vector<double>(R));
            for (uint32_t m = 0; m < N; m++) {
                if (m != n) {
                    for (uint32_t k = 0; k < K; k++) {
                        for (uint32_t r = 0; r < R; r++) {
                            dp_sum[k][r] += s0[t][r][k][n] * p[t][r][m][k] /
                                            exp_d[n][m][k][r];
                        }
                    }
                }
            }

            vector<double> dp_sum_2(R);
            for (uint32_t k = 0; k < K; k++) {
                for (uint32_t r = 0; r < R; r++) {
                    dp_sum_2[r] += dp_sum[k][r];
                }
            }

            for (uint32_t k = 0; k < K; k++) {
                for (uint32_t r = 0; r < R; r++) {
                    dp_s0_p_d[k][r] = 1 + dp_sum_2[r] - dp_sum[k][r];
                }
            }
        }

        double sum = 0;
        for (uint32_t k = 0; k < K; k++) {
            double accum_prod = 1;
            uint32_t count = 0;
            for (uint32_t r = 0; r < R; r++) {
                if (p[t][r][n][k] > 0) {
                    count++;
                    accum_prod *= p[t][r][n][k];
                    accum_prod *= s0[t][r][k][n];
                    accum_prod /= dp_s0_p_d[k][r];

                    for (uint32_t m = 0; m < N; m++) {
                        if (n != m) {
                            if (p[t][r][m][k] > 0) {
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

    double get_g(uint32_t t, uint32_t n) {
        return correct_get_g(t, n);

        double sum = 0;
        for (uint32_t k = 0; k < K; k++) {
            double accum_prod = 1;
            uint32_t count = 0;
            for (uint32_t r = 0; r < R; r++) {
                if (p[t][r][n][k] > 0) {
                    count++;
                    accum_prod *= p[t][r][n][k];
                    accum_prod *= s0[t][r][k][n];
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
    for (uint32_t test_case = 2; test_case <= 2; test_case++) {
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
        global_time_finish = global_time_start + nanoseconds(1'900'000'000ULL);

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
        cout << solution.get_score() << '/' << solution.J << ' ' << time << "s" << endl;
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
