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

    static const int buf_size = 4096;

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

bool is_spoiled(double num) {
    return std::isnan(num) || std::isinf(num);
}

//#define FAST_STREAM

#define DEBUG_MODE

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
    } use_build_weight_version = ALL;

    // s0[t][n][k][r]
    vector<vector<vector<vector<double>>>> s0;

    // d[n][m][k][r]
    vector<vector<vector<vector<double>>>> d;

    // exp_d[n][m][k][r]
    vector<vector<vector<vector<double>>>> exp_d;

    // exp_d_pow[n][m][k][r]
    vector<vector<vector<vector<double>>>> exp_d_pow;

    vector<request_t> requests;
    vector<bool> used_request;
    // count_of_set_r_for_request[j]
    vector<int> count_of_set_r_for_request;

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
        for (int t = 0; t < T; t++) {
            for (int k = 0; k < K; k++) {
                for (int r = 0; r < R; r++) {
                    for (int n = 0; n < N; n++) {
#ifdef FAST_STREAM
                        s0[t][n][k][r] = readDouble();
#else
                        input >> s0[t][n][k][r];
#endif
                    }
                }
            }
        }

        d.assign(N, vector(N, vector(K, vector<double>(R))));
        exp_d.assign(N, vector(N, vector(K, vector<double>(R))));
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

                count_of_set_r_for_request[j]++;
                // cout << count_of_set_r_for_request[j] << '\n';

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

/*#ifdef DEBUG_MODE
        verify();
#endif*/

        return set_of_weights;
    }

    void set_nice_power(int t, const vector<int> &js) {
        // наиболее оптимально расставить силу так
        // что это значит? наверное мы хотим как можно больший прирост g
        // а также чтобы доотправлять сообщения

        set_power(t, build_weights_of_power(t, js));

        return;

        // для каждого r выберем какой запрос поставим

        uint64_t s = js.size() + 1;
        uint64_t MAX = 1;
        for (int r = 0; r < R; r++) {
            MAX *= s;
        }

        cout << t << ' ' << pow(s, R) << ' ' << MAX << ' ';
        cout.flush();

        auto set_power_for_msk = [&](uint64_t msk) { // NOLINT
            // set zero power
            for (int n = 0; n < N; n++) {
                for (int k = 0; k < K; k++) {
                    for (int r = 0; r < R; r++) {
                        p[t][n][k][r] = 0;
                    }
                }
            }

            // (weight, n, k, r)
            vector<tuple<double, int, int, int>> set_of_weights;

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

            uint64_t x = msk;
            for (int r = 0; r < R; r++, x /= s) {
                uint64_t val = x % s;
                if (val != 0) {
                    val--;

                    int j = js[val];

                    auto [TBS, n, t0, t1, ost_len] = requests[j];

                    for (int k = 0; k < K; k++) {
                        double weight = 1;

                        ASSERT(total_g[j] < TBS, "failed");

                        weight *= 1.0 / log(1 + TBS);

                        //weight = 30 - log(1 + TBS);//+ log(1 + total_g[j]);

                        //cout << weight << endl;
                        ASSERT(weight >= 0 && !is_spoiled(weight), "invalid weight");

                        set_of_weights.emplace_back(weight, n, k, r);
                    }
                }
            }

            fix_sum();

            set_power(t, set_of_weights);

            // максимально сильно уменьшим затраченную силу
            auto update_power_to_min = [&]() { // NOLINT
                bool run = true;
                while (run) {
                    run = false;
                    for (int j: js) {
                        auto [TBS, n, t0, t1, ost_len] = requests[j];
                        double add_g = get_g(t, n);
                        if (add_g + total_g[j] >= TBS + 1e-1) {
                            run = true;
                            //cout << "update: " << TBS << ' ' << add_g + total_g[j] << "->";

                            auto calc_add_g = [&](double power_factor) {
                                vector<vector<double>> save_p(K, vector<double>(R));
                                for (int k = 0; k < K; k++) {
                                    for (int r = 0; r < R; r++) {
                                        save_p[k][r] = p[t][n][k][r];
                                        p[t][n][k][r] *= power_factor;
                                    }
                                }
                                double add_g = get_g(t, n);

                                for (int k = 0; k < K; k++) {
                                    for (int r = 0; r < R; r++) {
                                        p[t][n][k][r] = save_p[k][r];
                                    }
                                }

                                return add_g;
                            };

                            double tl = 0, tr = 1;
                            while (tl < tr - 1e-6) {
                                double tm = (tl + tr) / 2;
                                if (calc_add_g(tm) + total_g[j] >= TBS) {
                                    tr = tm;
                                } else {
                                    tl = tm;
                                }
                            }
                            double good_factor = tr;
                            for (int k = 0; k < K; k++) {
                                for (int r = 0; r < R; r++) {
                                    p[t][n][k][r] *= good_factor;
                                }
                            }
                            //cout << get_g(t, n) << endl;
                        }
                    }
                }
            };

            update_power_to_min();

            // добавить силы нуждающимся
            set<int> used;
            while(true){
                int best_j = -1;
                double best_f = -1e300;
                for (int j: js) {
                    auto [TBS, n, t0, t1, ost_len] = requests[j];
                    if (!used.contains(j) && get_g(t, n) != 0 && get_g(t, n) + total_g[j] < TBS) {
                        double cur_f = get_g(t, n);
                        if(best_j == -1 || best_f < cur_f){
                            best_j = j;
                            best_f = cur_f;
                        }
                    }
                }

                if(best_j == -1){
                    break;
                }

                int j = best_j;
                used.insert(j);
                auto [TBS, n, t0, t1, ost_len] = requests[j];

                //cout << "upkek: " << TBS << ' ' << get_g(t, n) + total_g[j] << ' ';

                auto calc_add_g = [&](double power_factor) {
                    vector<vector<double>> save_p(K, vector<double>(R));
                    for (int k = 0; k < K; k++) {
                        for (int r = 0; r < R; r++) {
                            save_p[k][r] = p[t][n][k][r];
                            p[t][n][k][r] *= power_factor;
                        }
                    }
                    double add_g = get_g(t, n);

                    for (int k = 0; k < K; k++) {
                        for (int r = 0; r < R; r++) {
                            p[t][n][k][r] = save_p[k][r];
                        }
                    }

                    return add_g;
                };

                auto verify = [&](double power_factor) { // NOLINT
                    vector<double> sum(K);
                    for (int m = 0; m < N; m++) {
                        for (int k = 0; k < K; k++) {
                            for (int r = 0; r < R; r++) {
                                if (m != n) {
                                    sum[k] += p[t][m][k][r];
                                } else {
                                    sum[k] += power_factor * p[t][m][k][r];
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
                                    sum += power_factor * p[t][m][k][r];
                                }
                            }
                            if (sum > 4) {
                                return false;
                            }
                        }
                    }
                    return true;
                };

                double tl = 0, tr = 1e9;
                while (tl < tr - 1e-6) {
                    double tm = (tl + tr) / 2;
                    if (calc_add_g(tm) + total_g[j] >= TBS) {
                        tr = tm;
                    } else {
                        tl = tm;
                    }
                }

                double good_factor = tr;
                //cout << good_factor << ' ' << verify(good_factor) << ' ' << calc_add_g(good_factor) + total_g[j] << endl;
                if(verify(good_factor)) {
                    for (int k = 0; k < K; k++) {
                        for (int r = 0; r < R; r++) {
                            p[t][n][k][r] *= good_factor;
                        }
                    }
                }

            }

        };

        double best_f = -1e300;
        uint64_t best_msk = 0;
        for (uint64_t msk = 0; msk < MAX; msk++) {
            set_power_for_msk(msk);

            double cur_f = 0;
            for (int j: js) {
                auto [TBS, n, t0, t1, ost_len] = requests[j];
                double add_g = get_g(t, n);
                //cur_f += add_g;
                if (add_g + total_g[j] >= TBS) {
                    cur_f += 1e5;
                }
            }

            if (cur_f > best_f) {
                //cout << cur_f << "->";
                best_f = cur_f;
                best_msk = msk;
            }
        }
        set_power_for_msk(best_msk);
        cout << "best_f: " << best_f << " best_msk: " << best_msk << ' ' << get_score() << endl;
    }

/*
C:\Windows\system32\wsl.exe --distribution Ubuntu --exec /bin/bash -c "cd '/root/Deterministic Scheduling for Extended Reality over 5G and Beyond/Deterministic-Scheduling-for-Extended-Reality-over-5G-and-Beyond' && '/root/Deterministic Scheduling for Extended Reality over 5G and Beyond/Deterministic-Scheduling-for-Extended-Reality-over-5G-and-Beyond/solution'"
TEST CASE==============
50 243 243 best_f: 200000 best_msk: 5 -2.92304e-07
98 1024 1024 best_f: 300000 best_msk: 27 2
72 3125 3125 best_f: 400000 best_msk: 819 5
77 3125 3125 best_f: 400000 best_msk: 969 8.99999
2 7776 7776 best_f: 500000 best_msk: 1865 13
5 7776 7776 best_f: 400000 best_msk: 317 18
9 7776 7776 best_f: 400000 best_msk: 353 22
39 7776 7776 best_f: 500000 best_msk: 1865 26
67 7776 7776 best_f: 500000 best_msk: 1865 31
82 7776 7776 best_f: 500000 best_msk: 1865 36
90 7776 7776 best_f: 400000 best_msk: 311 41
4 16807 16807 best_f: 500000 best_msk: 3275 45
25 16807 16807 best_f: 500000 best_msk: 6068 50
31 16807 16807 best_f: 500000 best_msk: 3267 55
33 16807 16807 best_f: 500000 best_msk: 6068 60
37 16807 16807 best_f: 400000 best_msk: 475 65
44 16807 16807 best_f: 500000 best_msk: 3267 69
55 16807 16807 best_f: 500000 best_msk: 3267 74
70 16807 16807 best_f: 500000 best_msk: 3275 79
83 16807 16807 best_f: 500000 best_msk: 3267 84
86 16807 16807 best_f: 500000 best_msk: 3267 89
89 16807 16807 best_f: 400000 best_msk: 2867 94
95 16807 16807 best_f: 500000 best_msk: 6068 98
6 32768 32768 best_f: 500000 best_msk: 5349 103
19 32768 32768 best_f: 500000 best_msk: 5349 108
27 32768 32768 best_f: 500000 best_msk: 5350 113
35 32768 32768 best_f: 500000 best_msk: 5367 118
40 32768 32768 best_f: 500000 best_msk: 5349 123
59 32768 32768 best_f: 500000 best_msk: 5422 128
60 32768 32768 best_f: 500000 best_msk: 5349 133
62 32768 32768 best_f: 500000 best_msk: 5349 138
64 32768 32768 best_f: 500000 best_msk: 5934 143
80 32768 32768 best_f: 500000 best_msk: 5358 148
84 32768 32768 best_f: 500000 best_msk: 10030 153
92 32768 32768 best_f: 500000 best_msk: 5350 158
97 32768 32768 best_f: 500000 best_msk: 5350 163
11 59049 59049 best_f: 500000 best_msk: 8304 168
12 59049 59049 best_f: 500000 best_msk: 8303 173
14 59049 59049 best_f: 500000 best_msk: 9124 178
16 59049 59049 best_f: 500000 best_msk: 8304 183
17 59049 59049 best_f: 500000 best_msk: 8303 188
18 59049 59049 best_f: 500000 best_msk: 8305 193
21 59049 59049 best_f: 500000 best_msk: 8303 198
24 59049 59049 best_f: 500000 best_msk: 8311 203
28 59049 59049 best_f: 500000 best_msk: 8303 208
42 59049 59049 best_f: 500000 best_msk: 8313 213
45 59049 59049 best_f: 500000 best_msk: 9123 218
47 59049 59049 best_f: 500000 best_msk: 8303 223
54 59049 59049 best_f: 500000 best_msk: 8303 228
63 59049 59049 best_f: 500000 best_msk: 8304 233
69 59049 59049 best_f: 500000 best_msk: 8303 238
73 59049 59049 best_f: 500000 best_msk: 8303 243
79 59049 59049 best_f: 500000 best_msk: 8303 248
94 59049 59049 best_f: 500000 best_msk: 8303 253
13 100000 100000 best_f: 500000 best_msk: 12345 258
15 100000 100000 best_f: 500000 best_msk: 12345 263
22 100000 100000 best_f: 500000 best_msk: 12345 268
23 100000 100000 best_f: 500000 best_msk: 12345 273
30 100000 100000 best_f: 500000 best_msk: 13467 278
32 100000 100000 best_f: 500000 best_msk: 24569 283
51 100000 100000 best_f: 500000 best_msk: 12345 288
53 100000 100000 best_f: 500000 best_msk: 12345 293
56 100000 100000 best_f: 500000 best_msk: 12345 298
57 100000 100000 best_f: 500000 best_msk: 12345 303
65 100000 100000 best_f: 500000 best_msk: 13456 308
75 100000 100000 best_f: 500000 best_msk: 23169 313
76 100000 100000 best_f: 500000 best_msk: 12345 318
78 100000 100000 best_f: 500000 best_msk: 12346 323
8 161051 161051 best_f: 500000 best_msk: 17715 328
20 161051 161051 best_f: 500000 best_msk: 17715 333
26 161051 161051 best_f: 500000 best_msk: 17715 338
34 161051 161051 best_f: 500000 best_msk: 17715 343
36 161051 161051 best_f: 500000 best_msk: 17715 348
43 161051 161051 best_f: 500000 best_msk: 17715 353
46 161051 161051 best_f: 500000 best_msk: 17715 358
48 161051 161051 best_f: 500000 best_msk: 17715 363
58 161051 161051 best_f: 500000 best_msk: 17715 368
66 161051 161051 best_f: 500000 best_msk: 17715 373
87 161051 161051 best_f: 500000 best_msk: 17715 378
93 161051 161051 best_f: 500000 best_msk: 17848 383
1 248832 248832 best_f: 500000 best_msk: 24677 388
7 248832 248832 best_f: 500000 best_msk: 24678 393
38 248832 248832 best_f: 500000 best_msk: 24677 398
49 248832 248832 best_f: 500000 best_msk: 24704 403
52 248832 248832 best_f: 500000 best_msk: 47298 408
61 248832 248832 best_f: 500000 best_msk: 24677 413
68 248832 248832 best_f: 500000 best_msk: 24690 418
85 248832 248832 best_f: 500000 best_msk: 24729 423
91 248832 248832 best_f: 500000 best_msk: 24677 428
10 371293 371293 best_f: 500000 best_msk: 33519 433
29 371293 371293 best_f: 500000 best_msk: 33702 438
41 371293 371293 best_f: 500000 best_msk: 33519 443
71 371293 371293 best_f: 500000 best_msk: 33519 448
74 371293 371293 best_f: 500000 best_msk: 35900 453
88 371293 371293 best_f: 500000 best_msk: 33519 458
96 537824 537824 best_f: 500000 best_msk: 44553 463
0 759375 759375 best_f: 500000 best_msk: 58115 468
3 759375 759375 best_f: 500000 best_msk: 58115 473
81 1.41986e+06 1419857 best_f: 500000 best_msk: 99507 478
483/829
*/

    void set_zero_power(int j, vector<vector<int>> &js) {
        auto [TBS, n, t0, t1, ost_len] = requests[j];
        for (int t = t0; t <= t1; t++) {
            for (int k = 0; k < K; k++) {
                for (int r = 0; r < R; r++) {
                    p[t][n][k][r] = 0;
                }
            }
            for (int j2: js[t]) {
                int m = requests[j2].n;
                total_g[j2] -= add_g[t][m];
                add_g[t][m] = get_g(t, m);
                total_g[j2] += add_g[t][m];
            }
        }
    }

    void solve() {
        p.assign(T, vector(N, vector(K, vector<double>(R))));
        total_g.assign(J, 0);
        add_g.assign(T, vector<double>(N));
        count_of_set_r_for_request.assign(J, 0);
        for (int j = 0; j < J; j++) {
            requests[j].ost_len = requests[j].t1 - requests[j].t0 + 1;
        }

        vector<vector<int>> js(T);
        for (int j = 0; j < J; j++) {
            if (used_request[j]) {
                auto [TBS, n, t0, t1, ost_len] = requests[j];
                for (int t = t0; t <= t1; t++) {
                    js[t].push_back(j);
                }
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
            {
                // TODO: выбирать при помощи суммы TBS и уже набранной total_g
                // а не при помощи размера, это треш

                for (int t = 0; t < T; t++) {
                    if (!js[t].empty()) {
                        if (best_time == -1 || js[best_time].size() > js[t].size()) {
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

            // обновить g
            for (int j: js[best_time]) {
                verify_ost_len(j);

                auto [TBS, n, t0, t1, ost_len] = requests[j];
                add_g[best_time][n] = get_g(best_time, n);
                total_g[j] += add_g[best_time][n];
            }

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

            // TODO: если мы не смогли отправить сообщение
            // то посмотрим на таких. возьмем того, у кого прям вообще не получилось
            // этот чел занимал r, которую мы могли бы дать другим и улучшить их score
            // давайте так и сделаем

            /*vector<tuple<double, int>> need_set_zero;
            for (int j: js[best_time]) {
                if (total_g[j] < requests[j].TBS && requests[j].ost_len == 0) {
                    need_set_zero.emplace_back(total_g[j], j);
                }
            }
            sort(need_set_zero.begin(), need_set_zero.end(), greater<>());
            for (auto [weight, j]: need_set_zero) {
                if (total_g[j] < requests[j].TBS) {

                }
            }

            for (int j: js[best_time]) {
                auto [TBS, n, t0, t1, ost_len] = requests[j];
            }*/

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
                                accum_prod *= exp_d[n][m][k][r];
                            }
                        }
                    }
                }
            }

            sum += count * log2(1 + pow(accum_prod, 1.0 / count));
        }
        ASSERT(sum >= 0 && !is_spoiled(sum), "invalid g");
        return 192 * sum;
    }

    double get_g(int t, int n) {
        //return correct_get_g(t, n);
/*
TEST CASE==============
0.999996/2
TEST CASE==============
144.993/150
TEST CASE==============
475.998/829
TEST CASE==============
183.995/184
*/
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

            sum += count * log2(1 + pow(accum_prod, 1.0 / count));
            //cout << "kek: " << accum_prod << ' ' << count << endl;
        }
        ASSERT(sum >= 0 && !is_spoiled(sum), "invalid g");
        //ASSERT(correct_get_g(t, n) == 192 * sum, "failed calc");
        return 192 * sum;
    }

    bool remove_bad_request() {
        auto f = [&](int j) {
            return -count_of_set_r_for_request[j] * 10000 + (requests[j].TBS - total_g[j]);
        };
        int best_j = -1;

        for (int j = 0; j < J; j++) {
            if (used_request[j] && total_g[j] < requests[j].TBS) {
                // bad request
                if (best_j == -1 || f(j) < f(best_j)) {
                    best_j = j;
                }
            }
        }
        if (best_j == -1) {
            return false;
        }
        // remove this request [best_j]
        used_request[best_j] = false;
        return true;
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
#endif
        solution.used_request.assign(solution.J, true);

        //Solution solution2 = solution;

        //solution2.use_build_weight_version = Solution::ONCE_IN_R;
        solution.use_build_weight_version = Solution::ONCE_IN_R;

        solution.solve();
        /*solution2.solve();

        if (solution.get_score() < solution2.get_score()) {
            solution = solution2;
        }*/

        /*while (true) {
            solution.solve();
    #ifndef FAST_STREAM
            cout << solution.get_score() << '/' << solution.J << '\n';
    #endif
            if (!solution.remove_bad_request()) {
                break;
            }
        }*/

#ifndef FAST_STREAM
        cout << solution.get_score() << '/' << solution.J << '\n';
#endif

#ifdef FAST_STREAM
        solution.print();
#endif

#ifndef FAST_STREAM
    }
#endif
}
