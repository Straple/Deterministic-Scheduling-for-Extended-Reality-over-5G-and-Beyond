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

    void set_power(int t, const vector<tuple<double, int, int, int>> &set_of_weights) {
#ifdef DEBUG_MODE
        vector<set<int>> kek(R);
#endif
        for (auto [weight, n, k, r]: set_of_weights) {
            p[t][n][k][r] = weight * 4;
#ifdef DEBUG_MODE
            kek[r].insert(n);
#endif
        }
#ifdef DEBUG_MODE
        for (int r = 0; r < R; r++) {
            ASSERT(kek[r].size() <= 1, "why many requests in one r?");
        }
#endif

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

    vector<tuple<double, int, int, int>> build_weights_of_power(int t, const vector<int> &js) {
        // (weight, n, k, r)
        vector<tuple<double, int, int, int>> set_of_weights;
        {
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

                    weight += 1e3;
                    requests_weights[r].emplace_back(weight, j);
                }
            }

            vector<tuple<double, int>> kek(R);
            for (int r = 0; r < R; r++) {
                sort(requests_weights[r].begin(), requests_weights[r].end(), greater<>());

                auto [request_weight, j] = requests_weights[r][0];

                kek[r] = {request_weight, r};
            }
            sort(kek.begin(), kek.end(), greater<>());
            for (auto [_, r]: kek) {
                sort(requests_weights[r].begin(), requests_weights[r].end(), greater<>());

                auto [__, j] = requests_weights[r][0];
                // relax other request_weights[r]
                {
                    for (int r2 = 0; r2 < R; r2++) {
                        if (r2 != r) {
                            for (auto &[request_weight, j2]: requests_weights[r2]) {
                                if (j == j2) {
                                    request_weight /= 1e50;
                                }
                            }
                        }
                    }
                }

                count_of_set_r_for_request[j]++;
                // cout << count_of_set_r_for_request[j] << '\n';

                auto [TBS, n, t0, t1, ost_len] = requests[j];

                for (int k = 0; k < K; k++) {
                    double weight = 1;

                    ASSERT(total_g[j] < TBS, "failed");

                    ASSERT(weight >= 0 && !is_spoiled(weight), "invalid weight");

                    set_of_weights.emplace_back(weight, n, k, r);
                }
            }
        }

        auto calc_sum = [&]() {
            vector<vector<double>> sum_weight(K, vector<double>(R));
            for (auto [weight, n, k, r]: set_of_weights) {
                ASSERT(weight >= 0, "invalid weight");
                sum_weight[k][r] += weight;
            }
            return sum_weight;
        };

        auto fix_sum = [&]() {
            auto sum_weight = calc_sum();

            for (auto &[weight, n, k, r]: set_of_weights) {
                //ASSERT(sum_weight[k][r] == 0 || sum_weight[k][r] > 1e-9, "very small sum_weight");
                if (sum_weight[k][r] != 0) {
                    weight = weight / sum_weight[k][r];
                } else {
                    weight = 0;
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
                for (int r = 0; r < R; r++) {
                    ASSERT(sum_weight[k][r] == 0 || abs(sum_weight[k][r] - 1) < 1e-9, "invalid sum_weight");
                }
            }
        };

        fix_sum();
        //threshold();
        //fix_sum();

        /*{
            // а давайте посмотрим на эти веса
            // возможно кого-то стоит понизить в весе, если мы сейчас ставим этих ребят
            // возможно кого-то стоит повысить

            vector<tuple<double, int, int, int>> new_set;

            for (auto [weight, n, k, r]: set_of_weights) {
                double X = weight;
                for (auto [weight2, m, k2, r2]: set_of_weights) {
                    if (r == r2 && n != m && k != k2) {

                        double d = s0[t][n][k2][r] * (4 * weight2) / exp_d[n][m][k2][r];
                        if (d <= 0) {
                            cout << "FATAL: " << ' ' << s0[t][n][k2][r] << ' ' << (4 * weight2) << ' '
                                 << exp_d[n][m][k2][r] << endl;
                            exit(1);
                        }
                        d = sqrt(d);
                        X /= d;
                    }
                }
                new_set.emplace_back(X, n, k, r);
            }
            set_of_weights = new_set;
        }

        fix_sum();
        threshold();
        fix_sum();*/


#ifdef DEBUG_MODE
        verify();
#endif

        return set_of_weights;
    }

    void set_nice_power(int t, const vector<int> &js) {
        // наиболее оптимально расставить силу так
        // что это значит? наверное мы хотим как можно больший прирост g
        // а также чтобы доотправлять сообщения

        set_power(t, build_weights_of_power(t, js));
    }

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
//                    cout << p[t][n][k][r] << ' ';
                    count++;
                    accum_prod *= p[t][n][k][r];
                    accum_prod *= s0[t][n][k][r];
                }
            }
            //          cout << endl;

            sum += count * log2(1 + pow(accum_prod, 1.0 / count));
            //cout << "kek: " << accum_prod << ' ' << count << endl;
        }
        ASSERT(sum >= 0 && !is_spoiled(sum), "invalid g");
        ASSERT(correct_get_g(t, n) == 192 * sum, "failed calc");
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
        solution.solve();
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
