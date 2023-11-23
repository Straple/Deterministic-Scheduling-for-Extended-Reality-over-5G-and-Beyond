#include <bits/stdc++.h>

using namespace std;

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

bool is_spoiled(double num) {
    return std::isnan(num) || std::isinf(num);
}

#include <chrono>

using namespace std::chrono;

//#define FAST_STREAM

int main() {
    for (int ABA = 2; ABA <= 2; ABA++) {
#ifndef FAST_STREAM
        std::ifstream cin("input.txt");
        if (ABA == 1) {
            cin = std::ifstream("6");
        } else if (ABA == 2) {
            cin = std::ifstream("21");
        } else if (ABA == 3) {
            cin = std::ifstream("51");
        } else {
            exit(1);
        }

#endif
        std::ios::sync_with_stdio(false);
        std::cin.tie(0);
        std::cout.tie(0);

        auto time_start = steady_clock::now();

        // ===========
        // ==READING==
        // ===========

        int N;
        int K;
        int T;
        int R;
#ifdef FAST_STREAM
        N = readInt();
        K = readInt();
        T = readInt();
        R = readInt();
#else
        cin >> N >> K >> T >> R;
#endif

        // s0[t][n][k][r]
        vector<vector<vector<vector<double>>>> s0(T, vector(N, vector(K, vector<double>(R))));
        for (int t = 0; t < T; t++) {
            for (int k = 0; k < K; k++) {
                for (int r = 0; r < R; r++) {
                    for (int n = 0; n < N; n++) {
#ifdef FAST_STREAM
                        s0[t][n][k][r] = readDouble();
#else
                        cin >> s0[t][n][k][r];
#endif
                    }
                }
            }
        }

        vector<vector<vector<vector<double>>>> d(N, vector(N, vector(K, vector<double>(R))));

        // exp_d[n][m][k][r]
        vector<vector<vector<vector<double>>>> exp_d(N, vector(N, vector(K, vector<double>(R))));

        // exp_d_pow[n][m][k][r]
        vector<vector<vector<vector<double>>>> exp_d_pow(N, vector(N, vector(K, vector<double>(R))));
        for (int k = 0; k < K; k++) {
            for (int r = 0; r < R; r++) {
                for (int m = 0; m < N; m++) {
                    for (int n = 0; n < N; n++) {
#ifdef FAST_STREAM
                        d[n][m][k][r] = readDouble();
#else
                        cin >> d[n][m][k][r];
#endif
                        exp_d[n][m][k][r] = exp(d[n][m][k][r]);

                        exp_d_pow[n][m][k][r] = pow(exp_d[n][m][k][r], 3);
                    }
                }
            }
        }

        struct message_window {
            int TBS;
            int user_id;
            int t0;
            int t1;
        };

        int J;
#ifdef FAST_STREAM
        J = readInt();
#else
        cin >> J;
#endif

        vector<message_window> Queries(J);
        for (int i = 0; i < J; i++) {
            int j;
            int t0, td;

#ifdef FAST_STREAM
            j = readInt();
            Queries[j].TBS = readInt();
            Queries[j].user_id = readInt();
            t0 = readInt();
            td = readInt();
#else
            cin >> j;
            cin >> Queries[j].TBS;
            cin >> Queries[j].user_id;
            cin >> t0 >> td;
#endif

            int t1 = t0 + td - 1;

            Queries[j].t0 = t0;
            Queries[j].t1 = t1;
        }

        // p[t][n][k][r]
        vector<vector<vector<vector<double>>>> p(T, vector(N, vector(K, vector<double>(R))));

        // ============
        // ==SOLUTION==
        // ============

        // dp_s0_p_d[t][n][k][r]
        vector<vector<vector<vector<double>>>> dp_s0_p_d(T, vector(N, vector(K, vector<double>(R))));

        auto calc_g = [&](int t, int n) {  // NOLINT
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
                        dp_s0_p_d[t][n][k][r] = 1 + dp_sum_2[r] - dp_sum[k][r];
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
                        accum_prod /= dp_s0_p_d[t][n][k][r];

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
            if (sum < 0 || is_spoiled(sum)) {
                while (true) {}
            }
            return 192 * sum;
        };

        vector<vector<int>> event_add(T), event_remove(T);
        for (int j = 0; j < J; j++) {
            event_add[Queries[j].t0].push_back(j);
            event_remove[Queries[j].t1].push_back(j);
        }

        struct data {
            int j;  // номер окна
            double g;
        };

        auto solution = [&](const vector<double> &weight_factor) {  // NOLINT
            for (int t = 0; t < T; t++) {
                for (int n = 0; n < N; n++) {
                    for (int k = 0; k < K; k++) {
                        for (int r = 0; r < R; r++) {
                            p[t][n][k][r] = 0;
                        }
                    }
                }
            }

            vector<double> total_g(J);

            vector<vector<double>> add_g(T, vector<double>(N));

            map<int, data> users;

            auto do_smaller = [&](int t, int j, double received_TBS) {  // NOLINT
                auto [TBS, n, t0, _] = Queries[j];
                // максимально сильно уменьшим применяемую силу, но так, чтобы все еще получали TBS
                auto calc_TBS = [&](double power_factor) {
                    vector<vector<double>> save_p(K, vector<double>(R));
                    for (int k = 0; k < K; k++) {
                        for (int r = 0; r < R; r++) {
                            save_p[k][r] = p[t][n][k][r];
                            p[t][n][k][r] *= power_factor;
                        }
                    }
                    double cur_TBS = calc_g(t, n);

                    for (int k = 0; k < K; k++) {
                        for (int r = 0; r < R; r++) {
                            p[t][n][k][r] = save_p[k][r];
                        }
                    }

                    return cur_TBS;
                };

                double X = calc_TBS(1);
                received_TBS -= X;
                if (received_TBS < 0) {
                    exit(1);
                }

                double tl = 0, tr = 1;
                while (tl < tr - 1e-6) {
                    double tm = (tl + tr) / 2;
                    if (received_TBS + calc_TBS(tm) >= TBS) {
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

                //total_g[j] = received_TBS + calc_TBS(1);

                // cout << "lol: " << total_g[j] << ' ' << TBS << '\n';
                if (total_g[j] < TBS) {
                    exit(1);
                }

                // TODO: update other users
                // почему-то это только ухудшает
                vector<int> need_delete;
                for (auto &[m, data2]: users) {
                    bool was_bad = data2.g < Queries[data2.j].TBS;

                    data2.g -= add_g[t][m];
                    add_g[t][m] = calc_g(t, m);
                    data2.g += add_g[t][m];

                    // стало лучше
                    if (was_bad && data2.g >= Queries[data2.j].TBS) {
                        need_delete.push_back(m);
                    }
                }
                for (int n: need_delete) {
                    total_g[users[n].j] = users[n].g;
                    users.erase(n);
                }
            };

            for (int t = 0; t < T; t++) {
                // add
                for (int j: event_add[t]) {
                    int n = Queries[j].user_id;
                    users[n] = {j, 0};
                }

                // веса нужны от [0, 1]
                // сумма весов при заданных k, r должна быть равна 1
                vector<tuple<double, int, int, int>> kek;
                {
                    for (auto [n, data]: users) {
                        auto [TBS, user_id, t0, t1] = Queries[data.j];

                        for (int k = 0; k < K; k++) {
                            for (int r = 0; r < R; r++) {
                                double weight = 1;
                                weight *= exp(pow((t1 - t0 + 1) * 1.0 / (t1 - t + 1), 2.1) - pow(TBS - data.g, 0.5));

                                //weight *= s0[t][n][k][r];

                                for (auto [m, data2]: users) {
                                    if (n != m) {
                                        weight *= exp_d_pow[n][m][k][r];
                                    }
                                }

                                /*weight += 100;
                                for (int k2 = 0; k2 < K; k2++) {
                                    if (k != k2) {
                                        weight -= pow(s0[t][n][k2][r], 0.5);
                                    }
                                }*/

                                kek.emplace_back(weight, n, k, r);

                                if (weight < 0 || is_spoiled(weight)) {
                                    exit(1);
                                }
                            }
                        }
                    }

                    // normalize weights

                    // sum
                    {
                        vector<vector<double>> sum_weight(K, vector<double>(R));
                        for (auto [weight, n, k, r]: kek) {
                            sum_weight[k][r] += weight;
                        }
                        for (auto &[weight, n, k, r]: kek) {
                            if (sum_weight[k][r] != 0) {
                                weight = weight / sum_weight[k][r];
                            } else {
                                weight = 0;
                            }
                        }
                    }

                    for (auto &[weight, n, k, r]: kek) {
                        weight *= weight_factor[users[n].j];
                    }

                    // sum
                    {
                        vector<vector<double>> sum_weight(K, vector<double>(R));
                        for (auto [weight, n, k, r]: kek) {
                            sum_weight[k][r] += weight;
                        }
                        for (auto &[weight, n, k, r]: kek) {
                            if (sum_weight[k][r] != 0) {
                                weight = weight / sum_weight[k][r];
                            } else {
                                weight = 0;
                            }
                        }
                    }

                    // cout << "weights: ";
                    // for (auto &[weight, n]: kek) {
                    //     cout << weight << ' ';
                    // }
                    // cout << '\n';

                    sort(kek.begin(), kek.end());
                    reverse(kek.begin(), kek.end());
                    while (kek.size() > 1 && get<0>(kek.back()) < 0.02) {
                        kek.pop_back();
                    }
                    reverse(kek.begin(), kek.end());

                    // sum
                    {
                        vector<vector<double>> sum_weight(K, vector<double>(R));
                        for (auto [weight, n, k, r]: kek) {
                            sum_weight[k][r] += weight;
                        }
                        for (auto &[weight, n, k, r]: kek) {
                            if (sum_weight[k][r] != 0) {
                                weight = weight / sum_weight[k][r];
                            } else {
                                weight = 0;
                            }
                        }
                    }

                    sort(kek.begin(), kek.end());

                    // verify
                    {
                        vector<vector<double>> sum_weight(K, vector<double>(R));
                        for (auto [weight, n, k, r]: kek) {
                            sum_weight[k][r] += weight;
                            if (weight < 0 || weight > 1 || is_spoiled(weight)) {
                                exit(1);
                            }
                        }
                        for (int k = 0; k < K; k++) {
                            for (int r = 0; r < R; r++) {
                                if (sum_weight[k][r] != 0 && abs(sum_weight[k][r] - 1) > 1e-9) {
                                    exit(1);
                                }
                            }
                        }
                    }
                }

                // set power in time t
                {
                    for (auto [weight, n, k, r]: kek) {
                        p[t][n][k][r] = weight * 4;
                    }

                    vector<double> sum(K);
                    for (auto [weight, n, k, r]: kek) {
                        sum[k] += p[t][n][k][r];
                    }
                    for (auto [weight, n, k, r]: kek) {
                        if (sum[k] > 1e-9) {
                            p[t][n][k][r] *= min(1.0, R / sum[k]);
                        }
                    }
                }

                // update g

                vector<pair<double, int>> need_delete;
                for (auto &[n, data]: users) {
                    add_g[t][n] = calc_g(t, n);
                    data.g += add_g[t][n];

                    // мы уже отправили все
                    if (data.g >= Queries[data.j].TBS) {
                        need_delete.emplace_back(data.g - Queries[data.j].TBS, n);
                    }
                }
                sort(need_delete.begin(), need_delete.end(), greater<>());
                for (auto [weight, n]: need_delete) {
                    int j = users[n].j;
                    double g = users[n].g;
                    total_g[j] = users[n].g;
                    users.erase(n);
                    do_smaller(t, j, g);
                }

                // trivial remove
                /*for (int j: event_remove[t]) {
                    int n = Queries[j].user_id;
                    if (users.contains(n)) {
                        total_g[j] = users[n].g;
                        users.erase(n);
                    }
                }*/

                // smart remove
                // проставляет силу 0, если мы не смогли отправить
                vector<pair<double, int>> lol;
                for (int j: event_remove[t]) {
                    int n = Queries[j].user_id;
                    if (users.contains(n)) {
                        lol.emplace_back(Queries[j].TBS - users[n].g, j);
                    }
                }
                sort(lol.begin(), lol.end(), greater<>());

                for (auto [_, j]: lol) {
                    int n = Queries[j].user_id;
                    if (!users.contains(n)) {
                        continue;
                    }
                    total_g[j] = users[n].g;
                    users.erase(n);

                    // TODO: если мы не смогли набрать TBS, то нам не нужно было тратить туда силу
                    for (int t = Queries[j].t0; t <= Queries[j].t1; t++) {
                        for (int k = 0; k < K; k++) {
                            for (int r = 0; r < R; r++) {
                                p[t][n][k][r] = 0;
                            }
                        }
                    }

                    // TODO: попробовать перераспределить силу, которую я убрал
                    // возможно лучшим способом будет сказать, что weight_factor[j] = 0
                    // а под конец пересчитать так еще раз

                    vector<pair<double, int>> need_delete;
                    for (auto &[m, data2]: users) {
                        for (int time = max(Queries[data2.j].t0, Queries[j].t0); time <= t; time++) {
                            data2.g -= add_g[time][m];
                            add_g[time][m] = calc_g(time, m);
                            data2.g += add_g[time][m];
                        }
                        if (data2.g >= Queries[data2.j].TBS) {
                            need_delete.emplace_back(data2.g - Queries[data2.j].TBS, m);
                        }
                    }
                    sort(need_delete.begin(), need_delete.end(), greater<>());
                    for (auto [weight, n]: need_delete) {
                        double g = users[n].g;
                        int j = users[n].j;
                        total_g[j] = users[n].g;
                        users.erase(n);
                        do_smaller(t, j, g);
                    }
                }
            }
            return total_g;
        };

        auto new_solution = [&]() { // NOLINT
            // давайте не будем заниматься сканлайном, а смотреть на всю проблему сверху
            // возможно так будет больше возможностей что-то улучшить

            vector<vector<int>> messagease_in_time(T);
            for (int j = 0; j < J; j++) {
                auto [TBS, n, t0, t1] = Queries[j];
                for (int t = t0; t <= t1; t++) {
                    messagease_in_time[t].push_back(j);
                }
            }

            vector<double> total_g(J);

            while (true) {
                // давайте найдем времена, в которых только один отрезок
                // тогда нам наиболее выгодно туда влить силу, так как другие не помешают

                int best_time = -1;
                double best_f = 0;
                auto calc_f = [&](int t, int j) -> double {
                    return 1000 * total_g[j] / Queries[j].TBS - Queries[j].TBS;
                };
                for (int t = 0; t < T; t++) {
                    if (!messagease_in_time[t].empty()) {
                        double sum = 0;
                        for (int j: messagease_in_time[t]) {
                            sum += calc_f(t, j);
                        }
                        if (best_time == -1 || sum > best_f) {
                            best_f = sum;
                            best_time = t;
                        }
                    }
                }
                if (best_time == -1) {
                    break;
                }

                {
                    // веса нужны от [0, 1]
                    // сумма весов при заданных k, r должна быть равна 1
                    vector<tuple<double, int, int, int>> kek;
                    {
                        for (int j: messagease_in_time[best_time]) {
                            auto [TBS, n, t0, t1] = Queries[j];

                            for (int k = 0; k < K; k++) {
                                for (int r = 0; r < R; r++) {
                                    double weight = 1;
                                    int ost_len = 0;
                                    for (int t = t0; t <= t1; t++) {
                                        ost_len += !messagease_in_time[t].empty();
                                    }
                                    //weight *= exp(pow((t1 - t0 + 1) * 1.0 / (t1 - t + 1), 2.1) - pow(TBS - data.g, 0.5));

                                    // weight *= exp(pow(ost_len / (t1 - t0 + 1), 2) - pow(TBS - total_g[j], 0.5));
                                    weight *= exp(pow(ost_len, 2) - pow(TBS - total_g[j], 0.4));

                                    for (int j2: messagease_in_time[best_time]) {
                                        int m = Queries[j2].user_id;
                                        if (n != m) {
                                            weight *= exp_d_pow[n][m][k][r];
                                        }
                                    }

                                    kek.emplace_back(weight, n, k, r);

                                    if (weight < 0 || is_spoiled(weight)) {
                                        cout << weight << endl;
                                        exit(1);
                                    }
                                }
                            }
                        }

                        // normalize weights

                        // sum
                        {
                            vector<vector<double>> sum_weight(K, vector<double>(R));
                            for (auto [weight, n, k, r]: kek) {
                                sum_weight[k][r] += weight;
                            }
                            for (auto &[weight, n, k, r]: kek) {
                                if (sum_weight[k][r] != 0) {
                                    weight = weight / sum_weight[k][r];
                                } else {
                                    weight = 0;
                                }
                            }
                        }

                        // sum
                        {
                            vector<vector<double>> sum_weight(K, vector<double>(R));
                            for (auto [weight, n, k, r]: kek) {
                                sum_weight[k][r] += weight;
                            }
                            for (auto &[weight, n, k, r]: kek) {
                                if (sum_weight[k][r] != 0) {
                                    weight = weight / sum_weight[k][r];
                                } else {
                                    weight = 0;
                                }
                            }
                        }

                        // cout << "weights: ";
                        // for (auto &[weight, n]: kek) {
                        //     cout << weight << ' ';
                        // }
                        // cout << '\n';

                        sort(kek.begin(), kek.end());
                        reverse(kek.begin(), kek.end());
                        while (kek.size() > 1 && get<0>(kek.back()) < 0.02) {
                            kek.pop_back();
                        }
                        reverse(kek.begin(), kek.end());

                        // sum
                        {
                            vector<vector<double>> sum_weight(K, vector<double>(R));
                            for (auto [weight, n, k, r]: kek) {
                                sum_weight[k][r] += weight;
                            }
                            for (auto &[weight, n, k, r]: kek) {
                                if (sum_weight[k][r] != 0) {
                                    weight = weight / sum_weight[k][r];
                                } else {
                                    weight = 0;
                                }
                            }
                        }

                        sort(kek.begin(), kek.end());

                        // verify
                        {
                            vector<vector<double>> sum_weight(K, vector<double>(R));
                            for (auto [weight, n, k, r]: kek) {
                                sum_weight[k][r] += weight;
                                if (weight < 0 || weight > 1 || is_spoiled(weight)) {
                                    exit(1);
                                }
                            }
                            for (int k = 0; k < K; k++) {
                                for (int r = 0; r < R; r++) {
                                    if (sum_weight[k][r] != 0 && abs(sum_weight[k][r] - 1) > 1e-9) {
                                        exit(1);
                                    }
                                }
                            }
                        }
                    }

                    // set power in time t
                    {
                        for (auto [weight, n, k, r]: kek) {
                            p[best_time][n][k][r] = weight * 4;
                        }

                        vector<double> sum(K);
                        for (auto [weight, n, k, r]: kek) {
                            sum[k] += p[best_time][n][k][r];
                        }
                        for (auto [weight, n, k, r]: kek) {
                            if (sum[k] > 1e-9) {
                                p[best_time][n][k][r] *= min(1.0, R / sum[k]);
                            }
                        }
                    }

                    // update total_g
                    for (int j: messagease_in_time[best_time]) {
                        auto [TBS, n, t0, t1] = Queries[j];
                        total_g[j] += calc_g(best_time, n);
                        if (total_g[j] >= TBS) {
                            for (int t = t0; t <= t1; t++) {
                                if (!messagease_in_time[t].empty() && t != best_time) {
                                    messagease_in_time[t].erase(
                                            std::find(messagease_in_time[t].begin(), messagease_in_time[t].end(), j));
                                }
                            }
                        }
                    }

                    /*cout << "NICE\n";
                    find = true;
                    int j = messagease_in_time[t].back();
                    messagease_in_time[t].pop_back();
                    auto [TBS, n, t0, t1] = Queries[j];

                    for (int k = 0; k < K; k++) {
                        for (int r = 0; r < R; r++) {
                            p[t][n][k][r] = 4;
                        }
                    }

                    vector<double> sum(K);
                    for (int k = 0; k < K; k++) {
                        for (int r = 0; r < R; r++) {
                            sum[k] += p[t][n][k][r];
                        }
                    }
                    for (int k = 0; k < K; k++) {
                        for (int r = 0; r < R; r++) {
                            if (sum[k] > 1e-9) {
                                p[t][n][k][r] *= min(1.0, R / sum[k]);
                            }
                        }
                    }

                    total_g[j] += calc_g(t, n);*/
                }

                messagease_in_time[best_time].clear();
            }
            return total_g;
        };

        auto f = [&](const vector<double> &total_g) {
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
                X += total_g[j] >= Queries[j].TBS;
            }
            return X - 1e-6 * power_sum;
        };

        auto new_solution2 = [&]() { // NOLINT
            // разделить время на блоки некоторой длины. типа 100 * C
            // в каждом блоке решать задачу по отдельности

            vector<double> total_g(J);

            /*auto solve = [&](const vector<bool> &used) {
                // used[j] = true => обязательно нужно доставить j

                // тогда нам обязательно нужно доставить и самые сложно-доставимые из этих
                // давайте сделаем упор на них, а маленькие как-нибудь уж доставим

                for (int j = 0; j < J; j++) {
                    total_g[j] = 0;
                }

                vector<tuple<double, int>> segments(J);
                for (int j = 0; j < J; j++) {
                    segments[j] = {Queries[j].TBS, j};
                }
                sort(segments.begin(), segments.end(), greater<>());
                for (auto [difficult, j]: segments) {
                    // удовлетворим этот запрос так, чтобы не поломать предыдущие

                }

                return true;
            };*/

            static mt19937 rnd(42);

            vector<double> best_g(J);

            vector<vector<int>> ss(T);
            for (int j = 0; j < J; j++) {
                auto [TBS, user_id, t0, t1] = Queries[j];
                ss[t0].push_back(j);
            }

            auto solve = [&](int t, vector<int> js) { // NOLINT
                // веса нужны от [0, 1]
                // сумма весов при заданных k, r должна быть равна 1
                vector<tuple<double, int, int, int>> kek;
                {
                    for (int j: js) {
                        auto [TBS, n, t0, t1] = Queries[j];

                        for (int k = 0; k < K; k++) {
                            for (int r = 0; r < R; r++) {
                                double weight = 1;
                                //weight *= exp(pow((t1 - t0 + 1) * 1.0 / (t1 - t + 1), 2.1) - pow(TBS - data.g, 0.5));

                                //weight *= s0[t][n][k][r];

                                /*for(int j2 : js){
                                    int m = Queries[j2].user_id;
                                    if (n != m) {
                                        weight *= exp_d_pow[n][m][k][r];
                                    }
                                }*/

                                /*weight += 100;
                                for (int k2 = 0; k2 < K; k2++) {
                                    if (k != k2) {
                                        weight -= pow(s0[t][n][k2][r], 0.5);
                                    }
                                }*/

                                kek.emplace_back(weight, n, k, r);

                                if (weight < 0 || is_spoiled(weight)) {
                                    exit(1);
                                }
                            }
                        }
                    }

                    // normalize weights

                    // sum
                    {
                        vector<vector<double>> sum_weight(K, vector<double>(R));
                        for (auto [weight, n, k, r]: kek) {
                            sum_weight[k][r] += weight;
                        }
                        for (auto &[weight, n, k, r]: kek) {
                            if (sum_weight[k][r] != 0) {
                                weight = weight / sum_weight[k][r];
                            } else {
                                weight = 0;
                            }
                        }
                    }

                    // sum
                    {
                        vector<vector<double>> sum_weight(K, vector<double>(R));
                        for (auto [weight, n, k, r]: kek) {
                            sum_weight[k][r] += weight;
                        }
                        for (auto &[weight, n, k, r]: kek) {
                            if (sum_weight[k][r] != 0) {
                                weight = weight / sum_weight[k][r];
                            } else {
                                weight = 0;
                            }
                        }
                    }

                    // cout << "weights: ";
                    // for (auto &[weight, n]: kek) {
                    //     cout << weight << ' ';
                    // }
                    // cout << '\n';

                    sort(kek.begin(), kek.end());
                    reverse(kek.begin(), kek.end());
                    while (kek.size() > 1 && get<0>(kek.back()) < 0.02) {
                        kek.pop_back();
                    }
                    reverse(kek.begin(), kek.end());

                    // sum
                    {
                        vector<vector<double>> sum_weight(K, vector<double>(R));
                        for (auto [weight, n, k, r]: kek) {
                            sum_weight[k][r] += weight;
                        }
                        for (auto &[weight, n, k, r]: kek) {
                            if (sum_weight[k][r] != 0) {
                                weight = weight / sum_weight[k][r];
                            } else {
                                weight = 0;
                            }
                        }
                    }

                    sort(kek.begin(), kek.end());

                    // verify
                    {
                        vector<vector<double>> sum_weight(K, vector<double>(R));
                        for (auto [weight, n, k, r]: kek) {
                            sum_weight[k][r] += weight;
                            if (weight < 0 || weight > 1 || is_spoiled(weight)) {
                                exit(1);
                            }
                        }
                        for (int k = 0; k < K; k++) {
                            for (int r = 0; r < R; r++) {
                                if (sum_weight[k][r] != 0 && abs(sum_weight[k][r] - 1) > 1e-9) {
                                    exit(1);
                                }
                            }
                        }
                    }
                }

                // set power in time t
                {
                    for (auto [weight, n, k, r]: kek) {
                        p[t][n][k][r] = weight * 4;
                    }

                    vector<double> sum(K);
                    for (auto [weight, n, k, r]: kek) {
                        sum[k] += p[t][n][k][r];
                    }
                    for (auto [weight, n, k, r]: kek) {
                        if (sum[k] > 1e-9) {
                            p[t][n][k][r] *= min(1.0, R / sum[k]);
                        }
                    }
                }

                // update g

                for (auto &[n, data]: users) {
                    add_g[t][n] = calc_g(t, n);
                    data.g += add_g[t][n];

                    // мы уже отправили все
                    if (data.g >= Queries[data.j].TBS) {
                        need_delete.emplace_back(data.g - Queries[data.j].TBS, n);
                    }
                }
                sort(need_delete.begin(), need_delete.end(), greater<>());
                for (auto [weight, n]: need_delete) {
                    int j = users[n].j;
                    double g = users[n].g;
                    total_g[j] = users[n].g;
                    users.erase(n);
                    do_smaller(t, j, g);
                }
                return true;
            };

            auto build_js = [&](int t, int msk) {
                vector<int> js;
                for (int i = 0; i < ss[t].size(); i++) {
                    if ((msk >> i) & 1) {
                        js.push_back(ss[t][i]);
                    }
                }
                return js;
            };

            for (int t = 0; t < T; t++) {
                if (ss[t].empty()) {
                    continue;
                }

                int best_msk = 0;
                for (int msk = 0; msk < (1 << ss[t].size()); msk++) {
                    if (solve(t, build_js(t, msk))) {
                        if (__builtin_popcount(best_msk) < __builtin_popcount(msk)) {
                            best_msk = msk;
                        }
                    }
                }
                solve(t, build_js(t, best_msk));
            }
            /*for (int step = 0; step < 10; step++) {
                vector<bool> used(J);
                for (int j = 0; j < J; j++) {
                    used[j] = rnd() & 1;
                }
                bool ok = solve(used);
                cout << "ok: " << ok << ' ';
                if (ok) {
                    double x = f(total_g);
                    cout << x;
                    if (x > f(best_g)) {
                        best_g = total_g;
                    }
                }
                cout << endl;
            }*/
            return best_g;
        };

        {
            vector<int> cnt(T);
            for (int j = 0; j < J; j++) {
                auto [TBS, user_id, t0, t1] = Queries[j];
                for (int t = t0; t <= t1; t++) {
                    cnt[t]++;
                }
            }
            cout << "T: " << T << '\n';
            for (int t = 0; t < T; t++) {
                cout << cnt[t] << ' ';
            }
            cout << '\n';
        }

        vector<double> weight_factor(J, 1.0);

        auto ans_power = p;
        int ans_count = 0;
        double best_score = 0;

        // 1 раз solution 10586.882 points
        // 1 секунду solution 11828.898 points с адаптивным weight

        // set nice weight_factor
        {
            for (int j = 0; j < J; j++) {
                auto [TBS, n, t0, t1] = Queries[j];
                weight_factor[j] = 1.0 / TBS * (t1 - t0 + 1);
            }

            // normalize
            {

                double min_weight = 0;
                for (int j = 0; j < J; j++) {
                    min_weight = min(min_weight, weight_factor[j]);
                }
                if (min_weight < 0) {
                    for (int j = 0; j < J; j++) {
                        weight_factor[j] -= min_weight;
                    }
                }

                double sum_weight = 0;
                for (int j = 0; j < J; j++) {
                    sum_weight += weight_factor[j];
                }
                for (int j = 0; j < J; j++) {
                    weight_factor[j] /= sum_weight;
                }
                sum_weight = 0;
                for (int j = 0; j < J; j++) {
                    sum_weight += weight_factor[j];
                }
                if (abs(sum_weight - 1) > 1e-9) {
                    cout << "fatal: " << sum_weight << '\n';
                    exit(1);
                }
            }
        }

        for (int step = 0; step < 1; step++) {
            /*static mt19937 rnd(42);
            if (step % 50 == 0) {
                for (int j = 0; j < J; j++) {
                    weight_factor[j] *= 1 + (int(rnd()) * 1.0 / INT_MAX) / 10;
                }
            }*/
            /*auto time_stop = steady_clock::now();
            auto duration = time_stop - time_start;
            double time = duration_cast<nanoseconds>(duration).count() / 1e9;

            if (time > 1.6) {
                break;
            }*/

            auto total_g = new_solution2();

            //
            /*{
                auto x = weight_factor;
                sort(x.begin(), x.end(), greater<>());
                for (int j = 0; j < J; j++) {
                    cout << x[j] << ' ';
                }
                cout << '\n';
            }*/

            // update best
            {
                best_score = max(best_score, f(total_g));
                int cur_count = 0;
                for (int j = 0; j < J; j++) {
                    cur_count += total_g[j] >= Queries[j].TBS;
                }
                if (cur_count > ans_count) {
                    ans_count = cur_count;
                    ans_power = p;
                }
            }

            // update weight_factor
            {
                // посмотрим на те, которые нам не удалось передать
                // возможно им стоило уделить больше внимания: повысить
                // weight_factor возможно меньше: понизить
                {
                    for (int j = 0; j < J; j++) {
                        auto [TBS, n, t0, t1] = Queries[j];
                        if (total_g[j] < 0) {
                            exit(1);
                        }
                        if (total_g[j] < TBS) {
                            weight_factor[j] += TBS / max(TBS / 3.0, total_g[j]);
                        } else if (total_g[j] > TBS) {
                            weight_factor[j] += total_g[j] / TBS;
                        }
                    }
                }

                // normalize
                {

                    double min_weight = 0;
                    for (int j = 0; j < J; j++) {
                        min_weight = min(min_weight, weight_factor[j]);
                    }
                    if (min_weight < 0) {
                        for (int j = 0; j < J; j++) {
                            weight_factor[j] -= min_weight;
                        }
                    }

                    double sum_weight = 0;
                    for (int j = 0; j < J; j++) {
                        sum_weight += weight_factor[j];
                    }
                    for (int j = 0; j < J; j++) {
                        weight_factor[j] /= sum_weight;
                    }
                    sum_weight = 0;
                    for (int j = 0; j < J; j++) {
                        sum_weight += weight_factor[j];
                    }
                    if (abs(sum_weight - 1) > 1e-9) {
                        cout << "fatal: " << sum_weight << '\n';
                        exit(1);
                    }
                }
            }
        }

        cout << "score: " << best_score << '/' << J << endl;

        /*for (int j = 0; j < J; j++) {
            cout << weight_factor[j] << ' ';
        }
        cout << '\n';
        cout << ans_count << '\n';*/

        // ==========
        // ==OUTPUT==
        // ==========

        // cout << fixed << setprecision(10);
        /*for (int t = 0; t < T; t++) {
            for (int k = 0; k < K; k++) {
                for (int r = 0; r < R; r++) {
                    for (int n = 0; n < N; n++) {
    #ifdef FAST_STREAM
                        writeDouble(ans_power[t][n][k][r]);
                        writeChar(' ');
    #else
                        cout << ans_power[t][n][k][r] << ' ';
    #endif
                    }
    #ifdef FAST_STREAM
                    writeChar('\n');
    #else
                    cout << '\n';
    #endif
                }
            }
        }*/
    }
}