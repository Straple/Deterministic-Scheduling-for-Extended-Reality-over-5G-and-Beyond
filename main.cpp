#include <bits/stdc++.h>

using namespace std;

/**
 * Author: Sergey Kopeliovich (Burunduk30@gmail.com)
 */

#define VERSION "0.1.8"

#include <cassert>
#include <cstdio>
#include <algorithm>

/** Fast allocation */

#ifdef FAST_ALLOCATOR_MEMORY
int allocator_pos = 0;
    char allocator_memory[(int)FAST_ALLOCATOR_MEMORY];
    inline void * operator new (size_t n) {
        // fprintf(stderr, "n=%ld\n", n);
        char *res = allocator_memory + allocator_pos;
        assert(n <= (size_t)((int)FAST_ALLOCATOR_MEMORY - allocator_pos));
        allocator_pos += n;
        return (void *)res;
    }
    inline void operator delete (void *) noexcept { }
    inline void operator delete (void *, size_t) noexcept { }
    //inline void * operator new [] (size_t) { assert(0); }
    //inline void operator delete [] (void *) { assert(0); }
#endif

/** Fast input-output */

template<class T = int>
inline T readInt();

inline double readDouble();

inline int readUInt();

inline int readChar(); // first non-blank character
inline void readWord(char *s);

inline bool readLine(char *s); // do not save '\n'
inline bool isEof();

inline int getChar();

inline int peekChar();

inline bool seekEof();

inline void skipBlanks();

template<class T>
inline void writeInt(T x, char end = 0, int len = -1);

inline void writeChar(int x);

inline void writeWord(const char *s);

inline void writeDouble(double x, int len = 10); // works correct only for |x| < 2^{63}
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
            x = x * 10 - (c - '0'); // take care about -2^{31}
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
        s[n++] = '0' + abs((int) (x % 10)), x /= 10;  // `abs`: take care about -2^{31}
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

#define FAST_STREAM

int main() {
    //std::ifstream cin("input.txt");
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
    vector<vector<vector<vector<double >> >> s0(T, vector(N, vector(K, vector<double>(R))));
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

    vector<vector<vector<vector<double >> >> d(N, vector(N, vector(K, vector<double>(R))));

    // exp_d[n][m][k][r]
    vector<vector<vector<vector<double >> >> exp_d(N, vector(N, vector(K, vector<double>(R))));
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

    vector<vector<int>> event_add(T), event_remove(T);
    for (int j = 0; j < J; j++) {
        event_add[Queries[j].t0].push_back(j);
        event_remove[Queries[j].t1].push_back(j);
    }

    struct data {
        int j; // номер окна
        double g;
    };

    auto calc_g = [&](int t, int n) { // NOLINT
        // dp_sum_noeq[k][r]
        vector<vector<double>> dp_sum_noeq(K, vector<double>(R, 1));
        // dp_sum[k][r]
        vector<vector<double>> dp_sum(K, vector<double>(R));
        {
            for (int m = 0; m < N; m++) {
                if (m != n) {
                    for (int k = 0; k < K; k++) {
                        for (int r = 0; r < R; r++) {
                            dp_sum[k][r] += s0[t][n][k][r] * p[t][m][k][r] / exp_d[n][m][k][r];
                        }
                    }
                }
            }

            for (int k = 0; k < K; k++) {
                for (int k1 = 0; k1 < K; k1++) {
                    if (k != k1) {
                        for (int r = 0; r < R; r++) {
                            dp_sum_noeq[k][r] += dp_sum[k1][r];
                        }
                    }
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
                    accum_prod /= dp_sum_noeq[k][r];

                    for (int m = 0; m < N; m++) {
                        if (n != m) {
                            if (p[t][m][k][r] > 0) {
                                accum_prod *= exp_d[n][m][k][r];
                            }
                        }
                    }
                }
            }

            sum += count * log2(1 + std::pow(accum_prod, 1.0 / count));
        }
        return 192 * sum;
    };

    vector<vector<double>> add_g(T, vector<double>(N));

    map<int, data> users;

    auto do_smaller = [&](int t, int j, double received_TBS) { // NOLINT
        auto [TBS, n, t0, _] = Queries[j];
        // максимально сильно уменьшим применяемую силу, но так, чтобы все еще получали TBS
        auto calc_TBS = [&](double power_factor) {
            double cur_TBS = 0;
            vector<vector<double>> save_p(K, vector<double>(R));
            for (int k = 0; k < K; k++) {
                for (int r = 0; r < R; r++) {
                    save_p[k][r] = p[t][n][k][r];
                    p[t][n][k][r] *= power_factor;
                }
            }
            cur_TBS += calc_g(t, n);

            for (int k = 0; k < K; k++) {
                for (int r = 0; r < R; r++) {
                    p[t][n][k][r] = save_p[k][r];
                }
            }

            return cur_TBS;
        };

        double X = calc_TBS(1);
        received_TBS -= X;

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
            users.erase(n);
        }
    };

    /*for (int t = 0; t < T; t++) {
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
                        weight *= exp(pow((t1 - t0 + 1) * 1.0 / (t1 - t + 1), 2.1));
                        weight /= exp(pow(TBS - data.g, 0.6));

                        for (auto [m, data2]: users) {
                            if (n != m) {
                                weight *= pow(exp_d[m][n][k][r], 1);
                                weight *= pow(exp_d[n][m][k][r], 2.5);
                            }
                        }

                        kek.emplace_back(weight, n, k, r);

                        if (is_spoiled(weight)) {
                            return 0;
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

            //cout << "weights: ";
            //for (auto &[weight, n]: kek) {
            //    cout << weight << ' ';
            //}
            //cout << '\n';

            sort(kek.begin(), kek.end());
            reverse(kek.begin(), kek.end());
            while (kek.size() > 1 && get<0>(kek.back()) < 0.001) {
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
                    if (weight < 0 || weight > 1) {
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
                p[t][n][k][r] *= min(1.0, R / sum[k]);
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
            double TBS = users[n].g;
            users.erase(n);
            do_smaller(t, j, TBS);
        }

        // remove

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
            users.erase(n);

            // TODO: если мы не смогли набрать TBS, то нам не нужно было тратить туда силу
            for (int t = Queries[j].t0; t <= Queries[j].t1; t++) {
                for (int k = 0; k < K; k++) {
                    for (int r = 0; r < R; r++) {
                        p[t][n][k][r] = 0;
                    }
                }
            }

            vector<pair<double, int>> need_delete;
            for (auto &[m, data2]: users) {
                for (int time = Queries[j].t0; time <= t; time++) {
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
                double TBS = users[n].g;
                int j = users[n].j;
                users.erase(n);
                do_smaller(t, j, TBS);
            }
        }
    }*/

    // ==========
    // ==RANDOM==
    // ==========

    vector<double> total_g(J);

    auto f = [&]() { // NOLINT
        double score = 0;

        // мы не хотим, чтобы были некорректные p
        // поэтому будем за такое снижать score
        {
            for (int t = 0; t < T; t++) {
                for (int k = 0; k < K; k++) {
                    double sum = 0;
                    for (int r = 0; r < R; r++) {
                        double sum_for_users = 0;
                        for (int n = 0; n < N; n++) {
                            sum_for_users += p[t][n][k][r];

                            if (p[t][n][k][r] < 0) {
                                score -= abs(p[t][n][k][r]) * 1e7;
                            }
                        }
                        if (sum_for_users > 4) {
                            score -= (sum_for_users - 4) * 1e9;
                        }
                        sum += sum_for_users;
                    }
                    if (sum > R) {
                        score -= (sum - R) * 1e8;
                    }
                }
            }
        }

        vector<vector<double>> dp_g(T, vector<double>(N));
        for (int t = 0; t < T; t++) {
            for (int n = 0; n < N; n++) {
                dp_g[t][n] = calc_g(t, n);
            }
        }

        int count_accept = 0;
        for (int j = 0; j < J; j++) {
            auto [TBS, user_id, t0, t1] = Queries[j];
            double g = 0;
            for (int t = t0; t <= t1; t++) {
                g += dp_g[t][user_id];
            }
            total_g[j] = g;
            count_accept += g >= TBS;
            if (g < TBS) {
                score += (g / TBS) * 10;
            }
        }
        return score + count_accept * 1000;
    };

// #define PRINT_UPDATES_F

    mt19937 rnd(42);
#ifdef PRINT_UPDATES_F
    cout << f();
#endif

    double cur_f = f();
    for (int step = 0;; step++) {

        auto time_stop = steady_clock::now();
        auto duration = time_stop - time_start;
        double time = duration_cast<nanoseconds>(duration).count() / 1e9;

        if (time > 1.8) {
            break;
        }

        int t = rnd() % T;
        int n = rnd() % N;
        int k = rnd() % K;
        int r = rnd() % R;
        // p[t][n][k][r]
        double x = 1.0 / (1ULL << (rnd() % 32)) * (rnd() & 1 ? +1 : -1);

        double new_f = cur_f;

        auto save_total_g = total_g;

        // вернем обратно
        {
            // мы не хотим, чтобы были некорректные p
            // поэтому будем за такое снижать score
            {
                double sum = 0;
                for (int r = 0; r < R; r++) {
                    double sum_for_users = 0;
                    for (int n = 0; n < N; n++) {
                        sum_for_users += p[t][n][k][r];

                        if (p[t][n][k][r] < 0) {
                            new_f += abs(p[t][n][k][r]) * 1e7;
                        }
                    }
                    if (sum_for_users > 4) {
                        new_f += (sum_for_users - 4) * 1e9;
                    }
                    sum += sum_for_users;
                }
                if (sum > R) {
                    new_f += (sum - R) * 1e8;
                }
            }

            for (int j = 0; j < J; j++) {
                auto [TBS, user_id, t0, t1] = Queries[j];
                if (t0 <= t && t <= t1) {
                    new_f -= (total_g[j] >= TBS) * 1000;
                    if (total_g[j] < TBS) {
                        new_f -= (total_g[j] / TBS) * 10;
                    }
                    total_g[j] -= calc_g(t, user_id);
                }
            }
        }
        p[t][n][k][r] += x;
        // поставим новое
        {
            // мы не хотим, чтобы были некорректные p
            // поэтому будем за такое снижать score
            {
                double sum = 0;
                for (int r = 0; r < R; r++) {
                    double sum_for_users = 0;
                    for (int n = 0; n < N; n++) {
                        sum_for_users += p[t][n][k][r];

                        if (p[t][n][k][r] < 0) {
                            new_f -= abs(p[t][n][k][r]) * 1e7;
                        }
                    }
                    if (sum_for_users > 4) {
                        new_f -= (sum_for_users - 4) * 1e9;
                    }
                    sum += sum_for_users;
                }
                if (sum > R) {
                    new_f -= (sum - R) * 1e8;
                }
            }

            for (int j = 0; j < J; j++) {
                auto [TBS, user_id, t0, t1] = Queries[j];
                if (t0 <= t && t <= t1) {
                    total_g[j] += calc_g(t, user_id);

                    new_f += int(total_g[j] >= TBS) * 1000;
                    if (total_g[j] < TBS) {
                        new_f += (total_g[j] / TBS) * 10;
                    }
                }
            }
        }

        if ((new_f - f()) > 1e-6) {
            exit(1);
            //cout << "error: " << (new_f - f()) << '\n';
        }

        if (new_f > cur_f) {
            cur_f = new_f;
#ifdef PRINT_UPDATES_F
            cout << "->" << cur_f;
#endif
        } else {
            p[t][n][k][r] -= x;
            total_g = save_total_g;
        }
    }
#ifdef PRINT_UPDATES_F
    cout << '\n';
#endif

    //cout << f() << '\n';

    // ==========
    // ==OUTPUT==
    // ==========

    //cout << fixed << setprecision(10);
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