#include <bits/stdc++.h>

using namespace std;

//#define VERIFY_G
//#define READ_POWER

int main() {
    //std::ifstream cin("input.txt");
    std::ios::sync_with_stdio(false);
    std::cin.tie(0);
    std::cout.tie(0);

    // ===========
    // ==READING==
    // ===========

    int N;
    int K;
    int T;
    int R;
    cin >> N >> K >> T >> R;

    // s0[t][n][k][r]
    vector<vector<vector<vector<double>>>> s0(T, vector(N, vector(K, vector<double>(R))));
    for (int t = 0; t < T; t++) {
        for (int k = 0; k < K; k++) {
            for (int r = 0; r < R; r++) {
                for (int n = 0; n < N; n++) {
                    cin >> s0[t][n][k][r];
                }
            }
        }
    }

    vector<vector<vector<vector<double>>>> d(N, vector(N, vector(K, vector<double>(R))));

    // exp_d[n][m][k][r]
    vector<vector<vector<vector<double>>>> exp_d(N, vector(N, vector(K, vector<double>(R))));
    for (int k = 0; k < K; k++) {
        for (int r = 0; r < R; r++) {
            for (int m = 0; m < N; m++) {
                for (int n = 0; n < N; n++) {
                    cin >> d[n][m][k][r];
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
    cin >> J;

    vector<message_window> Queries(J);
    for (int i = 0; i < J; i++) {
        int j;
        cin >> j;

        cin >> Queries[j].TBS;
        cin >> Queries[j].user_id;

        int t0, td;
        cin >> t0 >> td;
        int t1 = t0 + td - 1;

        Queries[j].t0 = t0;
        Queries[j].t1 = t1;
    }

    // p[t][n][k][r]
    vector<vector<vector<vector<double>>>> p(T, vector(N, vector(K, vector<double>(R))));

#ifdef READ_POWER
    for (int t = 0; t < T; t++) {
        for (int k = 0; k < K; k++) {
            for (int r = 0; r < R; r++) {
                for (int n = 0; n < N; n++) {
                    cin >> p[t][n][k][r];
                }
            }
        }
    }
#endif

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
    for (int t = 0; t < T; t++) {
        // add
        for (int j: event_add[t]) {
            int n = Queries[j].user_id;
            users[n] = {j, 0};
        }

        // веса нужны от [0, 1]
        // сумма весов должна быть 1
        vector<pair<double, int>> kek;
        {
            for (auto [n, data]: users) {
                auto [TBS, user_id, t0, t1] = Queries[data.j];

                double weight = 1;
                weight *= exp(pow(TBS - data.g, 0.58));
                weight /= exp(pow((t1 - t0 + 1) * 1.0 / (t1 - t + 1), 2.1));
                weight = 1 / weight;

                kek.emplace_back(weight, n);
            }

            // normalize weights

            double min_weight = 0;
            for (auto [weight, n]: kek) {
                min_weight = min(min_weight, weight);
            }
            for (auto &[weight, n]: kek) {
                weight -= min_weight;
            }

            double sum_weight = 0;
            for (auto [weight, n]: kek) {
                sum_weight += weight;
            }
            if (sum_weight == 0) {
                sum_weight = kek.size();
                for (auto &[weight, n]: kek) {
                    weight = 1;
                }
            }
            for (auto &[weight, n]: kek) {
                weight = weight / sum_weight;
            }

            //cout << "weights: ";
            //for (auto &[weight, n]: kek) {
            //    cout << weight << ' ';
            //}
            //cout << '\n';

            sort(kek.begin(), kek.end());

            // verify
            {
                double sum_weight = 0;
                for (auto [weight, n]: kek) {
                    sum_weight += weight;
                    if (weight + 1e-9 < 0 || weight - 1e-9 > 1) {
                        exit(1);
                    }
                }
                if (!kek.empty() && abs(sum_weight - 1) > 1e-9) {
                    exit(1);
                }
            }
        }

        // set power in time t
        {
            for (auto [weight, n]: kek) {
                for (int k = 0; k < K; k++) {
                    for (int r = 0; r < R; r++) {
                        p[t][n][k][r] = weight * 4;
                    }
                }
            }

            for (int k = 0; k < K; k++) {
                double sum = 0;
                for (auto [weight, n]: kek) {
                    for (int r = 0; r < R; r++) {
                        sum += p[t][n][k][r];
                    }
                }

                for (auto [weight, n]: kek) {
                    for (int r = 0; r < R; r++) {
                        p[t][n][k][r] *= min(1.0, R / sum);
                    }
                }
            }
        }

        // update g

        vector<int> need_delete;
        for (auto &[n, data]: users) {

            add_g[t][n] = calc_g(t, n);
            data.g += add_g[t][n];

            // мы уже отправили все
            if (data.g >= Queries[data.j].TBS) {
                need_delete.push_back(n);
            }

            //cout << "hi: " << n << ' ' << data.g << '\n';
        }
        for (int n: need_delete) {
            users.erase(n);
        }

        // remove
        for (int j: event_remove[t]) {
            int n = Queries[j].user_id;
            if (users.contains(n)) {

                // TODO: если мы не смогли набрать TBS, то нам не нужно было тратить туда силу
                for (int t = Queries[j].t0; t <= Queries[j].t1; t++) {
                    for (int k = 0; k < K; k++) {
                        for (int r = 0; r < R; r++) {
                            p[t][n][k][r] = 0;
                        }
                    }
                }
                users.erase(n);

                vector<int> need_delete;
                for (auto &[m, data2]: users) {
                    for (int time = Queries[j].t0; time <= t; time++) {
                        data2.g -= add_g[time][m];
                        add_g[time][m] = calc_g(time, m);
                        data2.g += add_g[time][m];
                    }
                    if (data2.g >= Queries[data2.j].TBS) {
                        need_delete.push_back(m);
                    }
                }
                for (int n: need_delete) {
                    users.erase(n);
                }
            }
        }
    }

#ifdef VERIFY_G
    {
        // NOLINTNEXTLINE
        auto build_s_krnt = [&]() {
            // s[k][r][n][t]
            vector<vector<vector<vector<double>>>> s(K, vector(R, vector(N, vector<double>(T))));

            for (int k = 0; k < K; k++) {
                for (int r = 0; r < R; r++) {
                    for (int n = 0; n < N; n++) {
                        for (int t = 0; t < T; t++) {

                            double prod = 1;
                            for (int m = 0; m < N; m++) {
                                if (m != n) {
                                    prod *= exp(d[n][m][k][r] * (p[t][m][k][r] > 0 ? 1 : 0));
                                }
                            }
                            prod *= p[t][n][k][r];
                            prod *= s0[t][n][k][r];

                            double sum = 1;
                            for (int k1 = 0; k1 < K; k1++) {
                                if (k1 != k) {
                                    for (int n1 = 0; n1 < N; n1++) {
                                        if (n1 != n) {
                                            sum += s0[t][n][k1][r] * p[t][n1][k1][r] * exp(-d[n][n1][k1][r]);
                                        }
                                    }
                                }
                            }

                            s[k][r][n][t] = prod / sum;
                        }
                    }
                }
            }
            return s;
        };

        auto build_s_knt = [&](vector<vector<vector<vector<double>>>> s) {
            vector<vector<vector<double>>> s_cur(K, vector(N, vector<double>(T)));
            for (int k = 0; k < K; k++) {
                for (int n = 0; n < N; n++) {
                    for (int t = 0; t < T; t++) {
                        double prod = 1;
                        int count = 0;
                        for (int r = 0; r < R; r++) {
                            if (p[t][n][k][r] > 0) {
                                count++;
                                prod *= s[k][r][n][t];
                            }
                        }

                        s_cur[k][n][t] = std::pow(prod, 1.0 / count);
                    }
                }
            }
            return s_cur;
        };

        auto build_g = [&](vector<vector<vector<double>>> s) {
            vector<double> g(J);
            for (int j = 0; j < J; j++) {
                int n = Queries[j].user_id;
                for (int t = Queries[j].t0; t <= Queries[j].t1; t++) {
                    for (int k = 0; k < K; k++) {
                        for (int r = 0; r < R; r++) {
                            g[j] += (p[t][n][k][r] > 0 ? 1 : 0) * log2(1 + s[k][n][t]);
                        }
                    }
                }
                g[j] *= 192;
            }
            return g;
        };

        auto correct_g_maybe = build_g(build_s_knt(build_s_krnt()));
        double error = 0;
        for (int j = 0; j < J; j++) {
            cout << correct_g_maybe[j] << '\n';
        }
        //cout << '\n';
        //cout << "error: " << error << '\n';
    }
#endif

    // ==========
    // ==OUTPUT==
    // ==========

    cout << fixed << setprecision(10);
    for (int t = 0; t < T; t++) {
        for (int k = 0; k < K; k++) {
            for (int r = 0; r < R; r++) {
                for (int n = 0; n < N; n++) {
                    cout << p[t][n][k][r] << ' ';
                }
                cout << '\n';
            }
        }
    }
}