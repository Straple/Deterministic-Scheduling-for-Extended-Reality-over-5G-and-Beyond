#include <bits/stdc++.h>

using namespace std;

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

    // d[n][r][k][m]
    vector<vector<vector<vector<double>>>> d(N, vector(R, vector(K, vector<double>(N))));
    for (int k = 0; k < K; k++) {
        for (int r = 0; r < R; r++) {
            for (int m = 0; m < N; m++) {
                for (int n = 0; n < N; n++) {
                    cin >> d[n][r][k][m];
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
        double g; // TODO: calculate this
    };

    map<int, data> users;

    for (int t = 0; t < T; t++) {
        // add
        for (int j: event_add[t]) {
            int n = Queries[j].user_id;
            if (users.contains(n)) {
                exit(1);
            }
            users[n] = {j, 0};
        }

        vector<pair<double, int>> kek;
        for (auto [n, data]: users) {
            auto [TBS, user_id, t0, t1] = Queries[data.j];
            double weight = pow(t1 - t + 1, 4);
            weight *= pow(TBS - data.g, 6);
            weight = 1 / weight;
            kek.emplace_back(weight, n);
        }
        sort(kek.begin(), kek.end());

        // set power in time t
        {
            double sum_weight = 0;
            for (auto [weight, n]: kek) {
                sum_weight += weight;
            }

            for (auto [weight, n]: kek) {
                for (int k = 0; k < K; k++) {
                    for (int r = 0; r < R; r++) {
                        p[t][n][k][r] = weight / sum_weight * 4;
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
                        p[t][n][k][r] *= min(1.0, (weight / sum_weight) * (R / sum));
                    }
                }
            }
        }

        // update g

        vector<int> need_delete;
        for (auto &[n, data]: users) {
            double sum = 0;
            for (int k = 0; k < K; k++) {
                double s_cur;
                {
                    double prod = 1;
                    int count = 0;
                    for (int r = 0; r < R; r++) {
                        if (p[t][n][k][r] > 0) {
                            count++;

                            double s;
                            {
                                double prod = 1;
                                for (int m = 0; m < N; m++) {
                                    if (m != n) {
                                        prod *= exp(d[n][r][k][m] * (p[t][n][k][r] > 0 ? 1 : 0));
                                    }
                                }
                                prod *= p[t][n][k][r];
                                prod *= s0[t][n][k][r];

                                double sum = 1;

                                for (int k1 = 0; k1 < K; k1++) {
                                    if (k1 != k) {
                                        for (int m = 0; m < N; m++) {
                                            if (m != n) {
                                                sum += s0[t][n][k1][r] * p[t][m][k1][r] * exp(-d[n][r][k1][m]);
                                            }
                                        }
                                    }
                                }

                                s = prod / sum;
                            }
                            prod *= s;
                        }
                    }

                    s_cur = std::pow(prod, 1.0 / count);
                }
                int count = 0;
                for (int r = 0; r < R; r++) {
                    count += p[t][n][k][r] > 0;
                }
                sum += count * log2(1 + s_cur);
            }
            data.g += 192 * sum;

            // мы уже отправили все
            if (data.g >= Queries[data.j].TBS) {
                need_delete.push_back(n);
            }
        }
        for (int n: need_delete) {
            users.erase(n);
        }

        // remove
        for (int j: event_remove[t]) {
            int n = Queries[j].user_id;
            users.erase(n);
        }
    }

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