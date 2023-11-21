#include <bits/stdc++.h>

using namespace std;

int main() {
    std::ifstream cin("input.txt");
    std::ios::sync_with_stdio(false);
    std::cin.tie(0);
    std::cout.tie(0);

    int N;
    int K;
    int T;
    int R;
    cin >> N >> K >> T >> R;

    // s0[k][r][n][t]
    vector<vector<vector<vector<double>>>> s0(K, vector(R, vector(N, vector<double>(T))));

    for (int t = 0; t < T; t++) {
        for (int k = 0; k < K; k++) {
            for (int r = 0; r < R; r++) {
                for (int n = 0; n < N; n++) {
                    cin >> s0[k][r][n][t];
                }
            }
        }
    }

    // d[k][m][r][n]
    vector<vector<vector<vector<double>>>> d(K, vector(N, vector(R, vector<double>(N))));
    for (int k = 0; k < K; k++) {
        for (int r = 0; r < R; r++) {
            for (int m = 0; m < N; m++) {
                for (int n = 0; n < N; n++) {
                    cin >> d[k][m][r][n];
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

    // p[k][r][n][t]
    vector<vector<vector<vector<double>>>> p(K, vector(R, vector(N, vector<double>(T))));

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
                                prod *= std::exp(d[k][m][r][n] * (p[k][r][n][t] > 0 ? 1 : 0));
                            }
                        }
                        prod *= p[k][r][n][t];
                        prod *= s0[k][r][n][t];

                        double sum = 1;
                        for (int k1 = 0; k1 < K; k1++) {
                            if (k1 != k) {
                                for (int n1 = 0; n1 < N; n1++) {
                                    if (n1 != n) {
                                        sum += s0[k1][r][n][t] * p[k1][r][n1][t] * exp(-d[k1][n1][r][n]);
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
                        if ((p[k][r][n][t] > 0 ? 1 : 0) == 1) {
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
                        g[j] += (p[k][r][n][t] > 0 ? 1 : 0) * log2(1 + s[k][n][t]);
                    }
                }
            }
            g[j] *= 192;
        }
        return g;
    };

    for (int t = 0; t < T; t++) {
        for (int k = 0; k < K; k++) {
            for (int r = 0; r < R; r++) {
                for (int n = 0; n < N; n++) {
                    cin >> p[k][r][n][t];
                }
            }
        }
    }

    auto g = build_g(build_s_knt(build_s_krnt()));

    for (int j = 0; j < J; j++) {
        cout << g[j] << ' ' << Queries[j].TBS << ' ' << (g[j] >= Queries[j].TBS) << '\n';
    }
    cout << '\n';

    cout << fixed << setprecision(10);
    for (int t = 0; t < T; t++) {
        for (int k = 0; k < K; k++) {
            for (int r = 0; r < R; r++) {
                for (int n = 0; n < N; n++) {
                    if (n != 0) {
                        cout << ' ';
                    }
                    cout << p[k][r][n][t];
                }
                cout << '\n';
            }
        }
    }
}