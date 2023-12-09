#include <bits/stdc++.h>

using namespace std;

vector<double> MAX_SCORE = {2,   266, 497, 800,  356, 150, 120, 160, 709, 180, 255, 380, 80,  821, 185, 571,  354,
                            150, 120, 160, 829,  181, 127, 255, 80,  837, 184, 282, 355, 150, 120, 160, 1654, 181,
                            254, 250, 80,  1657, 185, 805, 358, 150, 120, 160, 834, 182, 255, 380, 80,  825,  184};

template <typename T>
string kek_cast(T val, int len) {
    stringstream ss;
    ss << val;
    while (ss.str().size() < len) {
        ss << ' ';
    }
    return ss.str();
}

int main() {
    std::ifstream A("A.txt");
    std::ifstream B("B.txt");
    cout << "[test] [max score] [score A] [score B] [A-B]    [MAX-A] [MAX-B]\n";
    double sum_score_a = 0, sum_score_b = 0;
    for (int test = 0; test < 51; test++) {
        double a_score;
        {
            string tmp;
            A >> tmp;
            A >> tmp;
            A >> tmp;
            A >> tmp;
            A >> tmp;
            A >> tmp;
            A >> a_score;
            A >> tmp;
            sum_score_a += a_score;
        }
        double b_score;
        {
            string tmp;
            B >> tmp;
            B >> tmp;
            B >> tmp;
            B >> tmp;
            B >> tmp;
            B >> tmp;
            B >> b_score;
            B >> tmp;
            sum_score_b += b_score;
        }
        cout << kek_cast(test + 1, 6) << ' ' << kek_cast(MAX_SCORE[test], 11) << ' ' << kek_cast(a_score, 9) << ' '
             << kek_cast(b_score, 9) << ' ' << kek_cast(a_score - b_score, 8) << ' '
             << kek_cast(MAX_SCORE[test] - a_score, 7) << ' ' << kek_cast(MAX_SCORE[test] - b_score, 7) << '\n';
    }
    cout << "TOTAL A: " << sum_score_a << endl;
    cout << "TOTAL B: " << sum_score_b << endl;
}