#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <utility>
#include <iomanip>


void load_data(std::vector<std::pair<int, float>>& data) {
	std::ifstream ftest("test.txt");
	std::ifstream fpred("preds.txt"); 
	int tmpi;
	float tmpf;
	std::string s1, s2;
	if(ftest.is_open() && fpred.is_open())
		while (getline(ftest,s1) && getline(fpred, s2)) {
			tmpi = static_cast<int>(s1[0] - '0');
			tmpf = stof(s2);
			data.push_back(std::make_pair(tmpi, tmpf));
		}
}

void eval_decision_threshold(std::vector<std::pair<int, float>>& data, float d, std::ofstream& fnum, std::ofstream& fper) {
	double co0 = 0;
	double co1 = 0;
	double ud = 0;
	double fa = 0;
	for (auto it : data) {
		if (it.second > d) {
			if (it.first == 1) co1 = co1 + 1;
			else fa = fa + 1;
		}
		else {
			if (it.first == 0) co0 = co0 + 1;
			else ud = ud + 1;
		}
	}
	float szum = (co0 + fa) + (co1 + ud);
	if (fnum.is_open()) {
		fnum << std::setw(3) << d << '\t';
		fnum << std::setw(9) << co0 << '\t';
		fnum << std::setw(11) << fa << '\t';
		fnum << std::setw(10) << ud << '\t';
		fnum << std::setw(9) << co1 << std::endl;
	}
	if (fper.is_open()) {
		fper << std::setprecision(3) << d << '\t';
		fper << std::setprecision(8) << co0 / (co0 + fa) << '\t';
		fper << std::setprecision(8) << fa / (co0 + fa) << '\t';
		fper << std::setprecision(8) << ud / (co1 + ud) << '\t';
		fper << std::setprecision(8) << co1 / (co1 + ud) << std::endl;
	}
}

int main() {
	std::vector<std::pair<int, float>> data;
	load_data(data);
	std::cout << "Val\t" << "Correct 0" << '\t' << "False Alarm" << '\t' << "Undetected" << '\t' << "Correct 1" << std::endl;
	std::ofstream fnum;
	fnum.open("nums.txt");
	fnum << "Val\t" << "Correct 0" << '\t' << "False Alarm" << '\t' << "Undetected" << '\t' << "Correct 1" << std::endl;
	std::ofstream fper;
	fper.open("percentage.txt");
	fper << "Val\t" << "Correct 0" << '\t' << "False Alarm" << '\t' << "Undetected" << '\t' << "Correct 1" << std::endl;
	for(float y = 0.01; y < 0.5; y += 0.01)
		eval_decision_threshold(data, y, fnum, fper);
	fnum.close();
	fper.close();

}