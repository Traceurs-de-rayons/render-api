#include <cstddef>
#include <random>
#include <string>

std::string generateRandomString(size_t length = 15) {
	const char						charset[] = "0123456789"
												"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
												"abcdefghijklmnopqrstuvwxyz";
	std::random_device				rd;
	std::mt19937					gen(rd());
	std::uniform_int_distribution<> dist(0, sizeof(charset) - 2);

	std::string result;
	for (size_t i = 0; i < length; ++i)
		result += charset[dist(gen)];
	return result;
}
