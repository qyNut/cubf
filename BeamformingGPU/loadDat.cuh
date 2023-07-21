#ifndef LOADDAT_H
#define LOADDAT_H
#include <stdio.h>

template<typename T>
class loadDat {
private:
	size_t size_;
	T* data_;
	FILE* fp;

public:
	loadDat(const char* s, size_t size);
	~loadDat();

	T* data();

};



#endif