#include "loadDat.cuh"


template <typename T>
loadDat<T>::loadDat(const char* s, size_t size): size_(size) {
    fp = std::fopen(s, "rb+");
    if (!fp) {
        std::perror("File opening failed");
    }
    data_ = (T*)malloc(size * sizeof(T));
    size_t n = std::fread(data_, sizeof(T), size, fp);
    if (n != size) {
        std::perror("File reading error");
    }

}

template <typename T>
loadDat<T>::~loadDat() {
    std::fclose(fp);
    free(data_);
}

template<typename T>
T* loadDat<T>::data() {
    return data_;
}

template class loadDat<short>;
template class loadDat<double>;
template class loadDat<float>;
template class loadDat<size_t>;