#ifndef TENSOR_H
#define TENSOR_H

#include <vector>

template <typename T>
class Tensor {
protected:
    T* data_;
    std::vector<size_t> shape_;
    size_t size_;

public:
    Tensor();
    Tensor(const std::vector<size_t>& shape);
    Tensor(const std::vector<size_t>& shape, const T* buf);
    ~Tensor();

    void setData(const T* buf);
    void set(const std::vector<size_t>& indices, const T& value);
    T get(const std::vector<size_t>& indices) const;
    T* data() const;
    void loadData(const T* buf);

    size_t write(const char* path);

    const std::vector<size_t>& shape() const;

private:
    size_t calculateIndex(const std::vector<size_t>& indices) const;
};

#endif  // TENSOR_H