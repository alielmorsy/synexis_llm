#pragma once

#include <deque>

template<typename T>
/**
 * A circular buffer implementation that allows for fixed-size storage of elements.
 *
 * This class supports operations to enqueue and dequeue elements while maintaining a
 * fixed maximum capacity. When the buffer reaches its capacity, older elements can
 * be overwritten based on the operation logic.
 */
class RingBuffer {
    /**
     * Stores elements in a sequential container with dynamic sizing.
     *
     * This member variable acts as the underlying container for the ring buffer.
     * It retains the elements in the order they were pushed, ensuring
     * that newer elements are always added to the back, while older
     * elements can be removed from the front when the maximum size of
     * the ring buffer is exceeded.
     */
private:
    std::deque<T> buffer;
    /**
     * Specifies the maximum number of elements that the RingBuffer can hold.
     * If the number of elements exceeds this value, the oldest element in
     * the buffer will be removed to make space for the new element being added.
     */
    size_t max_size;

    /**
     * Constructs a RingBuffer object with the specified maximum size.
     * Initializes the buffer with the given maximum capacity.
     *
     * @param size The maximum size of the ring buffer.
     */
public:
    explicit RingBuffer(const size_t size) : max_size(size) {
    }

    /**
     * Adds an element to the end of the container, resizing it if necessary.
     *
     * This method appends the specified element to the back of the container.
     *
     * @param value The element to be added to the container.
     */
    void push_back(const T &item) {
        buffer.push_back(item);
        if (buffer.size() > max_size) {
            buffer.pop_front();
        }
    }

    /**
     * @brief Clears all elements from the buffer.
     *
     * This method removes all items currently stored in the buffer, leaving it empty.
     * After calling this method, the size of the buffer will be zero.
     */
    void clear() { buffer.clear(); }
    /**
     *
     */
    size_t size() const { return buffer.size(); }
    /**
     *
     */
    bool empty() const { return buffer.empty(); }

    /**
     * Overloads the operator to define a specific behavior when it is invoked.
     *
     * The operator is used to carry out a customized operation based on its implementation.
     *
     * @param lhs The left-hand side operand of the operation.
     * @param rhs The right-hand side operand of the operation.
     * @return The result of the operation based on the implementation.
     */
    const T &operator[](size_t index) const { return buffer[index]; }
    /**
     * Overloads an operator to provide custom functionality.
     *
     * This method allows for the redefinition of the behavior of a specific operator for objects of
     * the class it resides in, enabling custom logic as required.
     *
     * @param other The operand or object to be used in the operation with the current instance.
     *              The type and significance of this parameter depend on the operator being overloaded.
     * @return The result of the operation. The return type and value depend on the custom implementation
     *         of the overloaded operator.
     */
    T &operator[](size_t index) { return buffer[index]; }

    /**
     *
     */
    auto begin() const { return buffer.begin(); }
    /**
     *
     */
    auto end() const { return buffer.end(); }
};
