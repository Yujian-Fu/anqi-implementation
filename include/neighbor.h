#pragma once
// Ordered fixed-capacity candidate queue for graph beam search.

#include "utils.h"

namespace nndgraph
{

// 8 字节打包：distance + (bit31=expanded, bits0-30=id)。memmove 每元素 12→8 字节（-1/3）。
struct Neighbor
{
    float    distance;
    uint32_t pid;       // bit31 = expanded 标记；bits0-30 = id（id < 2^31）

    Neighbor() = default;
    Neighbor(unsigned id, float distance) : distance{distance}, pid{id} {}

    inline uint32_t id() const { return pid & 0x7FFFFFFFu; }
    inline bool expanded() const { return (pid >> 31) != 0u; }
    inline void mark_expanded() { pid |= 0x80000000u; }

    // 按距离升序排序；距离相等时按 id 决胜，保证确定性。
    inline bool operator<(const Neighbor &other) const {
        return distance < other.distance || (distance == other.distance && id() < other.id());
    }
    inline bool operator==(const Neighbor &other) const { return (id() == other.id()); }
};

struct NeighborQueueInsertResult
{
    bool inserted = false;
    bool evicted = false;
    bool sorted_head_changed = false;
    bool frontier_changed = false;
    bool sorted_tail_changed = false;
    size_t position = 0;
    size_t changed_end = 0;
    size_t old_size = 0;
    size_t new_size = 0;
};

// 不变量：每次 insert / closest_unexpanded 之后，_cur 都指向首个未扩展的候选。
class NeighborPriorityQueue
{
public:
    NeighborPriorityQueue() : _size(0), _capacity(0), _cur(0) {}
    explicit NeighborPriorityQueue(size_t capacity)
        : _size(0), _capacity(capacity), _cur(0), _data(capacity + 1) {}

    // 按距离有序插入；若与已有候选 id 重复、或表已满且比最末候选还远，则丢弃。
    // 插入位置若在 _cur 之前，会回退 _cur（新候选可能是更近的未扩展点）。
    void insert(const Neighbor &nbr)
    {
        insert_with_result(nbr);
    }

    // Trace-friendly insertion. The normal insert() path delegates here and
    // discards the report, so queue ordering and eviction semantics stay
    // identical. [position, changed_end] is the inclusive range whose occupant
    // changed after the ordered insertion/shift.
    NeighborQueueInsertResult insert_with_result(const Neighbor &nbr)
    {
        NeighborQueueInsertResult result;
        result.old_size = _size;
        result.new_size = _size;
        if (_capacity == 0 || (_size == _capacity && _data[_size - 1] < nbr))
            return result; // 满了且更差，丢弃

        const uint32_t old_head = _size ? _data[0].id() : UINT32_MAX;
        const uint32_t old_tail = _size ? _data[_size - 1].id() : UINT32_MAX;
        const uint32_t old_frontier = _cur < _size ? _data[_cur].id() : UINT32_MAX;

        // 二分找插入位；过程中发现重复 id 直接返回。
        size_t lo = 0, hi = _size;
        while (lo < hi) {
            size_t mid = (lo + hi) >> 1;
            if (nbr < _data[mid]) {
                hi = mid;
            } else if (_data[mid].id() == nbr.id()) {
                return result; // 重复
            } else {
                lo = mid + 1;
            }
        }

        if (lo < _capacity) // 给插入点腾位（右移一格）
            std::memmove(&_data[lo + 1], &_data[lo], (_size - lo) * sizeof(Neighbor));
        _data[lo] = Neighbor(nbr.id(), nbr.distance);
        if (_size < _capacity) _size++;
        if (lo < _cur) _cur = lo; // 插在游标前：回退游标

        result.inserted = true;
        result.evicted = result.old_size == _capacity;
        result.position = lo;
        result.changed_end = _size - 1;
        result.new_size = _size;
        result.sorted_head_changed = old_head != _data[0].id();
        result.sorted_tail_changed = old_tail != _data[_size - 1].id();
        const uint32_t new_frontier = _cur < _size ? _data[_cur].id() : UINT32_MAX;
        result.frontier_changed = old_frontier != new_frontier;
        return result;
    }

    // 取出当前最近的未扩展候选，标记其为已扩展，并把游标前移到下一个未扩展位。
    Neighbor closest_unexpanded()
    {
        _data[_cur].mark_expanded();
        size_t pre = _cur;
        while (_cur < _size && _data[_cur].expanded()) _cur++;
        return _data[pre];
    }

    bool has_unexpanded_node() const { return _cur < _size; }
    uint32_t peek_unexpanded_id() const { return _data[_cur].id(); } // 下一个要展开的（_cur 处），用于预取
    size_t size() const { return _size; }
    size_t capacity() const { return _capacity; }

    // Re-score retained candidates while preserving which nodes were expanded.
    template <typename KeyFunction>
    void rekey_and_reorder(KeyFunction key_function) {
        for (size_t i = 0; i < _size; i++)
            _data[i].distance = key_function(_data[i].id());
        std::sort(_data.begin(), _data.begin() + _size);
        _cur = 0;
        while (_cur < _size && _data[_cur].expanded()) _cur++;
    }

    void reserve(size_t capacity) {
        if (capacity + 1 > _data.size()) _data.resize(capacity + 1);
        _capacity = capacity;
    }

    Neighbor &operator[](size_t i) { return _data[i]; }
    Neighbor  operator[](size_t i) const { return _data[i]; }

    void clear() { _size = 0; _cur = 0; }

private:
    size_t _size, _capacity, _cur;
    std::vector<Neighbor> _data;
};

} // namespace nndgraph
