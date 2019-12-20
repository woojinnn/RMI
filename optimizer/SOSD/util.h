#pragma once

#include <algorithm>
#include <chrono>
#include <iostream>
#include <functional>
#include <fstream>
#include <thread>
#include <vector>
#include <cassert>
#include <unordered_map>

//#define PRINT_ERRORS

enum DataType {
  UINT32 = 0,
  UINT64 = 1
};

template<class KeyType>
struct KeyValue {
  KeyType key;
  uint64_t value;
} __attribute__((packed));

template<class KeyType = uint64_t>
struct EqualityLookup {
  KeyType key;
  uint64_t result;
};

namespace util {

const static uint64_t NOT_FOUND = std::numeric_limits<uint64_t>::max();

static std::string get_suffix(const std::string& filename) {
  const std::size_t pos = filename.find_last_of("_");
  if (pos==filename.size() - 1 || pos==std::string::npos)
    return "";
  return filename.substr(pos + 1);
}

static DataType resolve_type(const std::string& filename) {
  const std::string suffix = get_suffix(filename);
  if (suffix=="uint32") {
    return DataType::UINT32;
  } else if (suffix=="uint64") {
    return DataType::UINT64;
  } else {
    std::cerr << "type " << suffix << " not supported" << std::endl;
    exit(EXIT_FAILURE);
  }
}

static void fail(const std::string& message) {
  std::cerr << message << std::endl;
  exit(EXIT_FAILURE);
}

// Pins the current thread to core `core_id`.
static void set_cpu_affinity(const uint32_t core_id) __attribute__((unused));
static void set_cpu_affinity(const uint32_t core_id) {
#ifdef __linux__
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(core_id % std::thread::hardware_concurrency(), &mask);
  const int result = pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
  if (result != 0)
    fail("failed to set CPU affinity");
#else
  (void) core_id;
  std::cout << "we only support thread pinning under Linux" << std::endl;
#endif
}

static uint64_t timing(std::function<void()> fn) {
  const auto start = std::chrono::high_resolution_clock::now();
  fn();
  const auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      end - start).count();
}

// Checks whether data is duplicate free.
// Note that data has to be sorted.
template<typename T>
static bool is_unique(const std::vector<T>& data) {
  for (size_t i = 1; i < data.size(); ++i) {
    if (data[i]==data[i - 1])
      return false;
  }
  return true;
}

template<class KeyType>
static bool is_unique(const std::vector<KeyValue<KeyType>>& data) {
  for (size_t i = 1; i < data.size(); ++i) {
    if (data[i].key==data[i - 1].key)
      return false;
  }
  return true;
}

// Loads values from binary file into vector.
template<typename T>
static std::vector<T> load_data(const std::string& filename,
                                bool print = true) {
  std::vector<T> data;
  const uint64_t ns = util::timing([&] {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
      std::cerr << "unable to open " << filename << std::endl;
      exit(EXIT_FAILURE);
    }
    // Read size.
    uint64_t size;
    in.read(reinterpret_cast<char*>(&size), sizeof(uint64_t));
    data.resize(size);
    // Read values.
    in.read(reinterpret_cast<char*>(data.data()), size*sizeof(T));
    in.close();
  });
  const uint64_t ms = ns/1e6;

  if (print) {
    std::cout << "read " << data.size() << " values from " << filename << " in "
              << ms << " ms (" << static_cast<double>(data.size())/1000/ms
              << " M values/s)" << std::endl;
  }

  return data;
}

// Writes values from vector into binary file.
template<typename T>
static void write_data(const std::vector<T>& data,
                       const std::string& filename,
                       const bool print = true) {
  const uint64_t ns = util::timing([&] {
    std::ofstream out(filename, std::ios_base::trunc | std::ios::binary);
    if (!out.is_open()) {
      std::cerr << "unable to open " << filename << std::endl;
      exit(EXIT_FAILURE);
    }
    // Write size.
    const uint64_t size = data.size();
    out.write(reinterpret_cast<const char*>(&size), sizeof(uint64_t));
    // Write values.
    out.write(reinterpret_cast<const char*>(data.data()), size*sizeof(T));
    out.close();
  });
  const uint64_t ms = ns/1e6;
  if (print) {
    std::cout << "wrote " << data.size() << " values to " << filename << " in "
              << ms << " ms (" << static_cast<double>(data.size())/1000/ms
              << " M values/s)" << std::endl;
  }
}

// Returns a duplicate-free copy.
// Note that data has to be sorted.
template<typename T>
static std::vector<T> remove_duplicates(const std::vector<T>& data) {
  std::vector<T> result = data;
  auto last = std::unique(result.begin(), result.end());
  result.erase(last, result.end());
  return result;
}

// Returns a value for a key at position i.
template<class KeyType>
static uint64_t get_value(const KeyType i) {
  return i;
}

// Generates deterministic values for keys.
template<class KeyType>
static std::vector<KeyValue<KeyType>> add_values(const std::vector<KeyType>& keys) {
  std::vector<KeyValue<KeyType>> result;
  result.reserve(keys.size());
  for (uint64_t i = 0; i < keys.size(); ++i) {
    result.push_back({keys[i], get_value(i)});
  }
  return result;
}

// Returns the sum over all values with the given lookup key.
// Caution: data has to be sorted.
template<class KeyType>
static uint64_t binary_search(const std::vector<KeyValue<KeyType>>& data,
                              const KeyType lookup_key,
                              size_t* num_qualifying) {
  *num_qualifying = 0;
  // Search for first occurrence of key.
  auto it = std::lower_bound(data.begin(),
                             data.end(),
                             lookup_key,
                             [](const KeyValue<KeyType>& lhs,
                                const uint64_t lookup_key) {
                               return lhs.key < lookup_key;
                             });
  if (it==data.end() || it->key!=lookup_key) {
    std::cerr << "binary search: key not found\n";
    return 0;
  }
  // Sum over all values with that key.
  uint64_t result = it->value;
  ++(*num_qualifying);

  while (++it!=data.end() && it->key==lookup_key) {
    result += it->value;
    ++(*num_qualifying);
  }
  return result;
}

template<class KeyType>
static inline uint64_t linear_search(const std::vector<KeyValue<KeyType>>& data,
                                     const KeyType lookup_key,
                                     int64_t estimate) {
  // NOTE: not tested for look ups where the key does not exist .. could be wrong ..

#ifdef PRINT_ERRORS
  const int64_t org_estimate = estimate;
  int64_t actual_key_pos = -1;
#endif

  const int64_t limit = data.size();
  assert(0 <= estimate && estimate < limit);

  // Estimated too low
  if (data[estimate].key < lookup_key) {
    estimate++;
    while (estimate < limit && data[estimate].key < lookup_key) {
      estimate++;
    }

    uint64_t result = 0;
    while (estimate < limit && data[estimate].key==lookup_key) {
      result += data[estimate].value;
#ifdef PRINT_ERRORS
      // First encountered position == first key occurrence in forward search.
      if (actual_key_pos==-1)
        actual_key_pos = estimate;
#endif
      estimate++;
    }
#ifdef PRINT_ERRORS
    std::cout << org_estimate - actual_key_pos << std::endl;
#endif
    return result;
  }

  // We estimated too high
  if (data[estimate].key > lookup_key) {
    estimate--;
    while (estimate > 0 && data[estimate].key > lookup_key) {
      estimate--;
    }

    uint64_t result = 0;
    while (estimate > 0 && data[estimate].key==lookup_key) {
      result += data[estimate].value;
#ifdef PRINT_ERRORS
      // Last encountered position == first key occurrence in backwards search.
      actual_key_pos = estimate;
#endif
      estimate--;
    }
#ifdef PRINT_ERRORS
    std::cout << org_estimate - actual_key_pos << std::endl;
#endif
    return result;
  }

  // We estimated just about right
  uint64_t initial_pos = estimate;
  uint64_t result = 0;

  // Go down
  while (estimate > 0 && data[estimate].key==lookup_key) {
    result += data[estimate].value;
#ifdef PRINT_ERRORS
    // Last encountered position == first key occurrence in backwards search.
    actual_key_pos = estimate;
#endif
    estimate--;
  }

  // Go up
  estimate = initial_pos;
  estimate++;
  while (estimate < limit && data[estimate].key==lookup_key) {
    result += data[estimate].value;
    estimate++;
  }

#ifdef PRINT_ERRORS
  std::cout << org_estimate - actual_key_pos << std::endl;
#endif
  return result;
}

template<class KeyType>
static inline uint64_t exponential_search(const std::vector<KeyValue<KeyType>>& data,
                                          const KeyType lookup_key,
                                          int64_t estimate) {
  // NOTE: not tested for look ups where the key does not exist .. could be wrong ..

  const int64_t limit = data.size();
  assert(0 <= estimate && estimate < limit);

  // Estimated just about right
  if (data[estimate].key==lookup_key) {
    uint64_t initial_pos = estimate;
    uint64_t result = 0;

    // Go down
    while (estimate > 0 && data[estimate].key==lookup_key) {
      result += data[estimate].value;
      estimate--;
    }

    // Go up
    estimate = initial_pos;
    estimate++;
    while (estimate < limit && data[estimate].key==lookup_key) {
      result += data[estimate].value;
      estimate++;
    }

    return result;
  }

  // Exponential search towards the key
  uint64_t step = 1;
  int64_t prev = estimate;
  if (data[estimate].key < lookup_key) {
    // Estimated too low -> exponential search upwards
    estimate += step;
    while (estimate < limit && data[estimate].key < lookup_key) {
      prev = estimate;
      step = step << 1;
      estimate += step;
    }

    estimate = prev;
  } else {
    // Estimated too high -> exponential search downwards
    estimate -= step;
    while (estimate > 0 && data[estimate].key >= lookup_key) {
      step = step << 1;
      estimate -= step;
    }
    if (estimate < 0) {
      estimate = 0;
    }
  }

  // Rather close now: the next exponential step upwards would be too high
  while (estimate < limit && data[estimate].key < lookup_key) {
    estimate++;
  }

  // 'estimate' points to first key now -> sum up
  uint64_t result = 0;
  while (estimate < limit && data[estimate].key==lookup_key) {
    result += data[estimate].value;
    estimate++;
  }

  return result;
}

template<class KeyType>
static uint64_t binary_search(const std::vector<KeyValue<KeyType>>& data,
                              const KeyType lookup_key,
                              size_t* num_qualifying,
                              size_t start, size_t end) {
  *num_qualifying = 0;
  // Search for first occurrence of key.
  auto it = std::lower_bound(data.begin() + start,
                             data.begin() + end,
                             lookup_key,
                             [](const KeyValue<KeyType>& lhs,
                                const uint64_t lookup_key) {
                               return lhs.key < lookup_key;
                             });
  if (it==data.end() || it->key!=lookup_key) {
    std::cerr << "key " << lookup_key << " not found between "
              << start << " and " << end << "\n";

    auto corr = std::lower_bound(data.begin(),
                                 data.end(),
                                 lookup_key,
                                 [](const KeyValue<KeyType>& lhs,
                                    const uint64_t lookup_key) {
                                   return lhs.key < lookup_key;
                                 });
    std::cerr << "correct index: " << std::distance(data.begin(), corr) << "\n";

    return 0;
  }
  // Sum over all values with that key.
  uint64_t result = it->value;
  ++(*num_qualifying);

  while (++it!=data.end() && it->key==lookup_key) {
    result += it->value;
    ++(*num_qualifying);
  }
  return result;
}

// Based on: https://en.wikipedia.org/wiki/Xorshift
class FastRandom {
 public:
  explicit FastRandom(uint64_t seed = 2305843008139952128ull) // The 8th perfect number found 1772 by Euler with <3
      : seed(seed) {}
  uint32_t RandUint32() {
    seed ^= (seed << 13);
    seed ^= (seed >> 15);
    return (uint32_t) (seed ^= (seed << 5));
  }
  int32_t RandInt32() { return (int32_t) RandUint32(); }
  uint32_t RandUint32(uint32_t inclusive_min, uint32_t inclusive_max) {
    return inclusive_min + RandUint32()%(inclusive_max - inclusive_min + 1);
  }
  int32_t RandInt32(int32_t inclusive_min, int32_t inclusive_max) {
    return inclusive_min + RandUint32()%(inclusive_max - inclusive_min + 1);
  }
  float RandFloat(float inclusive_min, float inclusive_max) {
    return inclusive_min + ScaleFactor()*(inclusive_max - inclusive_min);
  }
  // returns float between 0 and 1
  float ScaleFactor() {
    return static_cast<float>(RandUint32())
        /std::numeric_limits<uint32_t>::max();
  }
  bool RandBool() { return RandUint32()%2==0; }

  uint64_t seed;

  static constexpr uint64_t Min() { return 0; }
  static constexpr uint64_t Max() { return std::numeric_limits<uint64_t>::max(); }
};

} // namespace util
