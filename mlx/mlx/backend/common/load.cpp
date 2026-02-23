// Copyright © 2023 Apple Inc.

#include <algorithm>
#include <utility>

#include "mlx/io/load.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"

namespace {

template <const uint8_t scalar_size>
void swap_endianness(uint8_t* data_bytes, size_t N) {
  struct Elem {
    uint8_t bytes[scalar_size];
  };

  Elem* data = reinterpret_cast<Elem*>(data_bytes);

  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < (scalar_size / 2); j++) {
      std::swap(data[i].bytes[j], data[i].bytes[scalar_size - j - 1]);
    }
  }
}

} // namespace

namespace mlx::core {

void Load::eval_cpu(const std::vector<array>& inputs, array& out) {
#ifndef _MSC_VER
  // mmap zero-copy path
  if (reader_->is_mmap() && !swap_endianness_) {
    auto mmap_reader = std::dynamic_pointer_cast<io::MmapReader>(reader_);
    // Zero-copy requires the byte offset to be aligned to the element size,
    // otherwise copy_shared_buffer's integer division would truncate the offset.
    // HuggingFace safetensors files pad the JSON header for alignment, so this
    // check passes for the common case. Unaligned files fall back to memcpy.
    if (mmap_reader && (offset_ % out.itemsize() == 0)) {
      auto metal_buf = mmap_reader->get_metal_buffer();
      if (metal_buf.ptr()) {
        // Create a parent array covering the entire mmap region.
        // Capture reader_ref in the deleter to extend MmapReader lifetime.
        auto reader_ref = reader_;
        auto parent = array(
            metal_buf,
            Shape{1},
            uint8,
            [reader_ref](allocator::Buffer) {
              // Don't release the buffer — MmapReader's destructor handles it.
              // reader_ref keeps MmapReader alive.
            });

        // Compute contiguous strides for the output
        Strides strides(out.ndim());
        int64_t stride = 1;
        for (int i = out.ndim() - 1; i >= 0; --i) {
          strides[i] = stride;
          stride *= out.shape(i);
        }
        array::Flags flags;
        flags.contiguous = true;
        flags.row_contiguous = true;
        flags.col_contiguous = out.size() <= 1;

        // Create an offset view into the parent — zero copy!
        out.copy_shared_buffer(
            parent,
            strides,
            flags,
            out.size(),
            static_cast<int64_t>(offset_) /
                static_cast<int64_t>(out.itemsize()));
        return;
      }
    }
  }
#endif // _MSC_VER

  // Original copy path
  out.set_data(allocator::malloc(out.nbytes()));
  auto read_task = [out_ptr = out.data<char>(),
                    size = out.size(),
                    itemsize = out.itemsize(),
                    offset = offset_,
                    reader = reader_,
                    swap_endianness_ = swap_endianness_]() mutable {
    reader->read(out_ptr, size * itemsize, offset);
    if (swap_endianness_) {
      switch (itemsize) {
        case 2:
          swap_endianness<2>(reinterpret_cast<uint8_t*>(out_ptr), size);
          break;
        case 4:
          swap_endianness<4>(reinterpret_cast<uint8_t*>(out_ptr), size);
          break;
        case 8:
          swap_endianness<8>(reinterpret_cast<uint8_t*>(out_ptr), size);
          break;
      }
    }
  };
  auto fut = io::thread_pool().enqueue(std::move(read_task)).share();
  scheduler::enqueue(stream(), [fut = std::move(fut)]() { fut.wait(); });
}

} // namespace mlx::core
