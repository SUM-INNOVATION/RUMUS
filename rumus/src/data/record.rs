// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Memory-mapped `.rrec` (RUMUS Record) format for high-throughput data loading.
//!
//! # File Layout
//!
//! ```text
//! [HEADER: 64 bytes]
//!   magic "RREC" (4B) | version u32 (4B) | num_records u64 (8B)
//!   | index_offset u64 (8B) | reserved (40B)
//! [DATA BLOCKS: variable]
//!   Record 0: InputMeta + InputData + TargetMeta + TargetData
//!   Record 1: ...
//! [INDEX: num_records * 16 bytes]
//!   Entry 0: offset u64 + length u64
//!   Entry 1: ...
//! ```
//!
//! The writer appends records sequentially, writes the index at the end,
//! then patches the header.  The reader mmap's the file and uses the
//! index for O(1) random access.

use std::fs::File;
use std::io::{self, BufWriter, Seek, SeekFrom, Write};

use crate::data::dataset::{DataItem, Dataset};
use crate::tensor::{DType, Tensor};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MAGIC: &[u8; 4] = b"RREC";
const VERSION: u32 = 1;
const HEADER_SIZE: u64 = 64;

// DType tags in the file format.
const DTYPE_TAG_F32: u32 = 0;
const DTYPE_TAG_F16: u32 = 1;

fn dtype_to_tag(dtype: DType) -> u32 {
    match dtype {
        DType::F32 => DTYPE_TAG_F32,
        DType::F16 => DTYPE_TAG_F16,
        DType::Q8 { .. } => panic!("Q8 tensors cannot be written to .rrec files; dequantize first"),
    }
}

// ---------------------------------------------------------------------------
// IndexEntry
// ---------------------------------------------------------------------------

/// Byte offset and length of a single record within the `.rrec` file.
#[derive(Debug, Clone, Copy)]
pub struct IndexEntry {
    /// Byte offset from file start to the record's InputMeta.
    pub offset: u64,
    /// Total bytes of this record (InputMeta+Data+TargetMeta+Data).
    pub length: u64,
}

// ---------------------------------------------------------------------------
// RecordWriter
// ---------------------------------------------------------------------------

/// Sequential writer for `.rrec` files.
///
/// # Usage
///
/// ```ignore
/// let mut writer = RecordWriter::create("train.rrec")?;
/// for (image, label) in samples {
///     writer.append(&image_tensor, &label_tensor)?;
/// }
/// writer.finish()?;
/// ```
pub struct RecordWriter {
    file: BufWriter<File>,
    index: Vec<IndexEntry>,
    current_offset: u64,
}

impl RecordWriter {
    /// Create a new `.rrec` file and write the placeholder header.
    pub fn create(path: &str) -> io::Result<Self> {
        let file = File::create(path)?;
        let mut file = BufWriter::new(file);

        // Write placeholder header (64 bytes).
        file.write_all(MAGIC)?;                           // [0..4]
        file.write_all(&VERSION.to_le_bytes())?;          // [4..8]
        file.write_all(&0u64.to_le_bytes())?;             // [8..16]  num_records (patched in finish)
        file.write_all(&0u64.to_le_bytes())?;             // [16..24] index_offset (patched in finish)
        file.write_all(&[0u8; 40])?;                      // [24..64] reserved
        file.flush()?;

        Ok(Self {
            file,
            index: Vec::new(),
            current_offset: HEADER_SIZE,
        })
    }

    /// Append a single input-target record.
    pub fn append(&mut self, input: &Tensor, target: &Tensor) -> io::Result<()> {
        let record_start = self.current_offset;
        let mut bytes_written: u64 = 0;

        // Write input tensor.
        bytes_written += self.write_tensor(input)?;

        // Write target tensor.
        bytes_written += self.write_tensor(target)?;

        self.index.push(IndexEntry {
            offset: record_start,
            length: bytes_written,
        });
        self.current_offset += bytes_written;

        Ok(())
    }

    /// Finalize the file: write the index table and patch the header.
    pub fn finish(mut self) -> io::Result<()> {
        let index_offset = self.current_offset;
        let num_records = self.index.len() as u64;

        // Write index table.
        for entry in &self.index {
            self.file.write_all(&entry.offset.to_le_bytes())?;
            self.file.write_all(&entry.length.to_le_bytes())?;
        }

        // Patch header: num_records at byte 8, index_offset at byte 16.
        self.file.seek(SeekFrom::Start(8))?;
        self.file.write_all(&num_records.to_le_bytes())?;
        self.file.write_all(&index_offset.to_le_bytes())?;

        self.file.flush()?;
        Ok(())
    }

    /// Write a single tensor (meta + data + padding) and return bytes written.
    fn write_tensor(&mut self, tensor: &Tensor) -> io::Result<u64> {
        let shape = tensor.shape();
        let ndim = shape.len() as u32;
        let dtype = tensor.dtype();
        let dtype_tag = dtype_to_tag(dtype);
        let numel: usize = shape.iter().product();

        let mut written: u64 = 0;

        // Meta: ndim (u32) + shape (ndim * u32) + dtype (u32).
        self.file.write_all(&ndim.to_le_bytes())?;
        written += 4;
        for &dim in shape {
            self.file.write_all(&(dim as u32).to_le_bytes())?;
            written += 4;
        }
        self.file.write_all(&dtype_tag.to_le_bytes())?;
        written += 4;

        // Data: raw bytes.
        let data_byte_size = match dtype {
            DType::F32 => {
                let guard = tensor.storage.data();
                let bytes: &[u8] = bytemuck::cast_slice(&*guard);
                self.file.write_all(bytes)?;
                bytes.len()
            }
            DType::F16 => {
                // Preserve native F16 bytes.
                #[cfg(feature = "gpu")]
                {
                    let raw = tensor.storage.download_raw_bytes();
                    // download_raw_bytes may include alignment padding;
                    // only write the exact data bytes (numel * 2).
                    let exact = numel * 2;
                    self.file.write_all(&raw[..exact])?;
                    exact
                }
                #[cfg(not(feature = "gpu"))]
                {
                    panic!("F16 tensor serialization requires GPU feature");
                }
            }
            DType::Q8 { .. } => unreachable!("Q8 blocked above"),
        };
        written += data_byte_size as u64;

        // Padding to 4-byte alignment.
        let padding = (4 - (data_byte_size % 4)) % 4;
        if padding > 0 {
            self.file.write_all(&vec![0u8; padding])?;
            written += padding as u64;
        }

        Ok(written)
    }
}

// ---------------------------------------------------------------------------
// RecordDataset (mmap reader)
// ---------------------------------------------------------------------------

/// Memory-mapped reader for `.rrec` files.
///
/// Implements [`Dataset`] for integration with [`DataLoader`](crate::data::DataLoader).
/// Worker threads can call `get(index)` concurrently — each reads from
/// different pages of the mmap with zero contention.
pub struct RecordDataset {
    mmap: memmap2::Mmap,
    index: Vec<IndexEntry>,
    num_records: usize,
}

// Safety: Mmap is Send + Sync.  IndexEntry is plain data.
// RecordDataset satisfies Dataset: Send + Sync.

impl RecordDataset {
    /// Open a `.rrec` file for reading via memory mapping.
    pub fn open(path: &str) -> io::Result<Self> {
        let file = File::open(path)?;

        // Safety: the file is opened read-only and we don't mutate the mmap.
        // This is the standard memmap2 usage pattern.
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        // Validate header.
        if mmap.len() < HEADER_SIZE as usize {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "file too small for rrec header"));
        }
        if &mmap[0..4] != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid rrec magic bytes"));
        }
        let version = u32::from_le_bytes(mmap[4..8].try_into().unwrap());
        if version != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported rrec version: {}", version),
            ));
        }

        let num_records = u64::from_le_bytes(mmap[8..16].try_into().unwrap()) as usize;
        let index_offset = u64::from_le_bytes(mmap[16..24].try_into().unwrap()) as usize;

        // Parse the index table.
        let index_size = num_records * 16;
        if index_offset + index_size > mmap.len() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "index extends beyond file"));
        }

        let mut index = Vec::with_capacity(num_records);
        for i in 0..num_records {
            let base = index_offset + i * 16;
            let offset = u64::from_le_bytes(mmap[base..base + 8].try_into().unwrap());
            let length = u64::from_le_bytes(mmap[base + 8..base + 16].try_into().unwrap());
            index.push(IndexEntry { offset, length });
        }

        Ok(Self { mmap, index, num_records })
    }
}

impl Dataset for RecordDataset {
    fn len(&self) -> usize {
        self.num_records
    }

    fn get(&self, index: usize) -> DataItem {
        assert!(index < self.num_records, "RecordDataset: index {} >= len {}", index, self.num_records);
        let entry = &self.index[index];
        let start = entry.offset as usize;
        let end = start + entry.length as usize;
        let record_bytes = &self.mmap[start..end];
        parse_record(record_bytes)
    }
}

// ---------------------------------------------------------------------------
// Record parsing
// ---------------------------------------------------------------------------

/// Parse a complete record (input + target) from raw bytes.
fn parse_record(bytes: &[u8]) -> DataItem {
    let mut cursor = 0;

    let (input, consumed) = parse_tensor(bytes, cursor);
    cursor += consumed;

    let (target, _) = parse_tensor(bytes, cursor);

    DataItem { input, target }
}

/// Parse a single tensor (meta + data + padding) and return (tensor, bytes_consumed).
fn parse_tensor(bytes: &[u8], start: usize) -> (Tensor, usize) {
    let mut cursor = start;

    // Read ndim.
    let ndim = read_u32(bytes, cursor) as usize;
    cursor += 4;

    // Read shape.
    let mut shape = Vec::with_capacity(ndim);
    for _ in 0..ndim {
        shape.push(read_u32(bytes, cursor) as usize);
        cursor += 4;
    }

    // Read dtype tag.
    let dtype_tag = read_u32(bytes, cursor);
    cursor += 4;

    let numel: usize = shape.iter().product();

    let tensor = match dtype_tag {
        DTYPE_TAG_F32 => {
            let byte_len = numel * 4;
            let data_bytes = &bytes[cursor..cursor + byte_len];
            let f32_slice: &[f32] = bytemuck::cast_slice(data_bytes);
            cursor += byte_len;
            // Pad to 4-byte alignment (F32 data is always aligned, but be explicit).
            cursor += (4 - (byte_len % 4)) % 4;
            Tensor::new(f32_slice.to_vec(), shape)
        }
        DTYPE_TAG_F16 => {
            let byte_len = numel * 2;
            let data_bytes = &bytes[cursor..cursor + byte_len];
            // Convert f16 bits (u16) to f32 on CPU.
            let u16_slice: &[u16] = bytemuck::cast_slice(data_bytes);
            let f32_data: Vec<f32> = u16_slice.iter().map(|&bits| f16_to_f32(bits)).collect();
            cursor += byte_len;
            // Pad to 4-byte alignment.
            cursor += (4 - (byte_len % 4)) % 4;
            Tensor::new(f32_data, shape)
        }
        _ => panic!("unknown dtype tag in rrec record: {}", dtype_tag),
    };

    (tensor, cursor - start)
}

/// Read a little-endian u32 from a byte slice at the given offset.
#[inline]
fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

/// Convert IEEE 754 half-precision (f16) bits to f32.
///
/// Standard bit manipulation — no external crate needed.
#[inline]
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exponent = ((bits >> 10) & 0x1F) as u32;
    let mantissa = (bits & 0x3FF) as u32;

    if exponent == 0 {
        if mantissa == 0 {
            // Zero (positive or negative).
            f32::from_bits(sign << 31)
        } else {
            // Subnormal f16 → normalized f32.
            let mut m = mantissa;
            let mut e: i32 = -14; // f16 subnormal exponent bias
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF; // remove implicit leading 1
            let f32_exp = ((e + 127) as u32) & 0xFF;
            f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13))
        }
    } else if exponent == 31 {
        // Inf or NaN.
        let f32_mantissa = mantissa << 13;
        f32::from_bits((sign << 31) | (0xFF << 23) | f32_mantissa)
    } else {
        // Normalized.
        let f32_exp = (exponent as i32 - 15 + 127) as u32;
        f32::from_bits((sign << 31) | (f32_exp << 23) | (mantissa << 13))
    }
}
