// Licensed to the Apache Software Foundation (ASF) under one or more
// contributor license agreements. See the NOTICE file distributed with
// this work for additional information regarding copyright ownership.
// The ASF licenses this file to You under the Apache License, Version 2.0
// (the "License"); you may not use this file except in compliance with
// the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This package is based on the work done by Keiron Liddle, Aftex Software
// <keiron@aftexsw.com> to whom the Ant project is very grateful for his
// great code.

//! BZip2 compression and decompression.
//!
//! Port of `util/bzip2/` from bc-csharp.
//!
//! ## Example
//!
//! ```rust
//! use std::io::{Read, Write};
//! use bc_rust::util::bzip2::{BZip2Reader, BZip2Writer};
//!
//! // Compress
//! let mut compressed = Vec::new();
//! {
//!     let mut writer = BZip2Writer::new(&mut compressed).unwrap();
//!     writer.write_all(b"Hello, world!").unwrap();
//!     writer.finish().unwrap();
//! }
//!
//! // Decompress
//! let mut reader = BZip2Reader::new(compressed.as_slice()).unwrap();
//! let mut output = Vec::new();
//! reader.read_to_end(&mut output).unwrap();
//!
//! assert_eq!(output, b"Hello, world!");
//! ```
//!
//! ## Public API
//!
//! | Type | Description |
//! |------|-------------|
//! | [`BZip2Reader`] | Decompression — wraps any [`std::io::Read`] |
//! | [`BZip2Writer`] | Compression — wraps any [`std::io::Write`] |
//!
//! ## bc-csharp mapping
//!
//! | bc-csharp | bc-rust |
//! |-----------|---------|
//! | `CBZip2InputStream.cs` | [`BZip2Reader`] |
//! | `CBZip2OutputStream.cs` | [`BZip2Writer`] |

pub(super) mod bzip2_reader;
pub(super) mod bzip2_writer;
pub(super) mod constants;
pub(super) mod crc;

pub use bzip2_reader::BZip2Reader;
pub use bzip2_writer::BZip2Writer;
