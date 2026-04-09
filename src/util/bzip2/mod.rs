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
//! ## bc-csharp mapping
//!
//! | bc-csharp | bc-rust | Notes |
//! |-----------|---------|-------|
//! | `BZip2Constants.cs` | [`constants`] | Shared constants |
//! | `CRC.cs` | [`crc`] | CRC-32 calculation |
//! | `CBZip2InputStream.cs` | [`bzip2_reader`] | Decompression reader |
//! | `CBZip2OutputStream.cs` | [`bzip2_writer`] | Compression writer |

pub mod bzip2_reader;
pub mod bzip2_writer;
pub(super) mod constants;
pub(super) mod crc;
