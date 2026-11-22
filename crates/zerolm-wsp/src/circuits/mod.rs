//! Circuit templates. Each one turns typed parameters into a program and its
//! constants; none may contain language data (constitution invariant 11).

pub mod atom_encode;
pub mod decide;
pub mod attention_read;
pub mod key_encode;
pub mod lexer;
pub mod row_lookup;
pub mod rewrite;
pub mod memory_write;
pub mod number;
pub mod probe;
pub mod relational;
pub mod value_codec;

pub mod token_number;
pub mod token_features;
pub mod question_tail;
pub mod template_match;
