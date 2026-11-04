//! A deliberately small HTTP/1.1 layer on top of `std::net`, for `zerolm serve`.
//!
//! It handles one request per connection (`Connection: close`), requires
//! `Content-Length` for bodies, and refuses everything it does not need
//! (chunked bodies, folded headers, oversized requests). All limits are
//! constants here so they are easy to audit.

use std::io::{self, ErrorKind, Read, Write};

pub const MAX_HEADER_BYTES: usize = 8 * 1024;
pub const MAX_HEADERS: usize = 64;
pub const MAX_BODY_BYTES: usize = 64 * 1024;

#[derive(Debug, Clone, PartialEq)]
pub struct Request {
    pub method: String,
    /// Path without the query string, still percent-encoded.
    pub path: String,
    pub query: String,
    /// Header names are lower-cased.
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Request {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
    }
}

/// An HTTP error to send back instead of routing: (status, short message).
pub type HttpError = (u16, &'static str);

pub fn reason(status: u16) -> &'static str {
    match status {
        100 => "Continue",
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        405 => "Method Not Allowed",
        408 => "Request Timeout",
        409 => "Conflict",
        413 => "Payload Too Large",
        431 => "Request Header Fields Too Large",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        503 => "Service Unavailable",
        505 => "HTTP Version Not Supported",
        _ => "Unknown",
    }
}

/// Decode `%XX` escapes. Returns `None` for a malformed escape or non-UTF-8 result.
/// `+` is left alone (this is a path decoder, not a form decoder).
pub fn percent_decode(s: &str) -> Option<String> {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' {
            let hex = s.get(i + 1..i + 3)?;
            out.push(u8::from_str_radix(hex, 16).ok()?);
            i += 3;
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

fn io_error(e: &io::Error) -> HttpError {
    match e.kind() {
        ErrorKind::WouldBlock | ErrorKind::TimedOut => (408, "timed out waiting for the request"),
        _ => (400, "connection error while reading the request"),
    }
}

fn find_header_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

/// Read and validate one request. `stream` is also written to, only to send
/// `100 Continue` when the client asks for it.
pub fn read_request<S: Read + Write>(stream: &mut S, max_body: usize) -> Result<Request, HttpError> {
    let mut buf: Vec<u8> = Vec::with_capacity(1024);
    let mut chunk = [0u8; 1024];
    let header_end = loop {
        if let Some(pos) = find_header_end(&buf) {
            break pos;
        }
        if buf.len() > MAX_HEADER_BYTES {
            return Err((431, "request headers too large"));
        }
        let n = stream.read(&mut chunk).map_err(|e| io_error(&e))?;
        if n == 0 {
            return Err((400, "connection closed before the request was complete"));
        }
        buf.extend_from_slice(&chunk[..n]);
    };
    if header_end > MAX_HEADER_BYTES {
        return Err((431, "request headers too large"));
    }
    let head = std::str::from_utf8(&buf[..header_end]).map_err(|_| (400, "headers must be UTF-8"))?;
    let mut lines = head.split("\r\n");

    // Request line.
    let line = lines.next().unwrap_or("");
    let mut parts = line.split(' ');
    let (method, target, version) = match (parts.next(), parts.next(), parts.next(), parts.next()) {
        (Some(m), Some(t), Some(v), None) if !m.is_empty() && !t.is_empty() => (m, t, v),
        _ => return Err((400, "malformed request line")),
    };
    if !method.bytes().all(|c| c.is_ascii_uppercase()) {
        return Err((400, "malformed method"));
    }
    if !version.starts_with("HTTP/1.") {
        return Err((505, "only HTTP/1.x is supported"));
    }
    if !target.starts_with('/') || target.bytes().any(|c| c <= b' ' || c == 0x7f) {
        return Err((400, "malformed request target"));
    }
    let (path, query) = match target.split_once('?') {
        Some((p, q)) => (p.to_string(), q.to_string()),
        None => (target.to_string(), String::new()),
    };

    // Headers.
    let mut headers: Vec<(String, String)> = Vec::new();
    for line in lines {
        if headers.len() >= MAX_HEADERS {
            return Err((431, "too many headers"));
        }
        if line.starts_with(' ') || line.starts_with('\t') {
            return Err((400, "folded headers are not supported"));
        }
        let (name, value) = line.split_once(':').ok_or((400, "malformed header"))?;
        if name.is_empty() || !name.bytes().all(|c| c.is_ascii_graphic() && c != b':') {
            return Err((400, "malformed header name"));
        }
        headers.push((name.to_ascii_lowercase(), value.trim().to_string()));
    }

    if headers.iter().any(|(k, _)| k == "transfer-encoding") {
        return Err((501, "chunked or encoded request bodies are not supported; send Content-Length"));
    }
    let lengths: Vec<&str> =
        headers.iter().filter(|(k, _)| k == "content-length").map(|(_, v)| v.as_str()).collect();
    let content_length = match lengths.as_slice() {
        [] => 0,
        [one] if !one.is_empty() && one.bytes().all(|c| c.is_ascii_digit()) && one.len() <= 9 => {
            one.parse::<usize>().map_err(|_| (400, "bad Content-Length"))?
        }
        _ => return Err((400, "bad or repeated Content-Length")),
    };
    if content_length > max_body {
        return Err((413, "request body too large"));
    }

    if content_length > 0
        && headers.iter().any(|(k, v)| k == "expect" && v.eq_ignore_ascii_case("100-continue"))
    {
        stream.write_all(b"HTTP/1.1 100 Continue\r\n\r\n").map_err(|e| io_error(&e))?;
    }

    // Body: whatever followed the headers, then the rest from the stream.
    let mut body: Vec<u8> = buf[header_end + 4..].to_vec();
    if body.len() > content_length {
        return Err((400, "more data than Content-Length"));
    }
    while body.len() < content_length {
        let want = (content_length - body.len()).min(chunk.len());
        let n = stream.read(&mut chunk[..want]).map_err(|e| io_error(&e))?;
        if n == 0 {
            return Err((400, "connection closed before the body was complete"));
        }
        body.extend_from_slice(&chunk[..n]);
    }

    Ok(Request { method: method.to_string(), path, query, headers, body })
}

/// Write a complete response and flush. `extra` headers are written verbatim,
/// so callers must only pass fixed, trusted values.
pub fn write_response<W: Write>(
    w: &mut W,
    status: u16,
    content_type: &str,
    extra: &[(&str, &str)],
    body: &str,
) -> io::Result<()> {
    let mut head = format!(
        "HTTP/1.1 {status} {}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n\
Connection: close\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n",
        reason(status),
        body.len()
    );
    for (k, v) in extra {
        head.push_str(&format!("{k}: {v}\r\n"));
    }
    head.push_str("\r\n");
    w.write_all(head.as_bytes())?;
    w.write_all(body.as_bytes())?;
    w.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// A fake connection: reads come from `input`, writes are collected.
    struct Conn {
        input: Cursor<Vec<u8>>,
        output: Vec<u8>,
    }
    impl Read for Conn {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            self.input.read(buf)
        }
    }
    impl Write for Conn {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.output.extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    fn parse(raw: &str) -> Result<Request, HttpError> {
        read_request(&mut Conn { input: Cursor::new(raw.as_bytes().to_vec()), output: vec![] }, 1024)
    }

    #[test]
    fn parses_a_normal_request_with_body_and_query() {
        let r = parse("POST /v1/ask?x=1 HTTP/1.1\r\nHost: a\r\nContent-Length: 4\r\nAuthorization: Bearer t\r\n\r\nbody")
            .unwrap();
        assert_eq!((r.method.as_str(), r.path.as_str(), r.query.as_str()), ("POST", "/v1/ask", "x=1"));
        assert_eq!(r.body, b"body");
        assert_eq!(r.header("authorization"), Some("Bearer t"));
    }

    #[test]
    fn rejects_malformed_and_dangerous_requests() {
        let cases: [(&str, u16); 14] = [
            ("GET /x\r\n\r\n", 400),                                           // no version
            ("GET /x HTTP/2\r\n\r\n", 505),                                    // version
            ("get /x HTTP/1.1\r\n\r\n", 400),                                  // lower-case method
            ("GET x HTTP/1.1\r\n\r\n", 400),                                   // target without slash
            ("GET /a b HTTP/1.1\r\n\r\n", 400),                                // space in target
            ("GET /x HTTP/1.1\r\nBad Header\r\n\r\n", 400),                    // no colon
            ("GET /x HTTP/1.1\r\n folded: x\r\n\r\n", 400),                    // obs-fold
            ("POST /x HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n", 501),   // chunked
            ("POST /x HTTP/1.1\r\nContent-Length: 5\r\nContent-Length: 5\r\n\r\nhello", 400), // repeated
            ("POST /x HTTP/1.1\r\nContent-Length: -1\r\n\r\n", 400),
            ("POST /x HTTP/1.1\r\nContent-Length: 99999999999\r\n\r\n", 400),
            ("POST /x HTTP/1.1\r\nContent-Length: 2000\r\n\r\n", 413),
            ("POST /x HTTP/1.1\r\nContent-Length: 10\r\n\r\nshort", 400),      // truncated body
            ("POST /x HTTP/1.1\r\nContent-Length: 2\r\n\r\ntoolong", 400),     // surplus data
        ];
        for (raw, want) in cases {
            assert_eq!(parse(raw).unwrap_err().0, want, "{raw:?}");
        }
    }

    #[test]
    fn caps_header_size_and_count() {
        let big = format!("GET /x HTTP/1.1\r\nX: {}\r\n\r\n", "a".repeat(MAX_HEADER_BYTES + 10));
        assert_eq!(parse(&big).unwrap_err().0, 431);
        let many: String = (0..MAX_HEADERS + 1).map(|i| format!("H{i}: v\r\n")).collect();
        assert_eq!(parse(&format!("GET /x HTTP/1.1\r\n{many}\r\n")).unwrap_err().0, 431);
        // headers that never end
        assert_eq!(parse(&format!("GET /x HTTP/1.1\r\nX: {}", "a".repeat(20_000))).unwrap_err().0, 431);
    }

    #[test]
    fn sends_100_continue_when_asked() {
        let mut c = Conn {
            input: Cursor::new(b"POST /x HTTP/1.1\r\nExpect: 100-continue\r\nContent-Length: 2\r\n\r\nhi".to_vec()),
            output: vec![],
        };
        assert_eq!(read_request(&mut c, 1024).unwrap().body, b"hi");
        assert!(String::from_utf8(c.output).unwrap().starts_with("HTTP/1.1 100 Continue"));
    }

    #[test]
    fn percent_decoding() {
        assert_eq!(percent_decode("my%20name").unwrap(), "my name");
        assert_eq!(percent_decode("caf%C3%A9").unwrap(), "caf\u{e9}");
        assert_eq!(percent_decode("a+b").unwrap(), "a+b");
        for bad in ["%", "%2", "%zz", "%FF", "%C3"] {
            assert!(percent_decode(bad).is_none(), "{bad}");
        }
    }

    #[test]
    fn response_has_exact_length_and_security_headers() {
        let mut out = Vec::new();
        write_response(&mut out, 200, "application/json", &[("Allow", "GET")], "{\"a\":1}").unwrap();
        let s = String::from_utf8(out).unwrap();
        assert!(s.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(s.contains("Content-Length: 7\r\n") && s.contains("Connection: close\r\n"));
        assert!(s.contains("X-Content-Type-Options: nosniff") && s.ends_with("\r\n\r\n{\"a\":1}"));
    }
}
