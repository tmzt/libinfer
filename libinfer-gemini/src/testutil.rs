//! Test helpers shared by the crate's tests: a local HTTP server, so no
//! test calls the real API, and a captured logger (`log` takes one logger
//! per process, so every test module uses this one).

use std::io::{Read, Write};
use std::sync::{Arc, Mutex, Once};
use std::thread::{JoinHandle, ThreadId};

/// A local HTTP server that answers each request with the next of
/// `responses` (status, body) and records each request (head and body).
pub(crate) struct TestServer {
    pub(crate) base: String,
    requests: Arc<Mutex<Vec<String>>>,
    thread: Option<JoinHandle<()>>,
}

impl TestServer {
    pub(crate) fn start(responses: Vec<(u16, String)>) -> TestServer {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}/v1beta", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let seen = requests.clone();
        let thread = std::thread::spawn(move || {
            for (status, body) in responses {
                let (mut stream, _) = listener.accept().unwrap();
                let mut head = Vec::new();
                let mut byte = [0u8; 1];
                while !head.ends_with(b"\r\n\r\n") {
                    if stream.read(&mut byte).unwrap() == 0 {
                        break;
                    }
                    head.push(byte[0]);
                }
                let head = String::from_utf8_lossy(&head).into_owned();
                let length = head
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().ok())?
                    })
                    .unwrap_or(0);
                let mut request_body = vec![0u8; length];
                stream.read_exact(&mut request_body).unwrap();
                seen.lock()
                    .unwrap()
                    .push(head + &String::from_utf8_lossy(&request_body));
                let reply = format!(
                    "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\n\
                     Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                stream.write_all(reply.as_bytes()).unwrap();
            }
        });
        TestServer {
            base,
            requests,
            thread: Some(thread),
        }
    }

    /// Every request, once the server has answered all its responses.
    pub(crate) fn requests(mut self) -> Vec<String> {
        self.thread.take().unwrap().join().unwrap();
        self.requests.lock().unwrap().clone()
    }
}

/// A logger keeping every record with the thread that logged it, so tests
/// running at once each see only their own lines.
struct CapturedLogger {
    records: Mutex<Vec<(ThreadId, log::Level, String)>>,
}

impl log::Log for CapturedLogger {
    fn enabled(&self, _: &log::Metadata<'_>) -> bool {
        true
    }

    fn log(&self, record: &log::Record<'_>) {
        self.records.lock().unwrap().push((
            std::thread::current().id(),
            record.level(),
            record.args().to_string(),
        ));
    }

    fn flush(&self) {}
}

static LOGGER: CapturedLogger = CapturedLogger {
    records: Mutex::new(Vec::new()),
};

/// The records this thread logs while `f` runs.
pub(crate) fn captured(f: impl FnOnce()) -> Vec<(log::Level, String)> {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        log::set_logger(&LOGGER).unwrap();
        log::set_max_level(log::LevelFilter::Trace);
    });
    let me = std::thread::current().id();
    LOGGER.records.lock().unwrap().retain(|(id, ..)| *id != me);
    f();
    LOGGER
        .records
        .lock()
        .unwrap()
        .iter()
        .filter(|(id, ..)| *id == me)
        .map(|(_, level, text)| (*level, text.clone()))
        .collect()
}
