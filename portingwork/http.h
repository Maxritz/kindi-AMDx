// http.h - minimal HTTP/1.1 server for the kindi OpenAI-compatible API.
//
// Winsock, one connection at a time (the engine is single-slot: a 5.9 GB model on an
// 8 GB card has no room for a second context). No TLS, no chunked *request* bodies
// (OpenAI clients send Content-Length), chunked *responses* for SSE streaming.
#pragma once
#include <winsock2.h>
#include <ws2tcpip.h>
#include <string>
#include <map>
#include <vector>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <algorithm>

namespace kindi_http {

struct Req {
    std::string method;
    std::string path;                      // without query string
    std::string query;
    std::map<std::string, std::string> hdr; // lower-case keys
    std::string body;
};

inline std::string lower(std::string s) {
    for (auto& c : s) c = (char)tolower((unsigned char)c);
    return s;
}

inline std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

// Read one request. Returns false on a clean close / error.
inline bool read_req(SOCKET s, Req& r) {
    std::string buf;
    char tmp[8192];
    size_t hdr_end = std::string::npos;
    // read until the end of the header block
    while ((hdr_end = buf.find("\r\n\r\n")) == std::string::npos) {
        int n = recv(s, tmp, sizeof(tmp), 0);
        if (n <= 0) return false;
        buf.append(tmp, (size_t)n);
        if (buf.size() > (1u << 20)) return false;   // header too large
    }
    std::string head = buf.substr(0, hdr_end);
    std::string rest = buf.substr(hdr_end + 4);

    size_t sp1 = head.find(' ');
    size_t sp2 = (sp1 == std::string::npos) ? std::string::npos : head.find(' ', sp1 + 1);
    if (sp1 == std::string::npos || sp2 == std::string::npos) return false;
    r.method = head.substr(0, sp1);
    std::string target = head.substr(sp1 + 1, sp2 - sp1 - 1);

    size_t q = target.find('?');
    if (q == std::string::npos) {
        r.path = target;
    } else {
        r.path = target.substr(0, q);
        r.query = target.substr(q + 1);
    }

    size_t pos = sp2 + 1;
    while (pos < head.size()) {
        size_t eol = head.find("\r\n", pos);
        if (eol == std::string::npos) eol = head.size();
        std::string line = head.substr(pos, eol - pos);
        size_t c = line.find(':');
        if (c != std::string::npos) {
            r.hdr[lower(trim(line.substr(0, c)))] = trim(line.substr(c + 1));
        }
        pos = eol + 2;
    }

    size_t want = 0;
    auto it = r.hdr.find("content-length");
    if (it != r.hdr.end()) want = (size_t)strtoull(it->second.c_str(), nullptr, 10);
    while (rest.size() < want) {
        int n = recv(s, tmp, sizeof(tmp), 0);
        if (n <= 0) return false;
        rest.append(tmp, (size_t)n);
    }
    r.body = rest.substr(0, want);
    return true;
}

inline void send_all(SOCKET s, const char* p, size_t n) {
    size_t off = 0;
    while (off < n) {
        int w = send(s, p + off, (int)(n - off), 0);
        if (w <= 0) return;
        off += (size_t)w;
    }
}

inline void send_str(SOCKET s, const std::string& s2) { send_all(s, s2.data(), s2.size()); }

inline const char* status_text(int code) {
    switch (code) {
        case 200: return "OK";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 413: return "Payload Too Large";
        case 500: return "Internal Server Error";
        case 503: return "Service Unavailable";
        default:  return "OK";
    }
}

// Non-streaming response.
inline void respond(SOCKET s, int code, const std::string& ctype, const std::string& body) {
    std::string h = "HTTP/1.1 " + std::to_string(code) + " " + status_text(code) + "\r\n";
    h += "Content-Type: " + ctype + "\r\n";
    h += "Content-Length: " + std::to_string(body.size()) + "\r\n";
    h += "Access-Control-Allow-Origin: *\r\n";
    h += "Connection: close\r\n\r\n";
    send_str(s, h);
    send_str(s, body);
}

inline void respond_json(SOCKET s, int code, const std::string& body) {
    respond(s, code, "application/json", body);
}

inline void respond_err(SOCKET s, int code, const std::string& msg) {
    // OpenAI's error envelope, so client libraries surface the message.
    std::string body = "{\"error\":{\"message\":\"" + msg + "\",\"type\":\"invalid_request_error\",\"code\":"
                       + std::to_string(code) + "}}";
    respond_json(s, code, body);
}

// ---- Server-sent events, chunked so the length need not be known up front ----
struct Sse {
    SOCKET s = INVALID_SOCKET;
    bool open = false;

    void begin(const char* ctype = "text/event-stream") {
        std::string h = "HTTP/1.1 200 OK\r\n";
        h += std::string("Content-Type: ") + ctype + "\r\n";
        h += "Cache-Control: no-cache\r\n";
        h += "Transfer-Encoding: chunked\r\n";
        h += "Access-Control-Allow-Origin: *\r\n";
        h += "Connection: close\r\n\r\n";
        send_str(s, h);
        open = true;
    }
    // one SSE frame: "data: <payload>\n\n"
    void data(const std::string& payload) {
        if (!open) return;
        std::string frame = "data: " + payload + "\n\n";
        char len[32];
        int n = snprintf(len, sizeof(len), "%zX\r\n", frame.size());
        send_all(s, len, (size_t)n);
        send_str(s, frame);
        send_str(s, "\r\n");
    }
    void done() {
        if (!open) return;
        send_str(s, "0\r\n\r\n");
        open = false;
    }
};

// JSON string escaping (control chars, quotes, backslash). UTF-8 passes through.
inline std::string json_escape(const std::string& in) {
    std::string o;
    o.reserve(in.size() + 8);
    for (unsigned char c : in) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            case '\b': o += "\\b";  break;
            case '\f': o += "\\f";  break;
            default:
                if (c < 0x20) {
                    char b[8];
                    snprintf(b, sizeof(b), "\\u%04x", c);
                    o += b;
                } else {
                    o += (char)c;
                }
        }
    }
    return o;
}

inline SOCKET listen_on(const char* host, int port) {
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        fprintf(stderr, "http: WSAStartup failed\n");
        return INVALID_SOCKET;
    }
    SOCKET ls = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (ls == INVALID_SOCKET) return INVALID_SOCKET;
    int yes = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, (const char*)&yes, sizeof(yes));
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons((u_short)port);
    if (!host || !*host || !strcmp(host, "0.0.0.0")) {
        a.sin_addr.s_addr = INADDR_ANY;
    } else if (inet_pton(AF_INET, host, &a.sin_addr) != 1) {
        fprintf(stderr, "http: bad host '%s'\n", host);
        closesocket(ls);
        return INVALID_SOCKET;
    }
    if (bind(ls, (sockaddr*)&a, sizeof(a)) != 0) {
        fprintf(stderr, "http: bind %s:%d failed (%d)\n", host ? host : "0.0.0.0", port, WSAGetLastError());
        closesocket(ls);
        return INVALID_SOCKET;
    }
    if (listen(ls, 8) != 0) {
        closesocket(ls);
        return INVALID_SOCKET;
    }
    return ls;
}

} // namespace kindi_http
