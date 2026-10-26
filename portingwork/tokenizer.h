// tokenizer.h — port of kindi/tokenizer.py (Qwen3.5 byte-level BPE, llama.cpp-exact).
// Pre-tokenizer regex, BPE merge loop, added-token splitting, encode/decode.
// Header-only. Requires <nlohmann/json.hpp> include path for vocab.json loading.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <unordered_map>

namespace kindi_tok {

// ---- generated Unicode category tables (from CPython unicodedata) ----
#include "unitbl.inc"  // LT[][], MT[][], NT[][], SPT[][] pairs {lo,hi}

static bool in_ranges(const uint32_t t[][2], size_t n, uint32_t cp) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t m = (lo + hi) / 2;
        if (cp < t[m][0]) hi = m;
        else if (cp > t[m][1]) lo = m + 1;
        else return true;
    }
    return false;
}
static inline bool is_l(uint32_t c)  { return in_ranges(LT,  sizeof(LT)/sizeof(LT[0]),  c); }
static inline bool is_m(uint32_t c)  { return in_ranges(MT,  sizeof(MT)/sizeof(MT[0]),  c); }
static inline bool is_lm(uint32_t c) { return is_l(c) || is_m(c); }
static inline bool is_n(uint32_t c)  { return in_ranges(NT,  sizeof(NT)/sizeof(NT[0]),  c); }
static inline bool is_sp(uint32_t c) { return in_ranges(SPT, sizeof(SPT)/sizeof(SPT[0]), c); }  // == Py str.isspace
static inline bool is_other(uint32_t c) { return !(is_sp(c) || is_lm(c) || is_n(c)); }

// ---- UTF-8 helpers (operate on codepoints like Python str) ----
static void utf8_decode(const std::string& s, std::vector<uint32_t>& cps, std::vector<int>& offs) {
    cps.clear(); offs.clear();
    size_t i = 0, n = s.size();
    while (i < n) {
        offs.push_back((int)i);
        unsigned char b = (unsigned char)s[i];
        uint32_t cp; size_t len;
        if (b < 0x80) { cp = b; len = 1; }
        else if ((b >> 5) == 0x6 && i + 1 < n) { cp = (b & 0x1F); len = 2; }
        else if ((b >> 4) == 0xE && i + 2 < n) { cp = (b & 0x0F); len = 3; }
        else if ((b >> 3) == 0x1E && i + 3 < n) { cp = (b & 0x07); len = 4; }
        else { cp = 0xFFFD; len = 1; cps.push_back(cp); i += 1; continue; }
        for (size_t k = 1; k < len; ++k) {
            unsigned char c = (unsigned char)s[i + k];
            if ((c >> 6) != 0x2) { cp = 0xFFFD; len = k; break; }
            cp = (cp << 6) | (c & 0x3F);
        }
        cps.push_back(cp); i += len;
    }
    offs.push_back((int)n);
}
static void utf8_encode_cp(uint32_t cp, std::string& out) {
    if (cp < 0x80) out.push_back((char)cp);
    else if (cp < 0x800) { out.push_back((char)(0xC0 | (cp >> 6))); out.push_back((char)(0x80 | (cp & 0x3F))); }
    else if (cp < 0x10000) { out.push_back((char)(0xE0 | (cp >> 12))); out.push_back((char)(0x80 | ((cp >> 6) & 0x3F))); out.push_back((char)(0x80 | (cp & 0x3F))); }
    else { out.push_back((char)(0xF0 | (cp >> 18))); out.push_back((char)(0x80 | ((cp >> 12) & 0x3F))); out.push_back((char)(0x80 | ((cp >> 6) & 0x3F))); out.push_back((char)(0x80 | (cp & 0x3F))); }
}
// Replace malformed UTF-8 with U+FFFD (== Python errors="replace" on decode).
static std::string utf8_sanitize(const std::string& s) {
    std::string out; out.reserve(s.size());
    size_t i = 0, n = s.size();
    while (i < n) {
        unsigned char b = (unsigned char)s[i];
        size_t len;
        uint32_t cp;
        if (b < 0x80) { out.push_back((char)b); ++i; continue; }
        else if ((b >> 5) == 0x6) { len = 2; cp = b & 0x1F; }
        else if ((b >> 4) == 0xE) { len = 3; cp = b & 0x0F; }
        else if ((b >> 3) == 0x1E) { len = 4; cp = b & 0x07; }
        else { out += "\xEF\xBF\xBD"; ++i; continue; }
        if (i + len > n) { out += "\xEF\xBF\xBD"; ++i; continue; }
        bool ok = true; uint32_t v = cp;
        for (size_t k = 1; k < len; ++k) {
            unsigned char c = (unsigned char)s[i + k];
            if ((c >> 6) != 0x2) { ok = false; break; }
            v = (v << 6) | (c & 0x3F);
        }
        // reject overlongs / surrogates / out-of-range (match CPython replace)
        if (ok) {
            if ((len == 2 && v < 0x80) || (len == 3 && v < 0x800) || (len == 4 && v < 0x10000) ||
                (v >= 0xD800 && v <= 0xDFFF) || v > 0x10FFFF) ok = false;
        }
        if (!ok) { out += "\xEF\xBF\xBD"; ++i; continue; }
        out.append(s, i, len); i += len;
    }
    return out;
}

// GPT-2 bytes_to_unicode tables (same construction as tokenizer.py).
struct ByteMap {
    uint32_t b2u[256];
    int32_t u2b[0x200];  // mapped cp (<0x200) -> byte, -1 if none
    ByteMap() {
        bool in_bs[256] = {false};
        for (int b = 33; b < 127; ++b) in_bs[b] = true;
        for (int b = 161; b < 173; ++b) in_bs[b] = true;
        for (int b = 174; b < 256; ++b) in_bs[b] = true;
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (in_bs[b]) b2u[b] = (uint32_t)b;
            else b2u[b] = (uint32_t)(256 + n++);
        }
        for (int i = 0; i < 0x200; ++i) u2b[i] = -1;
        for (int b = 0; b < 256; ++b) u2b[b2u[b]] = b;
    }
};

class BpeTokenizer {
  public:
    std::vector<std::string> vocab;                    // id -> token string (UTF-8)
    std::unordered_map<std::string, int> sid;          // token string -> id
    std::unordered_map<std::string, int> ranks;        // "A\0B" -> merge rank
    std::vector<std::vector<uint32_t>> added;          // added tokens as codepoints, len-desc
    std::vector<std::string> added_str;
    std::unordered_map<int, int> added_id;             // index in added[] -> vocab id
    std::unordered_map<std::string, std::vector<int>> cache;
    ByteMap bm;

    bool load(const std::string& pack_dir, std::string* err = nullptr) {
        // vocab.json: array of strings (or [str,...] lists -> use [0])
        {
            std::ifstream f(pack_dir + "/vocab.json", std::ios::binary);
            if (!f) { if (err) *err = "cannot open vocab.json"; return false; }
            std::string js((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            // minimal JSON string-array parser (vocab.json is exactly ["...",...])
            vocab.clear();
            size_t i = 0, n = js.size();
            auto skip_ws = [&]() { while (i < n && (js[i]==' '||js[i]=='\t'||js[i]=='\n'||js[i]=='\r'||js[i]==','||js[i]=='['||js[i]==']')) ++i; };
            // parse properly: expect '[' then sequence of strings/arrays
            while (i < n && js[i] != '[') ++i;
            if (i >= n) { if (err) *err = "vocab.json: no array"; return false; }
            ++i;
            std::string cur; bool in_str = false, in_arr = false; int depth = 0; bool esc = false;
            std::string acc; bool have_first = false; std::string first;
            for (; i < n; ++i) {
                char c = js[i];
                if (in_str) {
                    if (esc) {
                        if (c == 'u' && i + 4 < n) {
                            unsigned v = 0;
                            for (int k = 1; k <= 4; ++k) { char h = js[i+k]; v <<= 4; if (h>='0'&&h<='9') v|=h-'0'; else if (h>='a'&&h<='f') v|=h-'a'+10; else if (h>='A'&&h<='F') v|=h-'A'+10; }
                            std::string e; utf8_encode_cp(v, e); acc += e; i += 4;
                        } else if (c == 'n') acc += '\n'; else if (c == 't') acc += '\t';
                        else if (c == 'r') acc += '\r'; else if (c == 'b') acc += '\b';
                        else if (c == 'f') acc += '\f'; else acc += c;
                        esc = false;
                    } else if (c == '\\') esc = true;
                    else if (c == '"') {
                        in_str = false;
                        if (depth == 0) { vocab.push_back(acc); }
                        else if (depth == 1 && !have_first) { first = acc; have_first = true; }
                        acc.clear();
                    } else acc += c;
                } else {
                    if (c == '"') in_str = true;
                    else if (c == '[') { depth++; if (depth == 1) { have_first = false; first.clear(); } }
                    else if (c == ']') { if (depth == 1 && have_first) vocab.push_back(first); depth--; if (depth < 0) break; }
                }
            }
            if (vocab.empty()) { if (err) *err = "vocab.json: parsed 0 tokens"; return false; }
        }
        sid.reserve(vocab.size() * 2);
        for (size_t i = 0; i < vocab.size(); ++i) {
            if (sid.find(vocab[i]) == sid.end()) sid[vocab[i]] = (int)i;  // first wins (no dups in pack)
        }
        // merges.txt: lines 'A B' -> rank (rank++ per non-empty line, like load_ranks)
        {
            std::ifstream f(pack_dir + "/merges.txt", std::ios::binary);
            if (!f) { if (err) *err = "cannot open merges.txt"; return false; }
            std::string line; int r = 0;
            while (std::getline(f, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (line.empty()) continue;
                // split on ' ' exactly (Python str.split(" "))
                std::vector<std::string> parts; size_t s = 0;
                for (size_t k = 0; k <= line.size(); ++k)
                    if (k == line.size() || line[k] == ' ') { parts.push_back(line.substr(s, k - s)); s = k + 1; }
                if (parts.size() == 2) ranks[parts[0] + '\0' + parts[1]] = r;
                ++r;
            }
        }
        // added tokens: vocab[248044:] entries that look <...> (same rule as Tok)
        {
            std::vector<std::pair<std::string,int>> tmp;
            for (size_t i = 248044; i < vocab.size(); ++i) {
                const std::string& s = vocab[i];
                if (s.size() >= 2 && s.front() == '<' && s.back() == '>')
                    tmp.push_back({s, (int)i});
            }
            std::sort(tmp.begin(), tmp.end(),
                      [](const auto& a, const auto& b){ return a.first.size() > b.first.size(); });
            for (auto& kv : tmp) {
                std::vector<uint32_t> cps; std::vector<int> offs;
                utf8_decode(kv.first, cps, offs);
                added_id[(int)added.size()] = kv.second;
                added.push_back(std::move(cps));
                added_str.push_back(kv.first);
            }
        }
        return true;
    }

    bool load_from_strings(const std::vector<std::string>& tokens_vec, const std::vector<std::string>& merges_vec, std::string* err = nullptr) {
        vocab = tokens_vec;
        sid.clear();
        ranks.clear();
        added.clear();
        added_str.clear();
        added_id.clear();
        cache.clear();
        sid.reserve(vocab.size() * 2);
        for (size_t i = 0; i < vocab.size(); ++i) {
            if (sid.find(vocab[i]) == sid.end()) sid[vocab[i]] = (int)i;
        }
        int r = 0;
        for (const auto& m : merges_vec) {
            std::string line = m;
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) { continue; }
            std::vector<std::string> parts; size_t s = 0;
            for (size_t k = 0; k <= line.size(); ++k)
                if (k == line.size() || line[k] == ' ') { parts.push_back(line.substr(s, k - s)); s = k + 1; }
            if (parts.size() == 2) ranks[parts[0] + '\0' + parts[1]] = r;
            ++r;
        }
        // added tokens
        {
            std::vector<std::pair<std::string,int>> tmp;
            for (size_t i = 248044; i < vocab.size(); ++i) {
                const std::string& s = vocab[i];
                if (s.size() >= 2 && s.front() == '<' && s.back() == '>') tmp.push_back({s, (int)i});
            }
            std::sort(tmp.begin(), tmp.end(),
                      [](const auto& a, const auto& b){ return a.first.size() > b.first.size(); });
            for (auto& kv : tmp) {
                std::vector<uint32_t> cps; std::vector<int> offs;
                utf8_decode(kv.first, cps, offs);
                added_id[(int)added.size()] = kv.second;
                added.push_back(std::move(cps));
                added_str.push_back(kv.first);
            }
        }
        if (vocab.empty()) { if (err) *err = "no tokens"; return false; }
        return true;
    }

    // Qwen3.5 pre-tokenizer on codepoints -> [start,end) codepoint ranges.
    void pretoken(const std::vector<uint32_t>& t, std::vector<std::pair<int,int>>& out) const {
        out.clear();
        int i = 0, n = (int)t.size();
        auto lower2 = [&](int p, char a, char b) {
            if (p + 1 >= n) return false;
            char c0 = (char)t[p], c1 = (char)t[p+1];
            if (c0 >= 'A' && c0 <= 'Z') c0 += 32;
            if (c1 >= 'A' && c1 <= 'Z') c1 += 32;
            return c0 == a && c1 == b;
        };
        auto lower3 = [&](int p, char a, char b, char cc) {
            if (p + 2 >= n) return false;
            char c0 = (char)t[p], c1 = (char)t[p+1], c2 = (char)t[p+2];
            if (c0 >= 'A' && c0 <= 'Z') c0 += 32;
            if (c1 >= 'A' && c1 <= 'Z') c1 += 32;
            if (c2 >= 'A' && c2 <= 'Z') c2 += 32;
            return c0 == a && c1 == b && c2 == cc;
        };
        while (i < n) {
            uint32_t c = t[i];
            // 1 contractions
            if (c == '\'' && i + 1 < n) {
                if (lower3(i, '\'', 'r', 'e') || lower3(i, '\'', 'v', 'e') || lower3(i, '\'', 'l', 'l')) {
                    out.push_back({i, i + 3}); i += 3; continue;
                }
                if (lower2(i, '\'', 's') || lower2(i, '\'', 't') || lower2(i, '\'', 'm') || lower2(i, '\'', 'd')) {
                    out.push_back({i, i + 2}); i += 2; continue;
                }
            }
            // 2 optional prefix + letter/mark run
            int j = -1;
            if (!(c == '\r' || c == '\n' || is_l(c) || is_n(c)) && i + 1 < n && is_lm(t[i+1])) j = i + 1;
            else if (is_lm(c)) j = i;
            if (j >= 0) {
                int k = j;
                while (k < n && is_lm(t[k])) ++k;
                out.push_back({i, k}); i = k; continue;
            }
            // 3 one number
            if (is_n(c)) { out.push_back({i, i + 1}); i += 1; continue; }
            // 4 optional space + run of "other" + trailing CR/LF
            int jj = (c == ' ' && i + 1 < n && is_other(t[i+1])) ? i + 1 : i;
            if (jj < n && is_other(t[jj])) {
                int k = jj;
                while (k < n && is_other(t[k])) ++k;
                while (k < n && (t[k] == '\r' || t[k] == '\n')) ++k;
                out.push_back({i, k}); i = k; continue;
            }
            if (is_sp(c)) {
                int k = i;
                while (k < n && is_sp(t[k])) ++k;
                int last = -1;
                for (int q = i; q < k; ++q) if (t[q] == '\r' || t[q] == '\n') last = q;
                if (last >= 0) { out.push_back({i, last + 1}); i = last + 1; continue; }  // 5
                if (k == n) { out.push_back({i, k}); i = k; continue; }                    // 6a
                if (k - 1 > i) { out.push_back({i, k - 1}); i = k - 1; continue; }          // 6b
                out.push_back({i, k}); i = k; continue;                                     // 7
            }
            out.push_back({i, i + 1}); i += 1;
        }
    }

    // BPE merge loop on per-codepoint UTF-8 pieces (== tokenizer.py bpe()).
    std::vector<std::string> bpe(const std::vector<std::string>& word) const {
        std::vector<std::string> parts = word;
        std::string key; key.reserve(64);
        while (parts.size() > 1) {
            int best_r = -1, best_k = -1;
            for (size_t k = 0; k + 1 < parts.size(); ++k) {
                key.clear(); key += parts[k]; key += '\0'; key += parts[k+1];
                auto it = ranks.find(key);
                if (it != ranks.end() && (best_k < 0 || it->second < best_r)) {
                    best_r = it->second; best_k = (int)k;
                }
            }
            if (best_k < 0) break;
            parts[best_k] += parts[best_k + 1];
            parts.erase(parts.begin() + best_k + 1);
        }
        return parts;
    }

    std::vector<int> encode(const std::string& text) {
        std::vector<int> out;
        std::vector<uint32_t> cps; std::vector<int> offs;
        utf8_decode(text, cps, offs);
        int n = (int)cps.size();
        // split on added tokens first (longest match wins at each position)
        struct Frag { int a, b; bool is_added; int add_idx; };
        std::vector<Frag> frags;
        int i = 0;
        while (i < n) {
            int hit = -1; size_t hlen = 0;
            for (size_t a = 0; a < added.size(); ++a) {
                const auto& pat = added[a];
                if (pat.size() <= hlen || i + (int)pat.size() > n) continue;
                bool ok = true;
                for (size_t k = 0; k < pat.size(); ++k)
                    if (cps[i + k] != pat[k]) { ok = false; break; }
                if (ok) { hit = (int)a; hlen = pat.size(); }
            }
            if (hit >= 0) { frags.push_back({i, i + (int)hlen, true, hit}); i += (int)hlen; }
            else {
                int s = i;
                while (i < n) {
                    bool starts = false;
                    for (size_t a = 0; a < added.size(); ++a) {
                        const auto& pat = added[a];
                        if (i + (int)pat.size() > n) continue;
                        bool ok = true;
                        for (size_t k = 0; k < pat.size(); ++k)
                            if (cps[i + k] != pat[k]) { ok = false; break; }
                        if (ok) { starts = true; break; }
                    }
                    if (starts) break;
                    ++i;
                }
                frags.push_back({s, i, false, -1});
            }
        }
        std::vector<std::pair<int,int>> toks;
        for (auto& fr : frags) {
            if (fr.is_added) { out.push_back(added_id[fr.add_idx]); continue; }
            // pretoken on the codepoint slice
            std::vector<uint32_t> slice(cps.begin() + fr.a, cps.begin() + fr.b);
            pretoken(slice, toks);
            for (auto& pr : toks) {
                // word bytes from original text
                int ba = offs[fr.a + pr.first], bb = offs[fr.a + pr.second];
                std::string word = text.substr(ba, bb - ba);
                auto cit = cache.find(word);
                if (cit != cache.end()) {
                    out.insert(out.end(), cit->second.begin(), cit->second.end());
                    continue;
                }
                // gpt2_bytes: map each byte through b2u into per-char UTF-8 pieces
                std::vector<std::string> mapped;
                mapped.reserve(word.size());
                for (size_t k = 0; k < word.size(); ++k) {
                    std::string ch; utf8_encode_cp(bm.b2u[(unsigned char)word[k]], ch);
                    mapped.push_back(ch);
                }
                std::vector<int> ids;
                for (auto& p : bpe(mapped)) {
                    auto it = sid.find(p);
                    ids.push_back(it != sid.end() ? it->second : 0);
                }
                if (cache.size() < 200000) cache[word] = ids;
                out.insert(out.end(), ids.begin(), ids.end());
            }
        }
        return out;
    }

    // decode one id to raw bytes (== decode_bytes); full decode joins + replace.
    void decode_id(int id, std::string& raw) const {
        if (id < 0 || id >= (int)vocab.size()) { raw += "\xEF\xBF\xBD"; return; }
        const std::string& s = vocab[(size_t)id];
        std::vector<uint32_t> cps; std::vector<int> offs;
        utf8_decode(s, cps, offs);
        bool allmapped = true;
        for (uint32_t cp : cps) {
            if (cp >= 0x200 || bm.u2b[cp] < 0) { allmapped = false; break; }
        }
        if (allmapped) {
            for (uint32_t cp : cps) raw.push_back((char)bm.u2b[cp]);
        } else raw += s;
    }
    std::string decode(const std::vector<int>& ids) const {
        std::string raw; raw.reserve(ids.size() * 4);
        for (int id : ids) decode_id(id, raw);
        return utf8_sanitize(raw);
    }
};

}  // namespace kindi_tok
