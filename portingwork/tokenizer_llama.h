// tokenizer_llama.h - Llama-3 byte-level BPE, llama.cpp-exact (llama-bpe pre-tokenizer).
// Derived from tokenizer.h (Qwen3.5) with the three Llama-3 differences:
//   1. pre-tokenizer digit rule is \p{N}{1,3} instead of \p{N}
//   2. ignore_merges: a whole pre-token that exists in the vocab is emitted directly
//   3. the added/special-token scan starts at the first non-NORMAL token type
// Header-only. Needs unitbl.inc next to it.
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

namespace kindi_llama_tok {

#include "unitbl.inc"

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
static inline bool is_sp(uint32_t c) { return in_ranges(SPT, sizeof(SPT)/sizeof(SPT[0]), c); }
static inline bool is_other(uint32_t c) { return !(is_sp(c) || is_lm(c) || is_n(c)); }

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
        if (ok) {
            if ((len == 2 && v < 0x80) || (len == 3 && v < 0x800) || (len == 4 && v < 0x10000) ||
                (v >= 0xD800 && v <= 0xDFFF) || v > 0x10FFFF) ok = false;
        }
        if (!ok) { out += "\xEF\xBF\xBD"; ++i; continue; }
        out.append(s, i, len); i += len;
    }
    return out;
}

struct ByteMap {
    uint32_t b2u[256];
    int32_t u2b[0x200];
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

class LlamaTokenizer {
  public:
    std::vector<std::string> vocab;
    std::unordered_map<std::string, int> sid;
    std::unordered_map<std::string, int> ranks;
    std::vector<std::vector<uint32_t>> added;
    std::vector<std::string> added_str;
    std::unordered_map<int, int> added_id;
    std::unordered_map<std::string, std::vector<int>> cache;
    ByteMap bm;

    // specials_first: first token id that is not a NORMAL token (0xFFFFFFFF if none)
    bool load_from_strings(const std::vector<std::string>& tokens_vec,
                           const std::vector<std::string>& merges_vec,
                           size_t specials_first, std::string* err = nullptr) {
        vocab = tokens_vec;
        sid.clear(); ranks.clear(); added.clear(); added_str.clear(); added_id.clear(); cache.clear();
        sid.reserve(vocab.size() * 2);
        for (size_t i = 0; i < vocab.size(); ++i)
            if (sid.find(vocab[i]) == sid.end()) sid[vocab[i]] = (int)i;
        int r = 0;
        for (const auto& m : merges_vec) {
            std::string line = m;
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            std::vector<std::string> parts; size_t s = 0;
            for (size_t k = 0; k <= line.size(); ++k)
                if (k == line.size() || line[k] == ' ') { parts.push_back(line.substr(s, k - s)); s = k + 1; }
            if (parts.size() == 2) ranks[parts[0] + '\0' + parts[1]] = r;
            ++r;
        }
        if (specials_first < vocab.size()) {
            std::vector<std::pair<std::string,int>> tmp;
            for (size_t i = specials_first; i < vocab.size(); ++i) {
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

    // llama-bpe pre-tokenizer on codepoints -> [start,end) codepoint ranges.
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
            // 3 \p{N}{1,3}
            if (is_n(c)) {
                int k = i, cnt = 0;
                while (k < n && cnt < 3 && is_n(t[k])) { ++k; ++cnt; }
                out.push_back({i, k}); i = k; continue;
            }
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
                if (last >= 0) { out.push_back({i, last + 1}); i = last + 1; continue; }
                if (k == n) { out.push_back({i, k}); i = k; continue; }
                if (k - 1 > i) { out.push_back({i, k - 1}); i = k - 1; continue; }
                out.push_back({i, k}); i = k; continue;
            }
            out.push_back({i, i + 1}); i += 1;
        }
    }

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
            std::vector<uint32_t> slice(cps.begin() + fr.a, cps.begin() + fr.b);
            pretoken(slice, toks);
            for (auto& pr : toks) {
                int ba = offs[fr.a + pr.first], bb = offs[fr.a + pr.second];
                std::string word = text.substr(ba, bb - ba);
                auto cit = cache.find(word);
                if (cit != cache.end()) {
                    out.insert(out.end(), cit->second.begin(), cit->second.end());
                    continue;
                }
                std::vector<std::string> mapped;
                mapped.reserve(word.size());
                for (size_t k = 0; k < word.size(); ++k) {
                    std::string ch; utf8_encode_cp(bm.b2u[(unsigned char)word[k]], ch);
                    mapped.push_back(ch);
                }
                std::vector<int> ids;
                std::string whole;
                for (auto& p : mapped) whole += p;
                auto wit = sid.find(whole);   // ignore_merges
                if (wit != sid.end()) {
                    ids.push_back(wit->second);
                } else {
                    for (auto& p : bpe(mapped)) {
                        auto it = sid.find(p);
                        ids.push_back(it != sid.end() ? it->second : 0);
                    }
                }
                if (cache.size() < 200000) cache[word] = ids;
                out.insert(out.end(), ids.begin(), ids.end());
            }
        }
        return out;
    }

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

}  // namespace kindi_llama_tok
