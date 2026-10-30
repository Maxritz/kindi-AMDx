"""Gemma 4 tokenizer from the GGUF metadata (tokenizer.ggml.model "gemma4").

The vocabulary is a SentencePiece-style BPE: each space becomes "▁" and every other character stays as it is,
newlines, tabs and runs of spaces included. Tokenizing a text:

1. split out the special tokens (control, user-defined and unknown types), which map to their own ids;
2. in each plain segment, replace " " with "▁" and cut the segment into runs of newlines and runs of other
   characters (a BPE token never holds a newline next to other text);
3. a run of newlines that is itself a token becomes that token; any other run starts as one symbol per character,
   and the adjacent pair with the lowest merge rank merges first (on a tie, the leftmost pair), until no pair has a
   merge;
4. a symbol that is a token gives its id; another symbol gives one <0xXX> byte token for each of its UTF-8 bytes.

This is the input that the model was trained on, and the same token ids as llama.cpp for the same GGUF. The chat
template writes the <bos> itself, so encode() adds none. Each plain run is cached: agent prompts repeat most lines."""
import heapq
import re

from .gguf import GGUF

_RUNS = re.compile(r"[^\n]+|\n+")
_CACHE_MAX = 200000


class Tok:
    def __init__(self, model_path):
        kv = GGUF(model_path).kv
        if kv.get("tokenizer.ggml.model", "gemma4") != "gemma4":
            raise ValueError("kindi: this tokenizer is for gemma4 GGUFs, not %r" % kv.get("tokenizer.ggml.model"))
        self.tokens = kv.get("tokenizer.ggml.tokens")
        merges = kv.get("tokenizer.ggml.merges")
        if not self.tokens or not merges:
            raise ValueError("kindi: Gemma 4 GGUF needs token strings and BPE merges")
        self.types = kv.get("tokenizer.ggml.token_type") or [1] * len(self.tokens)
        self.bos = int(kv.get("tokenizer.ggml.bos_token_id", 2))
        self.eos = int(kv.get("tokenizer.ggml.eos_token_id", 1))
        self.unk = int(kv.get("tokenizer.ggml.unknown_token_id", 3))
        self.ids = {}
        self.special = {}
        self.byte_reverse = {}
        for idx, (piece, typ) in enumerate(zip(self.tokens, self.types)):
            self.ids.setdefault(piece, idx)
            if int(typ) in (2, 3, 4) and piece:            # unknown, control, user-defined
                self.special[piece] = idx
            if len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">"):
                try:
                    self.byte_reverse[idx] = int(piece[3:5], 16)
                except ValueError:
                    pass
        self.ranks = {}
        for rank, m in enumerate(merges):
            cut = m.find(" ", 1)                              # the first space after position 0 splits the pair
            if cut > 0:
                self.ranks.setdefault((m[:cut], m[cut + 1:]), rank)
        order = sorted(self.special, key=len, reverse=True)
        self._special_rx = re.compile("|".join(re.escape(t) for t in order)) if order else None
        self._cache = {}
        self.stop_ids = {self.eos}
        for i, piece in enumerate(self.tokens):
            # <|tool_response> follows a tool call: the client runs the tool and sends the response
            if piece in ("<turn|>", "<end_of_turn>", "<|im_end|>", "<eos>", "</s>", "<|tool_response>"):
                self.stop_ids.add(i)

    # ------------------------------------------------------------------ encode
    def _bpe(self, word):
        """one run of text (spaces already "▁") -> token ids"""
        hit = self._cache.get(word)
        if hit is not None:
            return hit
        if word[0] == "\n" and word in self.ids:
            out = [self.ids[word]]
        else:
            sym = list(word)                                  # symbols; "" marks one merged away
            nxt = list(range(1, len(sym))) + [-1]
            prv = list(range(-1, len(sym) - 1))
            heap = []
            ranks = self.ranks

            def push(i, j):
                if i >= 0 and j >= 0:
                    r = ranks.get((sym[i], sym[j]))
                    if r is not None:
                        heapq.heappush(heap, (r, i, sym[i] + sym[j]))
            for i in range(len(sym) - 1):
                push(i, i + 1)
            while heap:
                r, i, text = heapq.heappop(heap)
                j = nxt[i]
                if j < 0 or not sym[i] or not sym[j] or sym[i] + sym[j] != text:
                    continue                                  # an old pair: one side has changed
                sym[i] = text
                sym[j] = ""
                nxt[i] = nxt[j]
                if nxt[j] >= 0:
                    prv[nxt[j]] = i
                push(prv[i], i)
                push(i, nxt[i])
            out = []
            for s in sym:
                if not s:
                    continue
                t = self.ids.get(s)
                if t is not None:
                    out.append(t)
                else:
                    for b in s.encode("utf-8"):
                        t = self.ids.get("<0x%02X>" % b)
                        if t is not None:
                            out.append(t)
        if len(self._cache) < _CACHE_MAX:
            self._cache[word] = out
        return out

    def _plain(self, text):
        out = []
        for run in _RUNS.findall(text.replace(" ", "▁")):
            out.extend(self._bpe(run))
        return out

    def encode(self, text):
        if not isinstance(text, str):
            text = str(text)
        if self._special_rx is None:
            return self._plain(text)
        out, pos = [], 0
        for m in self._special_rx.finditer(text):
            if m.start() > pos:
                out.extend(self._plain(text[pos:m.start()]))
            out.append(self.special[m.group(0)])
            pos = m.end()
        if pos < len(text):
            out.extend(self._plain(text[pos:]))
        return out

    # ------------------------------------------------------------------ decode
    def decode_bytes(self, idx):
        idx = int(idx)
        if idx in self.byte_reverse:
            return bytes([self.byte_reverse[idx]])
        if idx < 0 or idx >= len(self.tokens):
            return b""
        return self.tokens[idx].replace("▁", " ").encode("utf-8", errors="replace")

    def decode(self, ids):
        return b"".join(self.decode_bytes(i) for i in ids).decode("utf-8", errors="replace")
