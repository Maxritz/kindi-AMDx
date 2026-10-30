"""Gemma 4 tool calls and thinking: the OpenAI messages in, the model's markup out.

The model writes its thinking as <|channel>thought\\n...<channel|> and each tool call as
<|tool_call>call:NAME{key:VALUE,...}<tool_call|>, where a string VALUE is <|"|>text<|"|>, and objects, lists,
numbers, true, false and null are written as in JSON but with unquoted keys. Each marker is one token."""
import json
import re
import uuid

Q = '<|"|>'
CALL_OPEN, CALL_CLOSE = "<|tool_call>", "<tool_call|>"
THINK_OPEN, THINK_CLOSE = "<|channel>", "<channel|>"


def prepare_messages(messages):
    """a copy of the OpenAI messages that the Gemma 4 template accepts: tool-call arguments as objects, not as
    JSON strings (OpenAI clients send strings), and no None content"""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("content") is None:
            m["content"] = ""
        if m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                tc = dict(tc)
                f = dict(tc.get("function") or {})
                a = f.get("arguments")
                if isinstance(a, str):
                    try:
                        a = json.loads(a) if a.strip() else {}
                    except ValueError:
                        a = {}
                    if not isinstance(a, dict):
                        a = {"value": a}
                f["arguments"] = a
                tc["function"] = f
                calls.append(tc)
            m["tool_calls"] = calls
        out.append(m)
    return out


class _Reader:
    def __init__(self, s):
        self.s, self.i = s, 0

    def skip(self):
        while self.i < len(self.s) and self.s[self.i] in " \t\r\n":
            self.i += 1

    def take(self, t):
        self.skip()
        if self.s.startswith(t, self.i):
            self.i += len(t)
            return True
        return False

    def key(self):
        self.skip()
        if self.take(Q):
            j = self.s.find(Q, self.i)
            if j < 0:
                raise ValueError("open key")
            k, self.i = self.s[self.i:j], j + len(Q)
            return k
        j = self.s.find(":", self.i)
        if j < 0:
            raise ValueError("no ':' after a key")
        k, self.i = self.s[self.i:j].strip(), j
        return k

    def obj(self):
        d = {}
        if self.take("}"):
            return d
        while True:
            k = self.key()
            if not self.take(":"):
                raise ValueError("no ':' after " + k)
            d[k] = self.value()
            if self.take(","):
                continue
            if self.take("}"):
                return d
            raise ValueError("no ',' or '}' after " + k)

    def value(self):
        self.skip()
        if self.take(Q):
            j = self.s.find(Q, self.i)
            if j < 0:
                raise ValueError("open string")
            v, self.i = self.s[self.i:j], j + len(Q)
            return v
        if self.take("{"):
            return self.obj()
        if self.take("["):
            out = []
            if self.take("]"):
                return out
            while True:
                out.append(self.value())
                if self.take(","):
                    continue
                if self.take("]"):
                    return out
                raise ValueError("no ',' or ']' in a list")
        j = self.i
        while j < len(self.s) and self.s[j] not in ",}]":
            j += 1
        word, self.i = self.s[self.i:j].strip(), j
        if word in ("true", "false", "null"):
            return {"true": True, "false": False, "null": None}[word]
        try:
            return json.loads(word)
        except ValueError:
            return word


def parse_call(body):
    """'call:NAME{...}' -> (NAME, arguments dict)"""
    body = body.strip()
    if body.startswith("call:"):
        body = body[5:]
    b = body.find("{")
    if b < 0:
        return body.strip(), {}
    r = _Reader(body)
    r.i = b + 1
    return body[:b].strip(), r.obj()


def split_reply(text):
    """the model's reply -> (content, reasoning, [OpenAI tool_calls]). A call that is cut off or malformed stays in
    the content as text, so that nothing is lost."""
    reasoning = []
    while THINK_OPEN in text:
        a = text.index(THINK_OPEN)
        b = text.find(THINK_CLOSE, a)
        inner = text[a + len(THINK_OPEN):] if b < 0 else text[a + len(THINK_OPEN):b]
        if inner.startswith("thought"):
            inner = inner[len("thought"):]
        reasoning.append(inner.strip())
        text = text[:a] + ("" if b < 0 else text[b + len(THINK_CLOSE):])
    # an unmatched <channel|> ends the reply (the model sometimes closes a channel it never opened): the content stops
    # there, as llama.cpp's Gemma 4 parser does, so the marker and the text after it never reach the conversation
    stray = text.find(THINK_CLOSE)
    if stray >= 0:
        text = text[:stray]
    calls, content, pos = [], text, 0
    first = text.find(CALL_OPEN)
    while True:
        a = text.find(CALL_OPEN, pos)
        if a < 0:
            break
        b = text.find(CALL_CLOSE, a)
        if b < 0:
            break
        try:
            name, args = parse_call(text[a + len(CALL_OPEN):b])
        except ValueError:
            break
        calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
        pos = b + len(CALL_CLOSE)
    if not calls and first < 0 and CALL_CLOSE in text:
        # the opening marker is missing, but a whole call body ends at <tool_call|>: "[call]:NAME{...}<tool_call|>"
        b = text.find(CALL_CLOSE)
        m = re.search(r"(?:call)?:?\s*([A-Za-z_][\w.-]*)\s*\{", text[:b])
        if m:
            try:
                name, args = parse_call(text[m.start(1):b])
                calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                              "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
                first = m.start()
            except ValueError:
                pass
    if calls:
        content = text[:first]
    return content.strip(), "\n".join(r for r in reasoning if r), calls


class CallGrammar:
    """The syntax of one Gemma 4 tool call, checked one character at a time, for constrained sampling.

    After <|tool_call>, the text must be: "call:" + one of the request's tool names + an argument object + "<tool_call|>".
    An object is "{" [ws] ("}" | key ":" [ws] value ("," [ws] key ":" [ws] value)* [ws] "}"), where a key is one or more
    characters other than ":" and "}"; a list is "[" [ws] ("]" | value ("," [ws] value)* [ws] "]"); a value is a string
    <|"|>...<|"|>, an object, a list, a JSON number, true, false or null. This is the call syntax that llama.cpp
    enforces with its grammar for Gemma 4 (the argument names are not checked by either engine).

    A state is a small tuple, so a candidate token is checked on a copy: advance(state, text) -> new state or None."""
    WS = " \t\n\r"
    NUM = re.compile(r"-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?\Z")

    def __init__(self, names):
        self.names = sorted(set(n for n in names if n))

    def start(self):
        return ("call", (), "")

    def done(self, st):
        return st is not None and st[0] == "done"

    def _value_start(self, stk, ch):
        if ch == "<":
            return ("sopen", stk, "<")
        if ch == "{":
            return ("dfirst", stk + ("D",), "")
        if ch == "[":
            return ("afirst", stk + ("A",), "")
        if ch == "-" or ch.isdigit():
            return ("num", stk, ch)
        if ch in "tfn":
            return ("kw", stk, ch)
        return None

    def _close(self, stk, ch):
        if not stk or ch != ("}" if stk[-1] == "D" else "]"):
            return None
        stk = stk[:-1]
        return ("tail", (), "") if not stk else ("after", stk, "")

    def _step(self, st, ch):
        mode, stk, buf = st
        if mode == "call":
            buf += ch
            if not "call:".startswith(buf):
                return None
            return ("name", (), "") if buf == "call:" else ("call", (), buf)
        if mode == "name":
            if ch == "{" and buf in self.names:
                return ("dfirst", ("D",), "")
            buf += ch
            return ("name", (), buf) if any(n.startswith(buf) for n in self.names) else None
        if mode in ("dfirst", "dkey"):
            if ch in self.WS:
                return st
            if mode == "dfirst" and ch == "}":
                return self._close(stk, ch)
            return None if ch in ":}" else ("key", stk, "")
        if mode == "key":
            if ch == ":":
                return ("vws", stk, "")
            return None if ch == "}" else st
        if mode in ("vws", "afirst", "aval"):
            if ch in self.WS:
                return st
            if mode == "afirst" and ch == "]":
                return self._close(stk, ch)
            return self._value_start(stk, ch)
        if mode == "sopen":
            buf += ch
            if not Q.startswith(buf):
                return None
            return ("str", stk, "") if buf == Q else ("sopen", stk, buf)
        if mode == "str":
            buf = (buf + ch)[-len(Q):]
            return ("after", stk, "") if buf == Q else ("str", stk, buf)
        if mode == "num":
            if ch in "0123456789.eE+-":
                return ("num", stk, buf + ch)
            if not self.NUM.match(buf):
                return None
            return self._step(("after", stk, ""), ch)
        if mode == "kw":
            buf += ch
            if not any(w.startswith(buf) for w in ("true", "false", "null")):
                return None
            return ("after", stk, "") if buf in ("true", "false", "null") else ("kw", stk, buf)
        if mode in ("after", "aws"):
            if ch in self.WS:
                return ("aws", stk, "")
            if ch == "," and mode == "after":
                return ("dkey", stk, "") if stk[-1] == "D" else ("aval", stk, "")
            return self._close(stk, ch)
        if mode == "tail":
            buf += ch
            if not CALL_CLOSE.startswith(buf):
                return None
            return ("done", (), "") if buf == CALL_CLOSE else ("tail", (), buf)
        return None                                           # "done": nothing more inside this call

    def advance(self, st, text):
        if st is None:
            return None
        for ch in text:
            st = self._step(st, ch)
            if st is None:
                return None
        return st


class Stream:
    """sorts the streamed pieces into ("text", s) and ("reasoning", s); stops sending text at the first tool call"""

    def __init__(self, thinking_open=False):
        self.mode = "think" if thinking_open else "text"
        self.label = thinking_open          # the "thought\n" label after <|channel> is not reasoning text
        self.head = ""

    def feed(self, piece):
        out = []
        if piece == THINK_OPEN and self.mode == "text":
            self.mode, self.label, self.head = "think", True, ""
            return out
        if piece == THINK_CLOSE and self.mode == "think":
            self.mode = "text"
            if self.head:
                out.append(("reasoning", self.head))
            self.head = ""
            return out
        if piece == THINK_CLOSE and self.mode == "text":
            self.mode = "end"                 # an unmatched <channel|>: nothing after it is sent (as split_reply)
            return out
        if piece == CALL_OPEN:
            self.mode = "call"
            return out
        if self.mode == "think":
            if self.label:
                self.head += piece
                if "\n" in self.head or len(self.head) > 8:
                    h = self.head[len("thought"):] if self.head.startswith("thought") else self.head
                    self.label, self.head = False, ""
                    h = h.lstrip("\n")
                    if h:
                        out.append(("reasoning", h))
                return out
            out.append(("reasoning", piece))
        elif self.mode == "text" and piece:
            out.append(("text", piece))
        return out
