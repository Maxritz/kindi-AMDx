// graph_hip.h - the Kindi decode graph for RDNA2.
//
// Included from kindi_hip.cpp inside main(), so the lambdas can capture the buffers and
// the loaded model directly. This is the layer logic: 24 layers, full attention every
// interval-th layer, gated-delta-net recurrence on the rest, then a SwiGLU FFN.
//
// The numeric contract is the Vulkan engine's, because that is the reference this has to
// match token for token. What differs is the structure: single-pass FWHT (64 KB of LDS
// makes it possible), wave32 with RPW rows, and no CUDA-shaped warp emulation.

// ---------------------------------------------------------------- GEMV dispatch
auto gemv = [&](const std::string& name, Buf& xb, Buf& yb) {
    auto it = m.rc.find(name);
    if (it == m.rc.end()) { fprintf(stderr, "missing tensor %s\n", name.c_str()); exit(1); }
    int R = it->second.first, C = it->second.second;
    int ws = G.prop.warpSize, nsub = 256 / ws, per = nsub * RPW;
    int grid = (R + per - 1) / per;
    size_t bytes = (size_t)R * (C >> 7) * 28;
    ktrace::Launch L = ktrace::begin(name.c_str(), grid, 1, 1, 256, bytes);
    gemv_ptq1<<<grid, 256>>>((const uint*)m.w[name].p, (const float*)xb.p, (float*)yb.p, R, C);
    ktrace::end(L);
};

// three tensors sharing one input, one launch
auto gemv3 = [&](Buf& xb, const std::string& n0, Buf& y0,
                 const std::string& n1, Buf& y1,
                 const std::string& n2, Buf& y2) -> bool {
    auto i0 = m.rc.find(n0), i1 = m.rc.find(n1), i2 = m.rc.find(n2);
    if (i0 == m.rc.end() || i1 == m.rc.end() || i2 == m.rc.end()) return false;
    if (i0->second.second != i1->second.second || i0->second.second != i2->second.second) return false;
    int R0 = i0->second.first, R1 = i1->second.first, R2 = i2->second.first;
    int C = i0->second.second;
    int ws = G.prop.warpSize, nsub = 256 / ws;
    auto grp = [&](int R) { return (R + RPW - 1) / RPW; };
    int total = grp(R0) + grp(R1) + grp(R2);
    int grid = (total + nsub - 1) / nsub;
    size_t bytes = (size_t)(R0 + R1 + R2) * (C >> 7) * 28;
    ktrace::Launch L = ktrace::begin("gemv_ptq1_3", grid, 1, 1, 256, bytes);
    gemv_ptq1_3<<<grid, 256>>>((const uint*)m.w[n0].p, (float*)y0.p,
                               (const uint*)m.w[n1].p, (float*)y1.p,
                               (const uint*)m.w[n2].p, (float*)y2.p,
                               (const float*)xb.p, R0, R1, R2, C);
    ktrace::end(L);
    return true;
};

// ---------------------------------------------------------------- rmsfold
// rms(h + residual) -> xn (normalised, for the gate scalars) and xp (folded, for the GEMV)
auto rmsfold_l = [&](Buf& x, Buf& resid, Buf& xn, Buf& xp, const std::string& wname, int C, bool addres) {
    int nblk = C / m.HB;
    size_t shm = (size_t)m.HB * 4;
    ktrace::Launch L = ktrace::begin("rmsfold", nblk, 1, 1, 256, (size_t)C * 12);
    rmsfold<<<nblk, 256, shm>>>((const float*)x.p, addres ? (const float*)resid.p : nullptr,
                                (const float*)m.f32[wname].p, (const float*)sgn(C).p,
                                (float*)xn.p, (float*)xp.p, C, m.HB, m.eps,
                                m.hbscale, addres);
    ktrace::end(L);
};

// ---------------------------------------------------------------- GDN layer
auto gdn_layer = [&](int il, int pos) {
    std::string p = "blk." + std::to_string(il) + ".";
    rmsfold_l(h, yD, xn, xp, p + "attn_norm.weight", m.D, pending);
    pending = false;

    // qkv and gate share xp; two launches (the fused kernel needs matching C, which holds)
    if (!gemv3(xp, p + "attn_qkv.weight", qkv, p + "attn_gate.weight", zv, p + "ssm_out.weight", yD)) {
        gemv(p + "attn_qkv.weight", xp, qkv);
        gemv(p + "attn_gate.weight", xp, zv);
    }
    if (il == 0 && getenv("KINDI_DBG")) {
        CK(hipDeviceSynchronize());
        float a[3] = {0,0,0}, b[3] = {0,0,0}, c2[3] = {0,0,0}, d2[3] = {0,0,0};
        CK(hipMemcpy(a, xn.p, 12, hipMemcpyDeviceToHost));
        CK(hipMemcpy(b, xp.p, 12, hipMemcpyDeviceToHost));
        CK(hipMemcpy(c2, qkv.p, 12, hipMemcpyDeviceToHost));
        CK(hipMemcpy(d2, zv.p, 12, hipMemcpyDeviceToHost));
        fprintf(stderr, "G xn: %.4f %.4f %.4f\n", a[0], a[1], a[2]);
        fprintf(stderr, "G xp: %.4f %.4f %.4f\n", b[0], b[1], b[2]);
        fprintf(stderr, "G qkv: %.4f %.4f %.4f\n", c2[0], c2[1], c2[2]);
        fprintf(stderr, "G z: %.4f %.4f %.4f\n", d2[0], d2[1], d2[2]);
    }

    // gates from xn
    {
        ktrace::Launch L = ktrace::begin("gdn_gates", m.NK, 1, 1, 256, (size_t)m.D * m.NK * 8);
        gdn_gates<<<m.NK, 256>>>((const float*)xn.p,
                    (const float*)m.f32[p + "ssm_beta.weight"].p,
                    (const float*)m.f32[p + "ssm_alpha.weight"].p,
                    (const float*)m.f32[p + "ssm_dt.bias"].p,
                    (const float*)m.f32[p + "ssm_a"].p,
                    (float*)gb.p, (float*)gg.p, m.D, m.NK);
        ktrace::end(L);
    }
    if (il == 0 && getenv("KINDI_DBG")) {
        CK(hipDeviceSynchronize());
        float b[3] = {0,0,0}, g[3] = {0,0,0};
        CK(hipMemcpy(b, gb.p, 12, hipMemcpyDeviceToHost));
        CK(hipMemcpy(g, gg.p, 12, hipMemcpyDeviceToHost));
        fprintf(stderr, "G beta: %.4f %.4f %.4f\n", b[0], b[1], b[2]);
        fprintf(stderr, "G gate: %.4f %.4f %.4f\n", g[0], g[1], g[2]);
    }
    // conv + SiLU
    {
        ktrace::Launch L = ktrace::begin("gdn_conv", (m.CONV_CH + 255) / 256, 1, 1, 256, (size_t)m.CONV_CH * 8);
        gdn_conv<<<(m.CONV_CH + 255) / 256, 256>>>((const float*)qkv.p,
                    (float*)convst[il].p, (const float*)m.f32[p + "ssm_conv1d.weight"].p,
                    (float*)cs.p, m.CONV_CH);
        ktrace::end(L);
    }
    if (il == 0 && getenv("KINDI_DBG")) {
        CK(hipDeviceSynchronize());
        float c3[3] = {0,0,0};
        CK(hipMemcpy(c3, cs.p, 12, hipMemcpyDeviceToHost));
        fprintf(stderr, "G conv_silu: %.4f %.4f %.4f\n", c3[0], c3[1], c3[2]);
    }
    // l2 normalise q and k
    {
        ktrace::Launch L = ktrace::begin("l2w", (2 * m.NK + 3) / 4, 1, 1, 256, (size_t)2 * m.NK * m.HK * 8);
        l2w<<<(2 * m.NK + 3) / 4, 256>>>((const float*)cs.p, (float*)qkbuf.p, m.HK, 2 * m.NK);
        ktrace::end(L);
    }
    // gated delta rule
    {
        ktrace::Launch L = ktrace::begin("gdn_delta", m.NV, 1, 1, m.HK, (size_t)m.NV * m.HK * m.HK * 8);
        // q and k come from the normalised qkbuf, v from the raw conv output at the
        // 2*NK*HK offset. Only q and k are L2-normalised.
        gdn_delta<<<m.NV, m.HK>>>((float*)M[il].p,
                    (const float*)qkbuf.p,
                    (const float*)((const char*)qkbuf.p + (size_t)m.NK * m.HK * 4),
                    (const float*)((const char*)cs.p + (size_t)2 * m.NK * m.HK * 4),
                    (const float*)gg.p, (const float*)gb.p,
                    (float*)gout.p, m.HK, m.NV, m.NK, 1.0f / sqrtf((float)m.HK));
        ktrace::end(L);
    }
    if (il == 0 && getenv("KINDI_DBG")) {
        CK(hipDeviceSynchronize());
        float d3[3] = {0,0,0};
        CK(hipMemcpy(d3, gout.p, 12, hipMemcpyDeviceToHost));
        fprintf(stderr, "G delta_out: %.4f %.4f %.4f\n", d3[0], d3[1], d3[2]);
    }
    // rms per value head
    {
        ktrace::Launch L = ktrace::begin("rmsw", (m.NV + 3) / 4, 1, 1, 256, (size_t)m.NV * m.HK * 8);
        rmsw<<<(m.NV + 3) / 4, 256>>>((const float*)gout.p, (const float*)m.f32[p + "ssm_norm.weight"].p,
                                      (float*)outgdn.p, m.HK, m.NV, 1e-6f);
        ktrace::end(L);
    }
    if (il == 0 && getenv("KINDI_DBG")) {
        CK(hipDeviceSynchronize());
        float n3[3] = {0,0,0};
        CK(hipMemcpy(n3, outgdn.p, 12, hipMemcpyDeviceToHost));
        fprintf(stderr, "G gdn_norm: %.4f %.4f %.4f\n", n3[0], n3[1], n3[2]);
    }
    // SwiGLU + fold in one pass
    {
        int nblk = (m.NV * m.HK) / m.HB;
        ktrace::Launch L = ktrace::begin("smfold", nblk, 1, 1, 256, (size_t)m.NV * m.HK * 12);
        smfold<<<nblk, 256, (size_t)m.HB * 4>>>((const float*)zv.p, (const float*)outgdn.p,
                                                (const float*)sgn(m.NV * m.HK).p, (float*)xp.p,
                                                m.HB, m.hbscale, m.NV * m.HK);
        ktrace::end(L);
    }
    gemv(p + "ssm_out.weight", xp, yD);
    if (il == 0 && getenv("KINDI_DBG")) {
        CK(hipDeviceSynchronize());
        float y3[3] = {0,0,0};
        CK(hipMemcpy(y3, yD.p, 12, hipMemcpyDeviceToHost));
        fprintf(stderr, "G ssm_out: %.4f %.4f %.4f\n", y3[0], y3[1], y3[2]);
    }
    pending = true;
};

// ---------------------------------------------------------------- attention layer
auto attn_layer = [&](int il, int pos, int kvi) {
    std::string p = "blk." + std::to_string(il) + ".";
    rmsfold_l(h, yD, xn, xp, p + "attn_norm.weight", m.D, pending);
    pending = false;

    if (!gemv3(xp, p + "attn_q.weight", qn, p + "attn_k.weight", kro, p + "attn_v.weight", qkvbuf)) {
        gemv(p + "attn_q.weight", xp, qn);
        gemv(p + "attn_k.weight", xp, kro);
        gemv(p + "attn_v.weight", xp, qkvbuf);
    }
    // qn is [q|gate] per head: copy the q half into qr
    {
        int nsub = 256 / G.prop.warpSize;
        ktrace::Launch L = ktrace::begin("cpq", m.NH, 1, 1, 256, (size_t)m.NH * m.HD * 8);
        // reuse addv-style copy: qr[i] = qn[(i/HD)*2*HD + i%HD]
        // done on the host side is too slow; a tiny kernel is in kernels_hip as cpq
        cpq<<<m.NH, m.HD>>>((const float*)qn.p, (float*)qr.p, m.HD);
        (void)nsub;
        ktrace::end(L);
    }
    // q norm, then rope
    {
        ktrace::Launch L = ktrace::begin("rmsw_q", (m.NH + 7) / 8, 1, 1, 256, (size_t)m.NH * m.HD * 8);
        rmsw<<<(m.NH + 7) / 8, 256>>>((const float*)qr.p, (const float*)m.f32[p + "attn_q_norm.weight"].p,
                                      (float*)fo.p, m.HD, m.NH, 1e-6f);
        ktrace::end(L);
    }
    {
        ktrace::Launch L = ktrace::begin("rope_q", m.NH, 1, 1, m.nrot, (size_t)m.NH * m.HD * 8);
        rope<<<m.NH, m.nrot>>>((float*)fo.p, (const float*)cosB.p, (const float*)sinB.p,
                               m.HD, pos, m.nrot);
        ktrace::end(L);
        // roped q goes back to qr for the scores
        hipMemcpy(qr.p, fo.p, (size_t)m.NH * m.HD * 4, hipMemcpyDeviceToDevice);
    }
    // k norm, then rope, then store
    {
        ktrace::Launch L = ktrace::begin("rmsw_k", (m.NHKV + 7) / 8, 1, 1, 256, (size_t)m.NHKV * m.HD * 8);
        rmsw<<<(m.NHKV + 7) / 8, 256>>>((const float*)kro.p, (const float*)m.f32[p + "attn_k_norm.weight"].p,
                                        (float*)kn.p, m.HD, m.NHKV, 1e-6f);
        ktrace::end(L);
    }
    {
        ktrace::Launch L = ktrace::begin("rope_k", m.NHKV, 1, 1, m.nrot, (size_t)m.NHKV * m.HD * 8);
        rope<<<m.NHKV, m.nrot>>>((float*)kn.p, (const float*)cosB.p, (const float*)sinB.p,
                                 m.HD, pos, m.nrot);
        ktrace::end(L);
    }
    // store K and V (fp32 cache: 12 GB of VRAM, and it keeps the path exact)
    hipMemcpy((char*)Kc[kvi].p + (size_t)pos * m.NHKV * m.HD * 4, kn.p,
              (size_t)m.NHKV * m.HD * 4, hipMemcpyDeviceToDevice);
    hipMemcpy((char*)Vc[kvi].p + (size_t)pos * m.NHKV * m.HD * 4, qkvbuf.p,
              (size_t)m.NHKV * m.HD * 4, hipMemcpyDeviceToDevice);

    // scores: sbuf[NH][n], softmax over n, combine with V, gate, fold
    int n = pos + 1;
    int G = m.NH / m.NHKV;
    {
        ktrace::Launch L = ktrace::begin("attn_scores", m.NHKV, (n + 31) / 32, 1, 32,
                                         (size_t)m.NHKV * m.HD * 8);
        attn_scores<<<dim3(m.NHKV, (n + 31) / 32), 32>>>((const float*)Kc[kvi].p, (const float*)qr.p,
                                                        (float*)sbuf.p, n, m.NHKV, G, m.HD,
                                                        1.0f / sqrtf((float)m.HD));
        ktrace::end(L);
    }
    {
        ktrace::Launch L = ktrace::begin("softmax_rows", m.NH, 1, 1, 256, (size_t)m.NH * n * 8);
        softmax_rows<<<m.NH, 256>>>((float*)sbuf.p, n);
        ktrace::end(L);
    }
    {
        ktrace::Launch L = ktrace::begin("attn_comb", m.NH, 1, 1, 256, (size_t)m.NH * m.HD * 8);
        attn_comb<<<m.NH, m.HD>>>((const float*)sbuf.p, (const float*)Vc[kvi].p,
                                  (float*)att.p, n, m.NH, m.NHKV, m.HD, G);
        ktrace::end(L);
    }
    // gate the combined output: att *= sigmoid(qn gate half), then fold
    {
        ktrace::Launch L = ktrace::begin("attn_gate_mul", m.NH, 1, 1, 256, (size_t)m.NH * m.HD * 8);
        attn_gate<<<m.NH, m.HD>>>((float*)att.p, (const float*)qn.p, m.HD);
        ktrace::end(L);
    }
    {
        int nblk = (m.NH * m.HD) / m.HB;
        ktrace::Launch L = ktrace::begin("fold_fwht", nblk, 1, 1, 256, (size_t)m.NH * m.HD * 12);
        fold_fwht<<<nblk, 256, (size_t)m.HB * 4>>>((const float*)att.p, (const float*)sgn(m.NH * m.HD).p,
                                                   (float*)xp.p, m.HB, m.hbscale, m.NH * m.HD);
        ktrace::end(L);
    }
    gemv(p + "attn_output.weight", xp, yD);
    pending = true;
};

// ---------------------------------------------------------------- FFN
auto ffn_layer = [&](int il) {
    std::string p = "blk." + std::to_string(il) + ".";
    rmsfold_l(h, yD, xn, xp, p + "post_attention_norm.weight", m.D, pending);
    pending = false;
    if (!gemv3(xp, p + "ffn_gate.weight", ffg, p + "ffn_up.weight", ffu, p + "ffn_down.weight", yD)) {
        gemv(p + "ffn_gate.weight", xp, ffg);
        gemv(p + "ffn_up.weight", xp, ffu);
    }
    {
        int nblk = m.NFF / m.HB;
        ktrace::Launch L = ktrace::begin("smfold", nblk, 1, 1, 256, (size_t)m.NFF * 12);
        smfold<<<nblk, 256, (size_t)m.HB * 4>>>((const float*)ffg.p, (const float*)ffu.p,
                                                (const float*)sgn(m.NFF).p, (float*)fo6.p,
                                                m.HB, m.hbscale, m.NFF);
        ktrace::end(L);
    }
    gemv(p + "ffn_down.weight", fo6, yD);
    if (il == 0 && getenv("KINDI_DBG")) {
        CK(hipDeviceSynchronize());
        float f3[3] = {0,0,0};
        CK(hipMemcpy(f3, fo6.p, 12, hipMemcpyDeviceToHost));
        fprintf(stderr, "G ffn_swiglu_folded: %.4f %.4f %.4f\n", f3[0], f3[1], f3[2]);
        float y4[3] = {0,0,0};
        CK(hipMemcpy(y4, yD.p, 12, hipMemcpyDeviceToHost));
        fprintf(stderr, "G ffn_out: %.4f %.4f %.4f\n", y4[0], y4[1], y4[2]);
    }
    pending = true;
};

// ---------------------------------------------------------------- one token
auto body = [&](int token, int pos) {
    // embedding: token_embd is inverse-Hadamard, so un-fold into h
    {
        ktrace::Launch L = ktrace::begin("deq_emb", m.D / m.HB, 1, 1, 256, (size_t)m.D * 4);
        deq_emb<<<m.D / m.HB, 256, (size_t)m.HB * 4>>>((const uint*)m.w["token_embd.weight"].p,
                                                       (const int*)token_d.p, (const float*)sgn(m.D).p,
                                                       (float*)h.p, m.D, m.HB, m.hbscale);
        ktrace::end(L);
    }
    if (pos == 0 && getenv("KINDI_DBG")) {
        CK(hipDeviceSynchronize());
        float hh[3] = {0, 0, 0};
        CK(hipMemcpy(hh, h.p, 12, hipMemcpyDeviceToHost));
        fprintf(stderr, "G embed_h: %.4f %.4f %.4f\n", hh[0], hh[1], hh[2]);
    }
    int kvi = 0;
    for (int il = 0; il < m.NL; ++il) {
        if ((il + 1) % m.interval != 0) {
            gdn_layer(il, pos);
        } else {
            attn_layer(il, pos, kvi);
            kvi++;
        }
        ffn_layer(il);
        // The residual is NOT added here. The next layer's rmsfold adds it, driven by
        // `pending`, exactly as the reference engine does. Adding it here as well doubled
        // every layer output.
    }
};

// ---------------------------------------------------------------- one step
auto step = [&](int token, int pos) -> int {
    body(token, pos);
    // final norm + head
    rmsfold_l(h, yD, xn, xp, "output_norm.weight", m.D, pending);
    pending = false;
    gemv(head_name, xp, logits);
    {
        ktrace::Launch L = ktrace::begin("argmax", 1, 1, 1, 256, (size_t)m.VOCAB * 4);
        argmax<<<1, 256>>>((const float*)logits.p, (int*)argout.p, m.VOCAB);
        ktrace::end(L);
    }
    CK(hipDeviceSynchronize());
    // argout is DEVICE memory. Reading it from the host faults (0xC0000005); the result
    // has to be copied back first.
    int host_arg = 0;
    CK(hipMemcpy(&host_arg, argout.p, 4, hipMemcpyDeviceToHost));
    return host_arg;
};
