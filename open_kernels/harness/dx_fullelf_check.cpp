// dx_fullelf_check: run designs/dense/dx.py's full ELF over every layer of a model for
// tokens 0..N-1 (teacher-forced: token t's input is make_decode.py's xres{t}.bin) and
// compare each token's residual after layer 0 and after the last layer with the fp64
// reference make_decode.py wrote (ref_res{l}[_t{t}].bin).
//
//   dx_fullelf_check --rt  <dx_rt.elf>  <data_dir> <ntok> [opts]   one ELF, position per dispatch
//   dx_fullelf_check --pos <elf_dir>    <data_dir> <ntok> [opts]   dx_pos{t}.elf per position
//
// --rt reads <elf>.params.txt (build_full_elf.py --rtpos) and writes dx.py's RTPOS_PARAMS
// into each run's control scratchpad before every dispatch:
//   dx_kv_row_off = t * kv_row, dx_ptab_off = t * ptab_row, dx_win_extra = max(t, 1) - 1.
// opts: --layers L (default: every pool_L*.bin), --passes P (default 2: the whole token
// sequence is run P times; a later pass rewrites each KV row before any window reads it,
// so every pass must produce the same bytes), --dump DIR (final residual per token),
// --kv-row B --ptab-row B --kv-bytes B --act-bytes B (default: Qwen3-0.6B's layout).
// Build: g++ -std=c++20 -O2 -I/opt/xilinx/xrt/include dx_fullelf_check.cpp
//        -L/opt/xilinx/xrt/lib -lxrt_coreutil -o dx_fullelf_check
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>

static std::vector<char> slurp(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + p);
  return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
static bool exists(const std::string& p) { return std::ifstream(p).good(); }
static std::string sfx(int t) { return t ? "_t" + std::to_string(t) : ""; }

static double cosine(const float* a, const float* b, size_t n) {
  double ab = 0, aa = 0, bb = 0;
  for (size_t i = 0; i < n; ++i) { ab += (double)a[i] * b[i]; aa += (double)a[i] * a[i]; bb += (double)b[i] * b[i]; }
  return ab / std::sqrt(aa * bb);
}
static double maxabs(const float* v, size_t n) {
  double m = 0;
  for (size_t i = 0; i < n; ++i) m = std::max(m, (double)std::fabs(v[i]));
  return m;
}

int main(int argc, char** argv) {
  if (argc < 5) { fprintf(stderr, "usage: see the header of dx_fullelf_check.cpp\n"); return 2; }
  const std::string mode = argv[1], elf = argv[2], data = argv[3];
  const int ntok = std::stoi(argv[4]);
  const bool rt = mode == "--rt";
  int layers = -1, passes = 2;
  size_t kv_row = 4096, ptab_row = 2048, kv_bytes = 16u << 20, act_bytes = 65536;
  std::string dump;
  for (int i = 5; i + 1 < argc; i += 2) {
    std::string k = argv[i], v = argv[i + 1];
    if (k == "--layers") layers = std::stoi(v);
    else if (k == "--passes") passes = std::stoi(v);
    else if (k == "--dump") dump = v;
    else if (k == "--kv-row") kv_row = std::stoul(v);
    else if (k == "--ptab-row") ptab_row = std::stoul(v);
    else if (k == "--kv-bytes") kv_bytes = std::stoul(v);
    else if (k == "--act-bytes") act_bytes = std::stoul(v);
    else { fprintf(stderr, "unknown option %s\n", k.c_str()); return 2; }
  }
  if (layers < 0) for (layers = 0; exists(data + "/pools/pool_L" + std::to_string(layers) + ".bin"); ++layers) {}
  const size_t max_ctx = kv_bytes / kv_row;
  if ((size_t)ntok > max_ctx) { fprintf(stderr, "ntok %d > max_ctx %zu\n", ntok, max_ctx); return 2; }

  // Scratchpad indices (rt mode).
  std::map<std::string, int> pidx;
  if (rt) {
    std::ifstream pf(elf + ".params.txt");
    if (!pf) { fprintf(stderr, "no %s.params.txt\n", elf.c_str()); return 2; }
    int n; pf >> n;
    for (int i = 0; i < n; ++i) { std::string name, ty, kind; int idx; pf >> name >> idx >> ty >> kind; pidx[name] = idx; }
    for (const char* need : {"dx_kv_row_off", "dx_ptab_off", "dx_win_extra"})
      if (!pidx.count(need)) { fprintf(stderr, "params.txt lacks %s\n", need); return 2; }
  }

  xrt::device dev(0);
  auto bo_from = [&](size_t bytes, const std::vector<char>* init) {
    xrt::bo b = xrt::ext::bo{dev, bytes};
    char* m = b.map<char*>();
    std::memset(m, 0, bytes);
    if (init) {
      if (init->size() > bytes) throw std::runtime_error("init larger than buffer");
      std::memcpy(m, init->data(), init->size());
    }
    b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    return b;
  };

  // Buffers: per layer pool / consts / kv; shared xres / act / ptab (layers run in order).
  auto ptab_v = slurp(data + "/ptab.bin");
  xrt::bo ptab = bo_from(ptab_v.size(), &ptab_v);
  const size_t hid_bytes = slurp(data + "/xres0.bin").size();
  xrt::bo xres = bo_from(hid_bytes, nullptr);
  const size_t hid = hid_bytes / 4;
  xrt::bo act = bo_from(act_bytes, nullptr);
  std::vector<xrt::bo> pool, consts, kv;
  for (int l = 0; l < layers; ++l) {
    auto p = slurp(data + "/pools/pool_L" + std::to_string(l) + ".bin");
    auto c = slurp(data + "/consts_" + std::to_string(l) + ".bin");
    pool.push_back(bo_from(p.size(), &p));
    consts.push_back(bo_from(c.size(), &c));
    kv.push_back(bo_from(kv_bytes, nullptr));
  }
  printf("%s %s: %d layers, %d tokens, hid %zu, max_ctx %zu\n", rt ? "rtpos" : "per-position", elf.c_str(),
         layers, ntok, hid, max_ctx);

  // Contexts / kernels.
  std::map<int, std::unique_ptr<xrt::hw_context>> ctx;
  std::map<int, std::unique_ptr<xrt::ext::kernel>> krn;
  auto kernel_for = [&](int t) -> xrt::ext::kernel& {
    int key = rt ? 0 : t;
    if (!krn.count(key)) {
      std::string p = rt ? elf : elf + "/dx_pos" + std::to_string(t) + ".elf";
      auto fb = slurp(p);
      ctx[key] = std::make_unique<xrt::hw_context>(dev, xrt::elf(fb.data(), fb.size()));
      krn[key] = std::make_unique<xrt::ext::kernel>(*ctx[key], "main:sequence");
    }
    return *krn[key];
  };

  // rt: one run per layer, args bound once, scratchpad rewritten per token.
  std::vector<xrt::run> runs;
  std::vector<uint32_t*> sp;
  std::vector<xrt::bo> sp_bo;
  auto bind = [&](xrt::run& r, int l) {
    r.set_arg(0, pool[l]); r.set_arg(1, xres); r.set_arg(2, consts[l]);
    r.set_arg(3, kv[l]); r.set_arg(4, act); r.set_arg(5, ptab);
  };
  if (rt) {
    auto& k = kernel_for(0);
    for (int l = 0; l < layers; ++l) {
      runs.emplace_back(k);
      bind(runs.back(), l);
      sp_bo.push_back(runs.back().get_ctrl_scratchpad_bo());
      sp.push_back(sp_bo.back().map<uint32_t*>());
    }
    printf("scratchpad: %zu bytes per run; idx row_off %d ptab_off %d win_extra %d\n", sp_bo[0].size(),
           pidx["dx_kv_row_off"], pidx["dx_ptab_off"], pidx["dx_win_extra"]);
  }

  std::vector<float> out(hid), ref(hid), first(hid);
  std::vector<std::vector<float>> prev(ntok);
  int bad = 0;
  for (int pass = 0; pass < passes; ++pass) {
    double worst = 1.0;
    for (int t = 0; t < ntok; ++t) {
      auto x = slurp(data + "/xres" + std::to_string(t) + ".bin");
      std::memcpy(xres.map<char*>(), x.data(), x.size());
      xres.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      const uint32_t nf = t > 1 ? t : 1;
      for (int l = 0; l < layers; ++l) {
        if (rt) {
          sp[l][pidx["dx_kv_row_off"]] = (uint32_t)(t * kv_row);
          sp[l][pidx["dx_ptab_off"]] = (uint32_t)(t * ptab_row);
          sp[l][pidx["dx_win_extra"]] = nf - 1;
          sp_bo[l].sync(XCL_BO_SYNC_BO_TO_DEVICE);
          runs[l].start();
          runs[l].wait2();
        } else {
          xrt::run r(kernel_for(t));
          bind(r, l);
          r.start();
          r.wait2();
        }
        if (l == 0 || l == layers - 1) {
          xres.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
          std::memcpy(out.data(), xres.map<char*>(), hid_bytes);
          auto rf = slurp(data + "/ref_res" + std::to_string(l) + sfx(t) + ".bin");
          std::memcpy(ref.data(), rf.data(), hid_bytes);
          double c = cosine(out.data(), ref.data(), hid);
          if (l == 0) first = out;
          if (l == layers - 1) {
            double c0 = cosine(first.data(), (const float*)slurp(data + "/ref_res0" + sfx(t) + ".bin").data(), hid);
            bool same = pass == 0 || std::memcmp(prev[t].data(), out.data(), hid_bytes) == 0;
            printf("pass %d pos %3d nf %3u  layer0 cos %.7f  layer%d cos %.7f  |res| %.2f vs %.2f%s\n", pass, t,
                   nf, c0, l, c, maxabs(out.data(), hid), maxabs(ref.data(), hid), same ? "" : "  DIFFERS from pass 0");
            if (!same) ++bad;
            worst = std::min(worst, std::min(c, c0));
            prev[t] = out;
            if (!dump.empty()) {
              std::ofstream o(dump + "/npu_res" + std::to_string(l) + sfx(t) + ".bin", std::ios::binary);
              o.write((const char*)out.data(), hid_bytes);
            }
          }
        }
      }
    }
    printf("pass %d: worst cosine %.7f\n", pass, worst);
  }
  printf("%s\n", bad ? "PASSES DIFFER" : "passes byte-identical");
  return bad ? 1 : 0;
}
