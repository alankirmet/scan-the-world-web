"use strict";
/* Scan the World — test app (v0). Phone-first PWA.
   Flow: home -> guided orbit capture (on-phone blur gate, same threshold
   as scanner-core's client) -> one bundle file -> share to Homie ->
   result appears under the scan code once the pipeline has run. */

// ---------- small helpers ----------
const $ = id => document.getElementById(id);
const screens = ["home", "capture", "review", "groundtruth", "send", "result"];
function show(name) {
  screens.forEach(s => $("screen-" + s).classList.toggle("hidden", s !== name));
  window.scrollTo(0, 0);
}
function loadScans() { try { return JSON.parse(localStorage.getItem("stw-scans") || "[]"); } catch { return []; } }
function saveScans(list) { localStorage.setItem("stw-scans", JSON.stringify(list)); }
function newCode() {
  const t = Date.now().toString(36).toUpperCase().replace(/[^A-Z0-9]/g, "");
  return t.slice(-4) + Math.floor(Math.random() * 1296).toString(36).toUpperCase().padStart(2, "0");
}
function fmtDate(ts) { return new Date(ts).toLocaleDateString(undefined, { month: "short", day: "numeric" }); }

// ---------- home ----------
function renderScanList() {
  const list = loadScans();
  const el = $("scan-list");
  if (!list.length) { el.innerHTML = '<p class="muted">No scans yet.</p>'; return; }
  el.innerHTML = "";
  list.slice().reverse().forEach(s => {
    const b = document.createElement("button");
    b.className = "scan-item";
    b.innerHTML = '<span class="code"></span><span class="status"></span>';
    b.querySelector(".code").textContent = s.code;
    b.querySelector(".status").textContent = fmtDate(s.ts) + " · tap for result";
    b.onclick = () => openResult(s.code);
    el.appendChild(b);
  });
}
$("btn-code").onclick = () => {
  const code = $("code-input").value.trim().toUpperCase();
  if (code) openResult(code);
};

// ---------- capture ----------
// ORBIT CAPTURE v2 (2026-11-07). One press of Start, then ONE
// continuous motion: the app auto-captures while the user walks
// around the object — no per-photo staging, no shutter taps, no
// hold-still beats. Two passes: pass 1 level, pass 2 a little
// higher. The progress bar is PASS progress (good photos kept in
// this pass) — never a coverage claim.
//
// THE GATE IS THE SERVER'S GATE. The pure block below is a 1:1 port
// of scanner-core's gates.py check_frame — the gate the measuring
// pipeline itself applies: grayscale Laplacian variance at NATIVE
// resolution (no downscale), plus exposure-mean, highlight-clip
// (0.25 — verification round 2 recalibrated this from 0.08 after
// the near-white mat tripped the old line) and shadow-clip checks.
// The frame that is gated is the exact canvas that is encoded into
// the bundle, so the phone and the server score the same pixels and
// can no longer disagree the way they did on MZEN08 (v1's phone
// gate scored a stretched 320x240 preview; the server scored the
// real frames 5-30 against the 60 line and dropped nearly all of
// them). Parity is proven headlessly in tools/gate_parity/ — the
// harness extracts this exact block and runs it against gates.py
// over the MZEN08 and DEMO1 frame sets.
//
// Frames that fail the gate are dropped LIVE with coaching, never
// silently kept. Smart budget: pass target 12 kept, pass cap 16,
// scan target 24, hard cap 32 — at the end, an over-full scan is
// thinned to the sharpest frame in each slice of the walk, so the
// kept set stays spread around the object.
//
// Honest platform notes (iOS Safari): focus/exposure lock modes are
// attempted through track constraints where the browser exposes
// them and the manifest records what ACTUALLY engaged (read back
// from getSettings) — "unsupported" is recorded as unsupported,
// never dressed up. IMU data is metadata only: it never gates and
// never implies size.

// ==== SHARED PURE BLOCK — no DOM; extracted verbatim by the parity
// ==== harness (tools/gate_parity). Keep it dependency-free.
const GATE_LIMITS = {
  SHARPNESS_MIN: 60.0,        // Laplacian variance, native res (gates.py)
  EXPOSURE_MEAN_MIN: 25.0,
  EXPOSURE_MEAN_MAX: 230.0,
  HIGHLIGHT_CLIP_MAX: 0.25,   // fraction of pixels >= 250 (was 0.08 in v1;
                              // VR2: healthy frames run 0.000-0.156, median
                              // 0.083 — the old line sat at the median)
  SHADOW_CLIP_MAX: 0.35       // fraction of pixels <= 5
};

// BT.601 grayscale, fixed-point weights in the spirit of cv2's
// integer path: gray = (77R + 150G + 29B) >> 8.
function grayFromRGBA(data) {
  const n = data.length >>> 2;
  const gray = new Uint8Array(n);
  for (let i = 0, j = 0; j < n; i += 4, j++)
    gray[j] = (77 * data[i] + 150 * data[i + 1] + 29 * data[i + 2]) >> 8;
  return gray;
}

// Exact port of gates.py check_frame on a grayscale array.
// Laplacian: 4-neighbour kernel, variance over ALL pixels with
// reflect-101 borders — the same pixel set cv2.Laplacian(...).var()
// uses, so the statistic matches, not just the threshold.
function gateFromGray(gray, w, h) {
  const n = w * h;
  let gsum = 0, hi = 0, lo = 0;
  for (let i = 0; i < n; i++) {
    const v = gray[i];
    gsum += v;
    if (v >= 250) hi++; else if (v <= 5) lo++;
  }
  const mean = gsum / n;
  let sum = 0, sum2 = 0;
  for (let y = 0; y < h; y++) {
    const yu = y === 0 ? 1 : y - 1, yd = y === h - 1 ? h - 2 : y + 1;
    const ru = yu * w, rc = y * w, rd = yd * w;
    for (let x = 0; x < w; x++) {
      const xl = x === 0 ? 1 : x - 1, xr = x === w - 1 ? w - 2 : x + 1;
      const v = gray[ru + x] + gray[rd + x] + gray[rc + xl] + gray[rc + xr]
              - 4 * gray[rc + x];
      sum += v; sum2 += v * v;
    }
  }
  const lm = sum / n;
  const sharpness = sum2 / n - lm * lm;
  const highlightFrac = hi / n, shadowFrac = lo / n;
  const failures = [];
  if (sharpness < GATE_LIMITS.SHARPNESS_MIN) failures.push("blurry");
  if (mean < GATE_LIMITS.EXPOSURE_MEAN_MIN) failures.push("too dark");
  if (mean > GATE_LIMITS.EXPOSURE_MEAN_MAX) failures.push("too bright");
  if (highlightFrac > GATE_LIMITS.HIGHLIGHT_CLIP_MAX) failures.push("highlights clipped");
  if (shadowFrac > GATE_LIMITS.SHADOW_CLIP_MAX) failures.push("shadows clipped");
  return { sharpness, exposureMean: mean, highlightFrac, shadowFrac,
           passed: failures.length === 0, failures };
}

// End-of-capture thinning: split the kept list (in walk order) into
// `target` contiguous slices and keep the sharpest frame of each —
// the sharpest set that stays spread around the whole orbit.
function selectKeptIndices(metas, target) {
  if (metas.length <= target) return metas.map((_, i) => i);
  const idx = [];
  for (let b = 0; b < target; b++) {
    const start = Math.floor(b * metas.length / target);
    const end = Math.floor((b + 1) * metas.length / target);
    let best = start;
    for (let i = start + 1; i < end; i++)
      if (metas[i].sharpness > metas[best].sharpness) best = i;
    idx.push(best);
  }
  return idx;
}

// The scan.json manifest for a v2 bundle. Same contract as before
// (code / paper / n_frames / frames_meta — run_bundle.py consumes it
// unchanged), extended with the capture record: what the camera
// actually delivered, what the locks actually did, per-pass counts.
function buildManifestV2(info) {
  return {
    code: info.code,
    paper: "LETTER",
    n_frames: info.frames.length,
    made_with: "Scan the World test app v2 (orbit capture)",
    capture: {
      version: 2,
      style: "continuous orbit, two passes (level, then higher)",
      gate: {
        definition: "port of scanner-core gates.py check_frame — native-resolution grayscale Laplacian variance + exposure/clipping checks; the gated canvas is the bundled frame",
        sharpness_min: GATE_LIMITS.SHARPNESS_MIN,
        exposure_mean_min: GATE_LIMITS.EXPOSURE_MEAN_MIN,
        exposure_mean_max: GATE_LIMITS.EXPOSURE_MEAN_MAX,
        highlight_clip_max: GATE_LIMITS.HIGHLIGHT_CLIP_MAX,
        shadow_clip_max: GATE_LIMITS.SHADOW_CLIP_MAX
      },
      stream: info.stream,
      locks: info.locks,
      passes: info.passes,
      attempts: info.attempts,
      gate_dropped: info.gateDropped,
      selection_dropped: info.selectionDropped,
      target_frames: info.targetFrames,
      cap_frames: info.capFrames
    },
    frames_meta: info.frames
  };
}
// ==== END SHARED PURE BLOCK ====

// ==== V32 PURE BLOCK — no DOM; extracted by the v3 proof harness
// ==== (tools/v3_proof) for its unit exercises. Ports of native
// ==== v1.3's initiation machinery (app-native/src/lib/
// ==== scaleLock.ts, segmentation.ts, provenance.ts, coverage.ts):
// ==== the same rules and the same record shapes, so the runner
// ==== and the evidence ledger read web and native bundles alike.
// ==== The theory is ORIGIN-TAP-THEORY-2026-11-09.md §§2-5, 13.
// ==== v3.4: the initiation record carries native v1.3.6's
// ==== aim-dot targeting (see buildInitiationRecord below).

// --- denominations: DECLARED by the user, never detected ---
const COIN_DENOMINATIONS = [
  { id: "quarter-us", name: "US quarter", diameterMm: 24.26 },
  { id: "nickel-us", name: "US nickel", diameterMm: 21.21 },
  { id: "penny-us", name: "US penny", diameterMm: 19.05 },
  { id: "dime-us", name: "US dime", diameterMm: 17.91 }
];
function denominationById(id) {
  return COIN_DENOMINATIONS.find(d => d.id === id) || COIN_DENOMINATIONS[0];
}

// --- the scale lock (port of scaleLock.ts) ---
// Reads arrive one still at a time. A read is the ellipse major
// axis of the region grown from the user's scale tap, in pixels
// of the still the read came from. The lock fires when
// LOCK_READS_REQUIRED successive reads agree within ±LOCK_AGREEMENT
// on the major axis. A read that disagrees with the previous one
// by more than the agreement BREAKS the streak (counted) and
// starts a new one; drift inside a streak slides the window.
// A null read (no region found on that still) is neutral: the
// attempt happened (the caller counts it), the streak is untouched.
const LOCK_AGREEMENT = 0.02;
const LOCK_READS_REQUIRED = 3;
class ScaleLockMachine {
  constructor(denomination) {
    this.denomination = denomination;
    this.state = "seeking"; // 'seeking' | 'locked' | 'shape-only'
    this.streak = [];
    this.breaks = 0;
    this.allReads = [];
    this.lockedPxPerMm = null;
    this.note = null;
  }
  get currentState() { return this.state; }
  get readCount() { return this.allReads.length; }
  get streakCount() { return this.streak.length; }
  feed(read) {
    if (this.state !== "seeking") return;
    if (!read) return; // neutral: attempt counted by the caller
    this.allReads.push(read);
    const last = this.streak[this.streak.length - 1];
    if (last && Math.abs(read.majorPx - last.majorPx) / last.majorPx > LOCK_AGREEMENT) {
      this.breaks++;
      this.streak = [read];
      return;
    }
    this.streak.push(read);
    // Slide: drop the oldest reads until the whole streak agrees.
    while (this.streak.length > 1) {
      const first = this.streak[0], lastR = this.streak[this.streak.length - 1];
      if (Math.abs(lastR.majorPx - first.majorPx) / first.majorPx <= LOCK_AGREEMENT) break;
      this.streak.shift();
    }
    if (this.streak.length >= LOCK_READS_REQUIRED) {
      const mean = this.streak.reduce((a, r) => a + r.majorPx, 0) / this.streak.length;
      this.lockedPxPerMm = mean / this.denomination.diameterMm;
      this.state = "locked";
      this.note = "Scale locked: " + LOCK_READS_REQUIRED + " reads of the declared " +
        this.denomination.name + " (" + this.denomination.diameterMm +
        " mm) agreed within ±2% on the ellipse major axis.";
    }
  }
  giveUp(reason) {
    if (this.state === "locked") return;
    this.state = "shape-only";
    this.note = reason;
  }
  userWord() {
    if (this.state === "locked") {
      return "Scale locked — the " + this.denomination.name + " gave " +
        (Math.round(this.lockedPxPerMm * 100) / 100) +
        " px per mm. The whole scan inherits this size.";
    }
    if (this.state === "shape-only") {
      return "Shape only this scan — no size was locked, and the scan will say so.";
    }
    return "Keep adjusting — the size hasn't locked yet.";
  }
  record() {
    return {
      declared: {
        id: this.denomination.id, name: this.denomination.name,
        diameter_mm: this.denomination.diameterMm, source: "user-declared"
      },
      state: this.state,
      reads: this.allReads.map(r => ({
        major_px: Math.round(r.majorPx * 10) / 10,
        minor_px: Math.round(r.minorPx * 10) / 10,
        at_epoch_ms: r.atEpochMs
      })),
      scatter_breaks: this.breaks,
      locked_px_per_mm: this.lockedPxPerMm == null ? null
        : Math.round(this.lockedPxPerMm * 10000) / 10000,
      note: this.note
    };
  }
}

// --- seeded segmentation (port of segmentation.ts) ---
// Nearest-neighbour downscale to an analysis frame of at most
// maxDim on the long edge (the native engine's ≤512 px frame).
// Returns factor = full width / small width, the native convention:
// small coords = full coords / factor; small px * factor = full px.
function downscaleGray(rgba, w, h, maxDim) {
  const gray = grayFromRGBA(rgba);
  const longEdge = Math.max(w, h);
  if (longEdge <= maxDim) return { gray, w, h, factor: 1 };
  const s = maxDim / longEdge;
  const sw = Math.max(1, Math.round(w * s)), sh = Math.max(1, Math.round(h * s));
  const small = new Uint8Array(sw * sh);
  for (let y = 0; y < sh; y++) {
    const sy = Math.min(h - 1, Math.floor(y / s));
    for (let x = 0; x < sw; x++) {
      const sx = Math.min(w - 1, Math.floor(x / s));
      small[y * sw + x] = gray[sy * w + sx];
    }
  }
  return { gray: small, w: sw, h: sh, factor: w / sw };
}

function patchMean(gray, w, h, cx, cy, r) {
  let sum = 0, n = 0;
  for (let y = Math.max(0, cy - r); y <= Math.min(h - 1, cy + r); y++) {
    for (let x = Math.max(0, cx - r); x <= Math.min(w - 1, cx + r); x++) {
      sum += gray[y * w + x]; n++;
    }
  }
  return n ? sum / n : 0;
}

// Flood-fill the connected region around the seed whose pixels sit
// within `tolerance` gray levels of the seed patch's mean.
// Deliberately the native test-build grow: a desk of similar tone
// can swallow the region — that failure is meant to be VISIBLE in
// the recorded boundary (or its recorded absence), never hidden.
function growRegion(gray, w, h, sx, sy, tolerance) {
  const tol = tolerance == null ? 24 : tolerance;
  const mask = new Uint8Array(w * h);
  const seedX = Math.max(0, Math.min(w - 1, Math.round(sx)));
  const seedY = Math.max(0, Math.min(h - 1, Math.round(sy)));
  const target = patchMean(gray, w, h, seedX, seedY, 2);
  const stack = [seedY * w + seedX];
  mask[seedY * w + seedX] = 1;
  let area = 0, x0 = w, y0 = h, x1 = -1, y1 = -1;
  while (stack.length) {
    const i = stack.pop();
    area++;
    const x = i % w, y = (i / w) | 0;
    if (x < x0) x0 = x; if (x > x1) x1 = x;
    if (y < y0) y0 = y; if (y > y1) y1 = y;
    const nb = [x > 0 ? i - 1 : -1, x < w - 1 ? i + 1 : -1,
                y > 0 ? i - w : -1, y < h - 1 ? i + w : -1];
    for (const j of nb) {
      if (j >= 0 && !mask[j] && Math.abs(gray[j] - target) <= tol) {
        mask[j] = 1; stack.push(j);
      }
    }
  }
  return { mask, areaPx: area, bbox: area > 0 ? { x0, y0, x1, y1 } : null };
}

// Second-moment ellipse of a mask: axes = 4σ along the principal
// directions — for a filled ellipse this recovers its full axes,
// at any tilt (the major axis is the true diameter of a tilted
// coin; theory §3). Returns axes in mask-frame pixels, or null.
function ellipseAxes(mask, w, h) {
  let n = 0, mx = 0, my = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    if (mask[y * w + x]) { n++; mx += x; my += y; }
  }
  if (n < 8) return null;
  mx /= n; my /= n;
  let sxx = 0, syy = 0, sxy = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    if (mask[y * w + x]) {
      const dx = x - mx, dy = y - my;
      sxx += dx * dx; syy += dy * dy; sxy += dx * dy;
    }
  }
  sxx /= n; syy /= n; sxy /= n;
  const tr = sxx + syy, det = sxx * syy - sxy * sxy;
  const disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
  const l1 = tr / 2 + disc, l2 = Math.max(0, tr / 2 - disc);
  return { majorPx: 4 * Math.sqrt(l1), minorPx: 4 * Math.sqrt(l2) };
}

// --- region provenance (port of provenance.ts; theory §13) ---
// Every region of the finished model carries where it came from.
// A FILLED region is closed algorithmically: it makes the trophy
// read whole and must NEVER feed a dimension.
function buildRegionProvenance(input) {
  const sp = input.secondPass;
  let bottom;
  if (sp && sp.completed && sp.kind === "flip") {
    bottom = {
      region: "bottom", source: "pass-2", pass_kind: "flip",
      feeds_dimension: true,
      note: "Measured in the flip pass — the object was turned over."
    };
  } else if (sp && sp.completed && sp.kind === "bottom") {
    bottom = {
      region: "bottom", source: "pass-2", pass_kind: "bottom",
      feeds_dimension: true,
      note: "Measured in the scan-bottom pass — the object was tilted and the pass was placed as the bottom."
    };
  } else if (input.applyFill) {
    bottom = {
      region: "bottom", source: "filled", pass_kind: null,
      feeds_dimension: false,
      note: "Never seen by any pass. Closed algorithmically to the support plane so the trophy reads whole. FILLED — not measured, never feeds a dimension."
    };
  } else {
    bottom = {
      region: "bottom", source: "unseen", pass_kind: null,
      feeds_dimension: false,
      note: "Not captured and not filled — the model is open underneath."
    };
  }
  const regions = [
    { region: "top", source: "pass-1", pass_kind: null, feeds_dimension: true,
      note: "Seen from above in the primary capture." },
    { region: "sides", source: "pass-1", pass_kind: null, feeds_dimension: true,
      note: "Seen from around the object in the primary capture." },
    bottom
  ];
  return {
    regions,
    has_filled: regions.some(r => r.source === "filled"),
    fully_measured: regions.every(r => r.source === "pass-1" || r.source === "pass-2")
  };
}
function regionMayFeedDimension(prov, region) {
  const r = prov.regions.find(x => x.region === region);
  return !!r && r.feeds_dimension;
}

// --- coverage + intake (port of coverage.ts, guided sectors) ---
// The model grows by BANKING views; a gated photo that adds no new
// coverage is DISCARDED with a one-line trace. The browser offers
// no pose channel the app may honestly use for banking (the IMU
// turn figure stays a diversity proxy in frames_meta, never an
// azimuth), so banking runs in GUIDED-SECTOR mode exactly as
// native's: the app walks the user sector by sector and banks
// frames into the sector it is currently guiding. The snapshot
// records pose_source 'guided-sectors-no-device-pose' — guided
// mode is never dressed up as measured pose.
const COVERAGE_PARTS = ["front", "right", "back", "left", "top"];
const VIEWS_PER_PART = 3;
const GUIDANCE_WORDS = {
  front: "Start at the front — keep it filling half the screen.",
  right: "Now from the right side.",
  back: "Now the back.",
  left: "Now from the left side.",
  top: "Now a little higher — let's get the top."
};
class CoverageTracker {
  constructor() {
    this.banked = { top: 0, front: 0, right: 0, back: 0, left: 0 };
    this.intake = [];
    this.counter = 0;
    this.frozen = false; // no pose source exists to lose; kept for shape parity
  }
  currentPart() {
    for (const p of ["front", "right", "back", "left"]) {
      if (this.banked[p] < VIEWS_PER_PART) return p;
    }
    return "top";
  }
  partState(p) {
    if (this.banked[p] >= VIEWS_PER_PART) return "complete";
    return this.banked[p] > 0 ? "partial" : "missing";
  }
  isComplete() {
    return COVERAGE_PARTS.every(p => this.banked[p] >= VIEWS_PER_PART);
  }
  guidance() {
    if (this.isComplete()) return "That's the whole object from up here — the model says complete.";
    return GUIDANCE_WORDS[this.currentPart()];
  }
  // Offer one gated-kept view. passIndex 0 (level pass) banks into
  // the sector currently being guided; passIndex >= 1 (the higher
  // pass) banks to 'top'. A view for a complete part is discarded
  // 'no-new-coverage' — the kept photo still ships in the bundle;
  // the ledger records that it added no new coverage.
  offer(passIndex) {
    this.counter++;
    const part = passIndex >= 1 ? "top" : this.currentPart();
    if (this.banked[part] >= VIEWS_PER_PART) {
      return this.trace("discarded", "no-new-coverage", part);
    }
    this.banked[part]++;
    return this.trace("banked", "new-coverage", part);
  }
  // Trace a view that never reached banking (gate failure or
  // end-of-capture selection thinning) so the ledger accounts for
  // it. Gate-passed attempts skipped by the keep-pacing gap are
  // NOT traced here — the offer cadence is the capture loop's own;
  // they are counted in capture.attempts_pacing_skipped instead.
  traceOutside(reason) {
    this.counter++;
    return this.trace("discarded", reason, null);
  }
  // v3.3 (native v1.4.0 parity, coverage.ts traceGateFailure): a
  // gate failure is traced WITH its failure names — the intake
  // ledger names why every refused frame was refused, so nothing
  // is dropped silently and the named-problem feedback has its
  // record. Same entry shape as native's: outcome 'discarded',
  // reason 'gate-failed', plus the failures list.
  traceGateFailure(failures) {
    this.counter++;
    const e = { n: this.counter, outcome: "discarded", reason: "gate-failed", part: null };
    e.failures = failures.slice();
    this.intake.push(e);
    return e;
  }
  trace(outcome, reason, part) {
    const e = { n: this.counter, outcome, reason, part };
    this.intake.push(e);
    return e;
  }
  intakeEntries() { return this.intake.slice(); }
  snapshot() {
    const parts = {};
    for (const p of COVERAGE_PARTS) {
      parts[p] = { banked: this.banked[p], needed: VIEWS_PER_PART, state: this.partState(p) };
    }
    return {
      pose_source: "guided-sectors-no-device-pose",
      parts,
      declared_complete: this.isComplete(),
      frozen: this.frozen,
      banked_total: this.intake.filter(e => e.outcome === "banked").length,
      discarded_total: this.intake.filter(e => e.outcome === "discarded").length
    };
  }
}

// --- v3.3 additions (ports of native v1.4.0's incorporation
// --- build: coverageDisplay.ts, feedback.ts, bankedFeedback.ts,
// --- prune.ts — the same rules, the same words, so web and
// --- native bundles and HUDs read alike. The theory is
// --- INCORPORATION-PLAN-2026-11-10.md; the native report is
// --- app-native/COORDINATOR-REPORT-1.4.0.md.)

// Coverage segments (port of coverageDisplay.ts): the display
// model is a PURE function of the coverage snapshot — a segment
// is 'banked' only when the ledger counted a banked view into
// that slot. No segment fills for time, attempts, or pacing.
function coverageDisplayWeb(snap) {
  const order = ["front", "right", "back", "left", "top"];
  const labels = { front: "Front", right: "Right", back: "Back", left: "Left", top: "Top" };
  let currentPart = "top";
  for (const p of ["front", "right", "back", "left"]) {
    if (snap.parts[p].banked < snap.parts[p].needed) { currentPart = p; break; }
  }
  const parts = order.map(part => {
    const p = snap.parts[part];
    const segments = [];
    for (let i = 0; i < p.needed; i++) {
      if (i < p.banked) segments.push("banked");
      else if (!snap.frozen && part === currentPart && i === p.banked) segments.push("next");
      else segments.push("waiting");
    }
    return {
      part, label: labels[part], banked: p.banked, needed: p.needed, state: p.state,
      segments,
      word: p.state === "complete" ? "✓ done" : p.banked > 0 ? p.banked + " of " + p.needed : "not yet"
    };
  });
  return {
    parts,
    bankedTotal: snap.banked_total,
    totalSegments: parts.reduce((a, p) => a + p.needed, 0),
    complete: snap.declared_complete,
    frozen: snap.frozen
  };
}

// Named problems (port of feedback.ts): capture problems are a
// small set of NAMED states keyed by the gate's own failure
// names, each with what the app DOES while it persists. The
// advice words are the gate's COACHING lines, attached by the
// display layer from the app's COACHING table (same strings —
// the named state and the coaching can never disagree).
const NAMED_PROBLEMS_WEB = {
  "blurry": { id: "blurry", name: "Too blurry", behaviour: "Photos aren't banking while this lasts." },
  "too dark": { id: "too dark", name: "Too dark", behaviour: "Photos aren't banking while it's this dark — the Apple rule: stop, don't bank garbage." },
  "too bright": { id: "too bright", name: "Too bright", behaviour: "Photos aren't banking while this lasts." },
  "highlights clipped": { id: "highlights clipped", name: "Too much glare", behaviour: "Photos aren't banking while this lasts." },
  "shadows clipped": { id: "shadows clipped", name: "Deep shadows", behaviour: "Photos aren't banking while this lasts." }
};
// The dominant named problem in a window of recent failed
// attempts. Named only on recurrence (>= minCount of the same
// failure); ties break toward the most recent failure — the
// window's tail is what the user is doing NOW (native rule).
function dominantProblemWeb(recentFailures, minCount) {
  minCount = minCount == null ? 2 : minCount;
  const counts = {}, lastSeen = {};
  recentFailures.forEach((failures, i) => {
    for (const f of failures) {
      if (!NAMED_PROBLEMS_WEB[f]) continue;
      counts[f] = (counts[f] || 0) + 1;
      lastSeen[f] = i;
    }
  });
  let best = null, bestCount = 0, bestLast = -1;
  for (const f in counts) {
    const l = lastSeen[f];
    if (counts[f] > bestCount || (counts[f] === bestCount && l > bestLast)) {
      best = f; bestCount = counts[f]; bestLast = l;
    }
  }
  if (!best || bestCount < minCount) return null;
  return NAMED_PROBLEMS_WEB[best];
}

// Banked-frame feedback plan (port of bankedFeedback.ts): the
// DECISION half only — the effect sink never guesses. Tone rule:
// chirps ON ⇒ the frame's own pose chirp IS the tone ('pose-
// chirp', never a second sound over the protocol); chirps OFF ⇒
// a soft UI tick ('ui-tick'). Haptics: a plain bank is a tick, a
// milestone (part / coverage complete) is the stronger buzz —
// where the platform has no vibration (iOS Safari), the sink's
// guarded call is a no-op and the gold pulse still shows.
function planBankedFeedbackWeb(kind, chirpsOn) {
  const tone = chirpsOn ? "pose-chirp" : "ui-tick";
  if (kind === "banked") return { haptic: "tick", tone, pulse: true };
  return { haptic: "milestone", tone, pulse: true };
}

// Pruning helpers (port of prune.ts): drop frames from the flat
// kept list by index, returning the kept indices and the dropped
// frames' records in the PRE-prune numbering the review showed.
function applyPruningWeb(metas, drop) {
  const keptIdx = [], pruned = [];
  metas.forEach((m, i) => {
    if (drop.has(i)) pruned.push({ frame: i, pass: m.pass, sharpness: m.sharpness });
    else keptIdx.push(i);
  });
  return { keptIdx, pruned };
}
// Remap a session record's pose chirps onto post-prune frame
// numbering: chirps whose frame was dropped are un-paired and
// counted, never left pointing at a stale frame (the same
// discipline WebChirpSession.remapPoseChirps applies at
// thinning time, here over the plain record at prune time).
function remapPoseChirpRecordsWeb(poseChirps, oldToNew) {
  let dropped = 0;
  const records = [];
  for (const c of poseChirps) {
    const n = oldToNew.get(c.frame_index);
    if (n == null) { dropped++; continue; }
    records.push(Object.assign({}, c, { frame_index: n }));
  }
  return { records, dropped };
}

// The initiation record (native manifest shape). v3.4 — native
// v1.3.6 parity: the centre aim dot IS the target, so the origin
// is the dot's position (the frame centre by construction) and
// `targeting` says how it was set. The initiation still is the
// record's evidence; still_file is null when the still chain
// failed before a photo existed — recorded, never faked.
function buildInitiationRecord(p) {
  const r4 = v => Math.round(v * 10000) / 10000;
  return {
    origin: { x: r4(p.origin.x), y: r4(p.origin.y) },
    targeting: "aim-dot",
    // v3.5 (method gating, native parity): the scale stage's
    // place in this scan's flow, in the native record's own
    // vocabulary — 'scale-tap' when the method carries the tap
    // stage, 'not-part-of-this-method' when the method has no
    // coin stage at all (phone). A caller that never says (the
    // pre-v3.5 harness shape) keeps the field off the record.
    ...(p.scaleStage ? { scale_stage: p.scaleStage } : {}),
    // An explicit stillFile of null means the chain failed
    // before a photo existed — that null must survive; only a
    // MISSING stillFile (a caller that never says) defaults.
    still_file: p.stillFile !== undefined ? p.stillFile : "initiation.jpg",
    boundary: p.boundary,
    boundary_note: p.boundaryNote,
    scale_tap: p.scaleTap ? { x: r4(p.scaleTap.x), y: r4(p.scaleTap.y) } : null,
    scale_still_file: p.scaleTap ? "scale-still.jpg" : null,
    scene_live_until: p.sceneLiveUntil
  };
}
// ==== END V32 PURE BLOCK ====

// ---------- audio (v3, 2026-11-09) ----------
// WEB CAPTURE v3: the browser build of the native app's chirp
// session (app-native v1.1/v1.2). Same frozen emission spec
// (audio-framework/BASELINE-PROTOCOL.md §7), same session shape:
//   baseline: latency-lock train x5 -> 0.5 s silence window ->
//             background train x8, in one held pose over the empty
//             table, recorder already running;
//   measurement: ONE chirp per kept frame (v1.2 per-pose), slot-
//             guarded one chirp period apart, each paired to its
//             frame in audio_session.pose_chirps — the manifest
//             block mirrors the native manifest's shape field for
//             field so tools/run_bundle.py and the server-side
//             estimator read it unchanged.
// Playback and recording share ONE AudioContext: chirps are
// scheduled on the context clock and the recorder timestamps its
// samples on the same clock (AudioWorklet currentFrame; a
// ScriptProcessor fallback estimates the anchor and FLAGS it).
// Requested-vs-actual honesty, native-style: the mic constraints
// ask for processing OFF; what the browser actually granted is
// read back from track.getSettings() and recorded verbatim.
// pipeline:"browser-web" rides on every manifest — a browser
// datum must never be mistaken for a native-pipeline datum.

// ==== AUDIO PURE BLOCK — no DOM; extracted by the v3 proof
// ==== harness (tools/v3_proof). Keep it dependency-free.
const CHIRP_SPEC = {
  waveform: "linear FM up-chirp, Hann-windowed",
  f0_hz: 1000, f1_hz: 16000, duration_ms: 10, window: "Hann",
  period_ms: 100, lock_chirps: 5, background_chirps: 8, silence_ms: 500,
  level_dbfs_peak: -20, floor_multiplier: 4.5, default_temp_c: 20
};
const AUDIO_REQUESTED_RATE_HZ = 48000;
const AUDIO_CONSTRAINTS_ASKED = { echoCancellation: false, noiseSuppression: false, autoGainControl: false };

function round3(v) { return Math.round(v * 1000) / 1000; }

function soundSpeedMps(tempC) { return 331.3 * Math.sqrt(1 + tempC / 273.15); }

// One chirp, peak-normalised — a line-for-line port of native
// lib/chirp.ts makeChirp (same phase integral, same Hann).
function makeChirpSamples(sampleRate, peak) {
  peak = peak == null ? 1 : peak;
  const n = Math.round((CHIRP_SPEC.duration_ms / 1000) * sampleRate);
  const out = new Float32Array(n);
  const T = n / sampleRate;
  for (let i = 0; i < n; i++) {
    const t = i / sampleRate;
    const w = 0.5 * (1 - Math.cos((2 * Math.PI * i) / Math.max(1, n - 1)));
    const phase = 2 * Math.PI * (CHIRP_SPEC.f0_hz * t +
      ((CHIRP_SPEC.f1_hz - CHIRP_SPEC.f0_hz) * t * t) / (2 * T));
    out[i] = peak * w * Math.sin(phase);
  }
  return out;
}

// A whole train rendered as one buffer (chirps at exact multiples
// of the 100 ms period) — port of native makeTrain.
function makeTrainSamples(sampleRate, count) {
  const chirp = makeChirpSamples(sampleRate, 1);
  const periodSamples = Math.round((CHIRP_SPEC.period_ms / 1000) * sampleRate);
  const total = (count - 1) * periodSamples + chirp.length;
  const samples = new Float32Array(total);
  const onsets_ms = [];
  for (let k = 0; k < count; k++) {
    samples.set(chirp, k * periodSamples);
    onsets_ms.push(round3((k * periodSamples / sampleRate) * 1000));
  }
  return { samples, onsets_ms, duration_ms: (total / sampleRate) * 1000 };
}

function baselineTimelineMs(sampleRate) {
  const lock = makeTrainSamples(sampleRate, CHIRP_SPEC.lock_chirps);
  const bg = makeTrainSamples(sampleRate, CHIRP_SPEC.background_chirps);
  const bgStart = lock.duration_ms + CHIRP_SPEC.silence_ms;
  return {
    lock: { start_ms: 0, duration_ms: lock.duration_ms },
    silence: { start_ms: lock.duration_ms, duration_ms: CHIRP_SPEC.silence_ms },
    background: { start_ms: bgStart, duration_ms: bg.duration_ms },
    total_ms: bgStart + bg.duration_ms
  };
}

// Port of native nextPoseChirpSlot: at least `leadS` from now and
// at least one chirp period after the previous pose chirp.
function nextPoseChirpSlot(nowCtxS, lastStartCtxS, leadS, minSpacingS) {
  leadS = leadS == null ? 0.03 : leadS;
  minSpacingS = minSpacingS == null ? CHIRP_SPEC.period_ms / 1000 : minSpacingS;
  const earliest = nowCtxS + leadS;
  if (lastStartCtxS == null) return earliest;
  return Math.max(earliest, lastStartCtxS + minSpacingS);
}

// Mono 16-bit PCM WAV (the native bundle's audio.wav format).
function encodeWav16Blob(samples, sampleRate) {
  const n = samples.length;
  const buf = new ArrayBuffer(44 + n * 2);
  const v = new DataView(buf);
  const wstr = (off, s) => { for (let i = 0; i < s.length; i++) v.setUint8(off + i, s.charCodeAt(i)); };
  wstr(0, "RIFF"); v.setUint32(4, 36 + n * 2, true); wstr(8, "WAVE");
  wstr(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true);
  v.setUint16(22, 1, true); v.setUint32(24, sampleRate, true);
  v.setUint32(28, sampleRate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  wstr(36, "data"); v.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
  return new Blob([buf], { type: "audio/wav" });
}

// Read the sample rate back out of a finished WAV's own header —
// the native v1.1 honesty discipline: the manifest's "actual" is
// measured from the file, never echoed from the request.
async function wavRateFromBlob(blob) {
  try {
    const head = new Uint8Array(await blob.slice(0, 44).arrayBuffer());
    if (head.length < 44) return null;
    const tag = String.fromCharCode(head[0], head[1], head[2], head[3]);
    if (tag !== "RIFF") return null;
    return new DataView(head.buffer, head.byteOffset).getUint32(24, true) || null;
  } catch (e) { return null; }
}

// The evidence ledger — a port of native buildEvidenceLedger
// (manifest.ts) over a web session record. Estimator-side groups
// stay null + flag, exactly as native: never dressed as values.
// v3.2: the optional third argument carries the v1.3 initiation
// records (scale lock, intake ledger, region provenance); absent
// (a pre-v3.2 flow) the groups ride as null + 'no_v13_initiation',
// the same flag native writes when a bundle has no initiation.
function buildEvidenceLedgerWeb(session, frames, v13) {
  const s = session || null;
  const poseChirps = (s && s.pose_chirps) || [];
  const noSessNum = () => ({ value: null, flag: s ? "not_computed_at_capture" : "no_audio_session" });
  let dpq;
  if (!s) dpq = { value: null, flag: "no_audio_session" };
  else if (s.baseline_failure) {
    dpq = { value: { outcome: "failed", reason: s.baseline_failure }, flag: "computed_on_device" };
  } else {
    const cal = s.session_calibration;
    const degraded =
      cal.session_flags.includes("route-not-built-in") ||
      cal.session_flags.includes("browser-audio-processing-left-on") ||
      cal.session_flags.includes("browser-audio-processing-state-unreported") ||
      s.trains.some(t => t.status === "failed") ||
      poseChirps.some(c => c.status === "failed");
    dpq = {
      value: {
        outcome: degraded ? "degraded"
          : cal.latency_lock === "not_evaluated_on_device" ? "not_evaluated_on_device" : "not_run",
        reason: degraded ? "session completed with recorded degradations (see session_flags / failed emissions)" : null
      },
      flag: "computed_on_device"
    };
  }
  const byFrame = new Map(frames.map(f => [f.frame, f]));
  const timing = s ? {
    value: {
      pose_chirps: poseChirps.map(c => {
        const f = byFrame.get(c.frame_index);
        const frameOff = f ? f.audio_offset_ms : null;
        return {
          frame: c.frame_index,
          chirp_audio_offset_ms: c.audio_offset_ms,
          frame_audio_offset_ms: frameOff,
          delta_ms: c.audio_offset_ms != null && frameOff != null
            ? round3(c.audio_offset_ms - frameOff) : null
        };
      }),
      trains: s.trains.map(t => ({ kind: t.kind, pass: t.pass, chirp_offsets_ms: t.chirp_offsets_ms }))
    },
    flag: "recorded"
  } : { value: null, flag: "no_audio_session" };
  const per = p => ({
    frames: frames.filter(f => f.pass === p).length,
    emitted: poseChirps.filter(c => c.pass === p && c.status === "emitted").length,
    failed: poseChirps.filter(c => c.pass === p && c.status === "failed").length
  });
  const p1 = per(1), p2 = per(2);
  const covOf = q => (q.frames > 0 ? q.emitted / q.frames : null);
  const c1 = covOf(p1), c2 = covOf(p2);
  const heightPair = s ? {
    value: {
      pass_1: { frames: p1.frames, chirps_emitted: p1.emitted, chirps_failed: p1.failed },
      pass_2: { frames: p2.frames, chirps_emitted: p2.emitted, chirps_failed: p2.failed },
      coverage_ratio: c1 != null && c2 != null && Math.max(c1, c2) > 0
        ? Math.round((Math.min(c1, c2) / Math.max(c1, c2)) * 10000) / 10000 : null
    },
    flag: "computed_on_device"
  } : { value: null, flag: "no_audio_session" };
  const temp = s ? {
    value: {
      temp_c: s.session_calibration.temp_c,
      source: s.session_calibration.temp_source,
      c_assumed_m_per_s: s.session_calibration.c_assumed_m_per_s
    },
    flag: s.session_calibration.temp_source
  } : { value: null, flag: "no_audio_session" };
  let decision, reason;
  if (s && s.session_calibration.session_flags.includes("latency-lock-failed")) {
    decision = { value: "abstained", flag: "computed_on_device" };
    reason = { value: "latency-lock-failed", flag: "overlay-contract-taxonomy" };
  } else if (!s || poseChirps.length === 0) {
    const hadTrains = !!s && s.trains.some(t => t.kind === "measurement" && t.status === "emitted");
    if (s && hadTrains) {
      decision = { value: "recorded", flag: "computed_on_device" };
      reason = { value: null, flag: "recorded" };
    } else {
      decision = { value: "abstained", flag: "computed_on_device" };
      reason = { value: "no-chirp-in-capture", flag: "overlay-contract-taxonomy" };
    }
  } else {
    decision = { value: "recorded", flag: "computed_on_device" };
    reason = { value: null, flag: "recorded" };
  }
  return {
    direct_path_quality: dpq,
    timing_evidence: timing,
    absolute_echo_support: noSessNum(),
    relative_separation: noSessNum(),
    spectral_consistency: noSessNum(),
    baseline_change: noSessNum(),
    height_pair_consistency: heightPair,
    temperature_state: temp,
    final_decision: decision,
    diagnostic_reason: reason,
    scale_lock: v13 && v13.scaleLock
      ? { value: v13.scaleLock, flag: "recorded" }
      : { value: null, flag: "no_v13_initiation" },
    intake_ledger: v13 && v13.intakeEntries
      ? { value: v13.intakeEntries, flag: "recorded" }
      : { value: null, flag: "no_v13_initiation" },
    region_provenance: v13 && v13.regionProvenance
      ? { value: v13.regionProvenance, flag: "recorded" }
      : { value: null, flag: "no_v13_initiation" }
  };
}

// The v3 manifest: the v2 runner contract (code / paper / n_frames /
// capture / frames_meta) PLUS the native-shaped extension blocks,
// PLUS pipeline:"browser-web" on the document itself.
// v3.5: the ground-truth block (native makeGroundTruth shape) —
// the label stays the pre-scan pairing label; a measured
// dimension the user enters AFTER the walk (the web Ground
// Truth act) rides as the witness, provenance 'user-measured';
// null remains the honest default when they didn't measure.
// Present only when the scan carries a label or a measure.
function buildGroundTruthWeb(info) {
  const name = (info.label || "").trim();
  const measured = info.measuredDimension || null;
  if (!name && !measured) return null;
  if (!measured) {
    // Label only — the pre-v3.5 shape, unchanged: the label was
    // typed before the scan, for pairing, and is not a measure.
    return {
      object_name: name || null,
      measured_dimension: null,
      provenance: {
        source: "user-entered",
        verified: false,
        note: "Typed into the web app before the scan as the object label (for pairing A/B bundles). User-supplied — not verified by the app."
      }
    };
  }
  return {
    object_name: name || null,
    measured_dimension: { kind: measured.kind, value_mm: measured.valueMm },
    provenance: {
      source: "user-measured",
      verified: false,
      note: "Typed in by the user after capture. User-supplied ground truth — not verified by the app."
    }
  };
}

function buildManifestV3(info) {
  const method = SCAN_METHODS_WEB[info.methodId] || SCAN_METHODS_WEB["phone"];
  return {
    code: info.code,
    paper: "LETTER",
    n_frames: info.frames.length,
    made_with: "Scan the World test app v3.5 (web capture — chirp-enabled browser pipeline)",
    pipeline: "browser-web",
    app: { name: "Scan the World test app (web)", version: "3.5", build: 8, platform: "browser-web" },
    method: { id: method.id, name: method.name, points: method.points },
    device: info.device,
    // v1.3 initiation blocks (native manifest placement: they
    // ride only when the flow produced them — every v3.2+ scan
    // does; v3.4's record carries aim-dot targeting).
    ...(info.initiation ? { initiation: info.initiation } : {}),
    ...(info.coverage ? { coverage: info.coverage } : {}),
    ...(info.secondPasses ? { second_passes: info.secondPasses } : {}),
    capture: {
      version: 4,
      style: "aim-dot initiation + declared scale lock, continuous orbit two passes (level, then higher), optional flip / scan-bottom second pass",
      gate: {
        definition: "port of scanner-core gates.py check_frame — native-resolution grayscale Laplacian variance + exposure/clipping checks, always computed on the native-resolution canvas; since v3.1 the bundled frame is that canvas downscaled to the frame_max_px long edge (JPEG 0.85) when the stream is larger — gate scores are computed pre-downscale, and frames_meta records the bundled dimensions",
        sharpness_min: GATE_LIMITS.SHARPNESS_MIN,
        exposure_mean_min: GATE_LIMITS.EXPOSURE_MEAN_MIN,
        exposure_mean_max: GATE_LIMITS.EXPOSURE_MEAN_MAX,
        highlight_clip_max: GATE_LIMITS.HIGHLIGHT_CLIP_MAX,
        shadow_clip_max: GATE_LIMITS.SHADOW_CLIP_MAX
      },
      stream: info.stream,
      locks: info.locks,
      passes: info.passes,
      attempts: info.attempts,
      gate_dropped: info.gateDropped,
      attempts_pacing_skipped: info.pacingSkipped || 0,
      selection_dropped: info.selectionDropped,
      // v3.3 pre-send pruning (native v1.4.0 capture block shape):
      // present ONLY when the user actually left frames out at
      // the review — an unpruned scan's manifest keeps the exact
      // v3.2 shape (harness-pinned).
      ...(info.userPruned ? {
        user_pruned: info.userPruned.count,
        user_pruned_frames: info.userPruned.frames,
        user_pruned_pose_chirps: info.userPruned.poseChirpsDropped
      } : {}),
      target_frames: info.targetFrames,
      cap_frames: info.capFrames,
      frame_max_px: info.frameMaxPx != null ? info.frameMaxPx : 1600,
      started_at_epoch_ms: info.startedAtEpochMs,
      finished_at_epoch_ms: info.finishedAtEpochMs
    },
    audio: info.audio,
    audio_session: info.audioSession,
    evidence_ledger: info.ledger,
    sync: {
      clock: "audio-context clock (the recorder's sample timeline) for chirp and frame offsets; device epoch ms recorded alongside",
      audio_started_at_epoch_ms: info.audio ? info.audio.started_at_epoch_ms : null,
      note: "Chirp offsets are computed on the recording AudioContext's clock from the recorder's start frame, and each frame's offset uses the same clock at grab time. The browser's input path adds one constant delay to the recorded track; the measuring side works that constant out from the opening sound-check tones themselves, so the pairing timestamps are a starting hint — the sound file itself is the truth."
    },
    // v3.5 (sound-off truth, native parity — Alan's 2026-11-10
    // ruling): chirps off means NO recording exists, and the
    // manifest says so in the native words, 'off-by-design', so
    // the runner grades the sound side ABSTAINED by design
    // exactly as it does a native bundle. 'silent' is retired
    // as a manifest value (the setup toggle's own word is now
    // chirps on/off — see the home screen).
    sound_mode: info.soundMode === "silent" ? "off-by-design" : "chirp",
    label: info.label || null,
    ground_truth: buildGroundTruthWeb(info),
    research_corpus_eligible: method.calibrated,
    research_note: info.soundMode === "silent"
      ? method.calibrated
        ? "Web-capture scan (pipeline: browser-web) on a calibration method, but chirps were off: no sound was recorded at all, so there is no sound to pair with the calibrated reference as research data."
        : "Web-capture scan (pipeline: browser-web), no calibration method. Chirps were off: no sound was recorded at all."
      : method.calibrated
        ? "Web-capture scan (pipeline: browser-web) on a calibration method. Its sound recording is paired with that calibrated reference as research data (acoustic estimator unproven — data first)."
        : "Web-capture scan (pipeline: browser-web), no calibration method. Its audio is still included.",
    frames_meta: info.frames
  };
}
// ==== END AUDIO PURE BLOCK ====

const errMsg = e => (e && e.message) ? e.message : String(e || "unknown error");
const sleepMs = ms => new Promise(r => setTimeout(r, ms));

// The web chirp session — mirrors native ChirpSession's behaviour
// and record shape (audioSession.ts), with browser-honest unknowns
// recorded as nulls + flags instead of native library values.
class WebChirpSession {
  constructor() {
    this.ctx = null; this.fs = null;
    this.broken = false; this.baselineFailure = null;
    this.flags = ["browser-web-pipeline"];
    this.trains = []; this.poseChirps = [];
    this.lastPoseStartCtxS = null;
    this.silenceWindow = { audio_offset_ms: null, duration_ms: CHIRP_SPEC.silence_ms };
    this.chirpBuffer = null;
    this.recKind = null; this.recFailure = null;
    this.recStartFrame = null; this.recStartEpochMs = null;
    this.chunks = []; this.totalSamples = 0; this.recNodes = [];
    this.recording = false; this.truncated = false;
    this.grantedAudio = undefined; // undefined = not read yet; null = browser wouldn't say
    this.ua = (navigator && navigator.userAgent) || "";
    // "chirp" (full protocol) or "silent" (sound off: no
    // emission and NO recording at all — v3.5). Set by startScan.
    this.soundMode = "chirp";
  }
  recCtxTime() {
    return (this.recStartFrame == null || !this.fs) ? null : this.recStartFrame / this.fs;
  }
  offsetMs(ctxTime) {
    const t0 = this.recCtxTime();
    return t0 == null ? null : round3((ctxTime - t0) * 1000);
  }
  async prepare(stream) {
    // v3.5 (sound-off truth — Alan's 2026-11-10 ruling, native
    // parity): chirps OFF means the microphone is never read and
    // no recording exists. The session prepares playback only —
    // the recorder never starts, no ambient track is captured,
    // and the record states the absence as by design.
    const offByDesign = this.soundMode === "silent";
    const track = !offByDesign && stream && stream.getAudioTracks
      ? stream.getAudioTracks()[0] : null;
    if (offByDesign) {
      this.flags.push("sound-off-by-design-no-recording");
      this.grantedAudio = null; // never read — there is no track
    }
    if (track) {
      try { this.grantedAudio = track.getSettings ? (track.getSettings() || null) : null; }
      catch (e) { this.grantedAudio = null; }
      const g = this.grantedAudio || {};
      const keys = ["echoCancellation", "noiseSuppression", "autoGainControl"];
      if (keys.some(k => g[k] === true)) this.flags.push("browser-audio-processing-left-on");
      else if (!keys.every(k => g[k] === false)) this.flags.push("browser-audio-processing-state-unreported");
    } else {
      this.flags.push("microphone-not-in-stream");
    }
    try {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) throw new Error("no Web Audio in this browser");
      this.ctx = new AC();
      this.fs = this.ctx.sampleRate;
    } catch (e) {
      this.broken = true;
      this.baselineFailure = "the sound checks could not start (the browser would not start sound playback: " + errMsg(e) + ")";
      this.flags.push("chirp-playback-unavailable");
      return;
    }
    try { await this.ctx.resume(); } catch (e) { /* state checked next */ }
    if (this.ctx.state !== "running") {
      this.broken = true;
      this.baselineFailure = "the sound checks could not start (the browser's sound system stayed asleep — check the volume and silent switch, then try again)";
      this.flags.push("audio-context-suspended");
      return;
    }
    const chirp = makeChirpSamples(this.fs, 1);
    this.chirpBuffer = this.ctx.createBuffer(1, chirp.length, this.fs);
    this.chirpBuffer.copyToChannel(chirp, 0);
    if (track) await this.startRecorder(stream);
  }
  async startRecorder(stream) {
    const src = this.ctx.createMediaStreamSource(stream);
    this.recStartEpochMs = Date.now();
    const cap = this.fs * 300; // ~5 minutes of samples — far past any scan
    const push = (frame, samples) => {
      if (!this.recording) return;
      if (this.recStartFrame == null) this.recStartFrame = frame;
      if (this.totalSamples >= cap) {
        if (!this.truncated) { this.truncated = true; this.flags.push("audio-truncated-at-5-minutes"); }
        return;
      }
      this.chunks.push(samples);
      this.totalSamples += samples.length;
    };
    try {
      if (!this.ctx.audioWorklet || !this.ctx.audioWorklet.addModule) throw new Error("no AudioWorklet");
      await this.ctx.audioWorklet.addModule("pcm-worklet.js");
      const node = new AudioWorkletNode(this.ctx, "stw-pcm-capture");
      node.port.onmessage = e => {
        const d = e.data;
        if (d && d.samples) push(d.frame, d.samples);
      };
      const mute = this.ctx.createGain(); mute.gain.value = 0;
      src.connect(node); node.connect(mute); mute.connect(this.ctx.destination);
      this.recNodes = [src, node, mute]; this.recKind = "worklet";
    } catch (e1) {
      try {
        const node = this.ctx.createScriptProcessor(4096, 1, 1);
        let count = 0, firstEst = null;
        node.onaudioprocess = ev => {
          const data = ev.inputBuffer.getChannelData(0);
          count += data.length;
          if (firstEst == null) firstEst = Math.max(0, Math.round(this.ctx.currentTime * this.fs) - count);
          push(firstEst + (count - data.length), data.slice(0));
        };
        const mute = this.ctx.createGain(); mute.gain.value = 0;
        src.connect(node); node.connect(mute); mute.connect(this.ctx.destination);
        this.recNodes = [src, node, mute]; this.recKind = "script-processor";
        this.flags.push("audio-anchor-estimated");
      } catch (e2) {
        this.recKind = null;
        this.recFailure = "the browser would not record raw sound (" + errMsg(e2) + ")";
        this.flags.push("audio-recording-unavailable");
      }
    }
    this.recording = !!this.recKind;
  }
  stopRecording() {
    this.recording = false;
    this.recNodes.forEach(n => { try { n.disconnect(); } catch (e) { /* already gone */ } });
    this.recNodes = [];
    if (!this.recKind || this.totalSamples === 0) return null;
    const all = new Float32Array(this.totalSamples);
    let o = 0;
    for (const c of this.chunks) { all.set(c, o); o += c.length; }
    this.chunks = [];
    return { samples: all, sampleRate: this.fs };
  }
  emitTrain(kind, pass, count) {
    const fail = reason => {
      const rec = { kind, pass, chirp_count: count, status: "failed", failure_reason: reason,
        started_at_epoch_ms: null, audio_offset_ms: null, chirp_offsets_ms: [] };
      this.trains.push(rec); return rec;
    };
    if (this.broken || !this.ctx || !this.fs) return fail(this.baselineFailure || "chirp playback is not available");
    if (!this.recording) return fail(this.recFailure || "the microphone recording is not running, so a chirp train would not be captured");
    try {
      const train = makeTrainSamples(this.fs, count);
      const buffer = this.ctx.createBuffer(1, train.samples.length, this.fs);
      buffer.copyToChannel(train.samples, 0);
      const source = this.ctx.createBufferSource(); source.buffer = buffer;
      const gain = this.ctx.createGain();
      gain.gain.value = Math.pow(10, CHIRP_SPEC.level_dbfs_peak / 20); // -20 dBFS peak, never clips
      source.connect(gain); gain.connect(this.ctx.destination);
      const when = this.ctx.currentTime + 0.05;
      source.start(when);
      const audioOffset = this.offsetMs(when);
      const rec = {
        kind, pass, chirp_count: count, status: "emitted", failure_reason: null,
        started_at_epoch_ms: (this.recStartEpochMs != null && audioOffset != null)
          ? this.recStartEpochMs + Math.round(audioOffset) : null,
        audio_offset_ms: audioOffset,
        chirp_offsets_ms: train.onsets_ms.map(x => audioOffset == null ? x : round3(audioOffset + x))
      };
      this.trains.push(rec); return rec;
    } catch (e) {
      return fail("chirp playback failed (" + errMsg(e) + ")");
    }
  }
  // Baseline phase, protocol order, one held pose over the empty
  // table: lock x5 -> 0.5 s silence -> background x8 (~2 s total).
  async runBaseline() {
    if (this.soundMode === "silent") return null; // sound off: no emission at all
    if (this.broken || !this.fs) {
      this.flags.push("baseline-not-run");
      return this.baselineFailure || "the sound checks could not run";
    }
    const wall0 = Date.now();
    const tl = baselineTimelineMs(this.fs);
    const lock = this.emitTrain("latency-lock", null, CHIRP_SPEC.lock_chirps);
    if (lock.status === "failed") {
      this.baselineFailure = "the sound checks didn't work (" + lock.failure_reason + "). The photos still count — this scan's sound just won't carry size evidence.";
      this.flags.push("latency-lock-failed");
      return this.baselineFailure;
    }
    if (lock.audio_offset_ms != null) {
      this.silenceWindow = {
        audio_offset_ms: round3(lock.audio_offset_ms + tl.lock.duration_ms),
        duration_ms: CHIRP_SPEC.silence_ms
      };
    }
    const waitBg = tl.background.start_ms - (Date.now() - wall0);
    if (waitBg > 0) await sleepMs(waitBg);
    const bg = this.emitTrain("background", null, CHIRP_SPEC.background_chirps);
    if (bg.status === "failed") {
      this.baselineFailure = "the sound checks only partly worked (" + bg.failure_reason + "). The photos still count.";
      this.flags.push("background-train-failed");
      return this.baselineFailure;
    }
    const waitEnd = tl.total_ms - (Date.now() - wall0);
    if (waitEnd > 0) await sleepMs(waitEnd);
    return null;
  }
  emitPoseChirp(frameIndex, pass, frameCapturedAtEpochMs) {
    const fail = reason => {
      const rec = { chirp_index: this.poseChirps.length, frame_index: frameIndex, pass,
        frame_captured_at_epoch_ms: frameCapturedAtEpochMs, started_at_epoch_ms: null,
        audio_offset_ms: null, status: "failed", failure_reason: reason };
      this.poseChirps.push(rec);
      if (!this.flags.includes("pose-chirp-failed")) this.flags.push("pose-chirp-failed");
      return rec;
    };
    if (this.broken || !this.ctx || !this.fs || !this.chirpBuffer) {
      return fail(this.baselineFailure || "chirp playback is not available");
    }
    if (!this.recording) return fail(this.recFailure || "the microphone recording is not running");
    try {
      const source = this.ctx.createBufferSource(); source.buffer = this.chirpBuffer;
      const gain = this.ctx.createGain();
      gain.gain.value = Math.pow(10, CHIRP_SPEC.level_dbfs_peak / 20);
      source.connect(gain); gain.connect(this.ctx.destination);
      const when = nextPoseChirpSlot(this.ctx.currentTime, this.lastPoseStartCtxS);
      source.start(when);
      this.lastPoseStartCtxS = when;
      const audioOffset = this.offsetMs(when);
      const rec = {
        chirp_index: this.poseChirps.length, frame_index: frameIndex, pass,
        frame_captured_at_epoch_ms: frameCapturedAtEpochMs,
        started_at_epoch_ms: (this.recStartEpochMs != null && audioOffset != null)
          ? this.recStartEpochMs + Math.round(audioOffset) : null,
        audio_offset_ms: audioOffset, status: "emitted", failure_reason: null
      };
      this.poseChirps.push(rec); return rec;
    } catch (e) {
      return fail("chirp playback failed (" + errMsg(e) + ")");
    }
  }
  remapPoseChirps(oldToNew) {
    let dropped = 0;
    const out = [];
    for (const c of this.poseChirps) {
      const n = oldToNew.get(c.frame_index);
      if (n == null) { dropped++; continue; }
      out.push(Object.assign({}, c, { frame_index: n }));
    }
    this.poseChirps = out;
    return dropped;
  }
  record(selectionDroppedChirps) {
    const flags = this.flags.slice();
    const silent = this.soundMode === "silent";
    if (silent && !flags.includes("sound-mode-silent")) flags.push("sound-mode-silent");
    if (!flags.includes("latency-lock-not-evaluated-on-device")) flags.push("latency-lock-not-evaluated-on-device");
    const lat = (this.ctx && (this.ctx.outputLatency || this.ctx.baseLatency)) || null;
    return {
      // Sound-off truth (v3.5, native parity — Alan's ruling):
      // chirps off means NO recording was made at all. The
      // record's mode is the native word, 'off-by-design', and
      // the empty trains / pose_chirps lists mean "nothing was
      // emitted and nothing was recorded, by the user's choice"
      // — never a measured baseline, never an ambient track.
      mode: silent ? "off-by-design" : "chirp",
      baseline_note: silent
        ? "Sound was OFF for this scan (chosen at setup): no baseline trains and no pose chirps were emitted, and no sound was recorded at all — the microphone was never read. The empty trains and pose_chirps lists are an absence by design, not a measurement."
        : null,
      chirp_spec: {
        waveform: CHIRP_SPEC.waveform, f0_hz: CHIRP_SPEC.f0_hz, f1_hz: CHIRP_SPEC.f1_hz,
        duration_ms: CHIRP_SPEC.duration_ms, window: CHIRP_SPEC.window, period_ms: CHIRP_SPEC.period_ms,
        context_sample_rate_hz: this.fs, emission_level_dbfs_peak: CHIRP_SPEC.level_dbfs_peak
      },
      trains: this.trains,
      pose_chirps: this.poseChirps,
      aperture: {
        mode: silent ? "silent" : "per-pose",
        pose_chirps_emitted: this.poseChirps.filter(c => c.status === "emitted").length,
        pose_chirps_failed: this.poseChirps.filter(c => c.status === "failed").length,
        selection_dropped_chirps: selectionDroppedChirps || 0
      },
      silence_window: silent ? { audio_offset_ms: null, duration_ms: 0 } : this.silenceWindow,
      session_calibration: {
        device_profile_id: null,
        sample_rate_hz_reported: this.fs,
        audio_route: null,
        d_sm_mm: null,
        beta_hat_ms: null,
        beta_spread_samples: null,
        beta_n_chirps: (this.trains.find(t => t.kind === "latency-lock" && t.status === "emitted") || {}).chirp_count ?? null,
        latency_lock: "not_evaluated_on_device",
        noise_floor_median_residual: null,
        ambient_peak_to_floor: null,
        floor_multiplier: CHIRP_SPEC.floor_multiplier,
        alpha: null,
        alpha_source: "not_available",
        scale_check: "not_available",
        temp_c: CHIRP_SPEC.default_temp_c,
        temp_source: "default-20C",
        c_assumed_m_per_s: round3(soundSpeedMps(CHIRP_SPEC.default_temp_c) * 100) / 100,
        session_flags: flags
      },
      runtime_self_report: {
        captured_at: new Date().toISOString(),
        identity: {
          model_string: this.ua, model_identifier: this.ua,
          os: "browser", os_version: this.ua, profile_id_matched: null
        },
        route: {
          inputs: [], outputs: [],
          note: "The browser does not tell the page the audio route — recorded as unknown, never guessed. If a Bluetooth speaker or earbuds were connected, the route was not the phone's own speaker/mic and nothing here can prove otherwise."
        },
        rates: { requested_hz: AUDIO_REQUESTED_RATE_HZ, device_preferred_hz: null, context_hz: this.fs },
        latency_estimates: {
          library_input_latency_s: lat,
          note: "Browser-reported audio latency estimate only — a prior for the server-side latency lock, never the latency itself."
        }
      },
      baseline_failure: this.baselineFailure,
      browser_audio: {
        constraints_requested: AUDIO_CONSTRAINTS_ASKED,
        track_settings_granted: this.grantedAudio === undefined ? null : this.grantedAudio,
        recorder: this.recKind,
        note: "echoCancellation / noiseSuppression / autoGainControl were requested OFF. track_settings_granted is the browser's own readback of what it actually applied; where it is null or missing a key, the browser would not say — recorded as unreported, never assumed off."
      }
    };
  }
  async close() {
    try { if (this.ctx) await this.ctx.close(); } catch (e) { /* already closed */ }
    this.ctx = null;
  }
}

function browserDeviceBlock() {
  const ua = (navigator && navigator.userAgent) || "";
  let model = "Browser device";
  if (/iPhone/.test(ua)) model = "iPhone (browser)";
  else if (/iPad/.test(ua)) model = "iPad (browser)";
  else if (/Android/.test(ua)) model = "Android phone (browser)";
  return {
    model, model_identifier: ua, os: "browser", os_version: ua,
    device_profile_id: null, matched_profile_id: null, profile_status: "to-be-measured"
  };
}

// Scan method chosen on the home screen (persisted; default the
// printed mat, matching the how-to instructions). v3.5 (method
// gating, native parity — Alan's 2026-11-10 ruling): the web
// method set IS the native MethodSelect set, same ids, names
// and order (lib/methods.ts). The method composes the flow:
// phone has NO coin stage at all (its manifest stamps the
// initiation scale_stage 'not-part-of-this-method'); the coin
// stage exists only inside the coin method; print-sheet and
// cal-block keep the tap-scale stage they share with native.
// 'coin' here is the tap-ritual coin method — the native
// auto-finder stays unported (COORDINATOR-REPORT-WEB-3.3.md
// names the four structural reasons; W-CAP-1 stays a known,
// stated divergence).
const SCAN_METHODS_WEB = {
  "phone": { id: "phone", name: "Phone camera", points: 5, calibrated: false },
  "print-sheet": { id: "print-sheet", name: "Print sheet", points: null, calibrated: true },
  "coin": { id: "coin", name: "Coin", points: 10, calibrated: true },
  "cal-block": { id: "cal-block", name: "Calibration block", points: 50, calibrated: true }
};
let scanMethodId = "print-sheet";
try {
  const saved = localStorage.getItem("stw-method-v3");
  if (saved && SCAN_METHODS_WEB[saved]) scanMethodId = saved;
} catch (e) { /* private mode etc. — default stands */ }
function renderMethodSeg() {
  document.querySelectorAll("#method-seg .seg-btn").forEach(b => {
    b.classList.toggle("active", b.dataset.method === scanMethodId);
    b.onclick = () => {
      scanMethodId = b.dataset.method;
      try { localStorage.setItem("stw-method-v3", scanMethodId); } catch (e) { /* ignore */ }
      renderMethodSeg();
    };
  });
}

// Sound on/off (Alan's setup toggle): "chirp" = the full
// protocol (baseline trains + per-pose chirps, sound recorded);
// "silent" (the stored value's name is kept so an existing
// preference still reads) = NO chirps and NO recording at all —
// the microphone is never read (Alan's 2026-11-10 ruling, the
// native sound_mode 'off-by-design'). The frame pipeline is
// identical either way. Persisted, default On.
let scanSoundMode = "chirp";
try {
  const savedSound = localStorage.getItem("stw-sound-v3");
  if (savedSound === "chirp" || savedSound === "silent") scanSoundMode = savedSound;
} catch (e) { /* default stands */ }
function renderSoundSeg() {
  document.querySelectorAll("#sound-seg .seg-btn").forEach(b => {
    b.classList.toggle("active", b.dataset.sound === scanSoundMode);
    b.onclick = () => {
      scanSoundMode = b.dataset.sound;
      try { localStorage.setItem("stw-sound-v3", scanSoundMode); } catch (e) { /* ignore */ }
      renderSoundSeg();
    };
  });
}

// The scan label (optional; pre-filled with the previous scan's
// label so pairing an A/B set is one tap). Read at Start, carried
// into the manifest verbatim.
function currentLabel() {
  const el = $("label-input");
  return el ? el.value.trim() : "";
}
function initLabelField() {
  const el = $("label-input");
  if (!el) return;
  try { el.value = localStorage.getItem("stw-label-v3") || ""; } catch (e) { /* ignore */ }
}
function persistLabel() {
  try { localStorage.setItem("stw-label-v3", currentLabel()); } catch (e) { /* ignore */ }
}


const PASS_TARGET = 12, PASS_CAP = 16;      // kept photos per pass
const TARGET_TOTAL = 24, CAP_TOTAL = 32;    // the smart frame budget
const FRAME_MAX_PX = 1600;                  // v3.1 bundle diet: bundled frames are capped at this long edge
const FRAME_JPEG_QUALITY_CAPPED = 0.85;     // JPEG quality used for a frame downscaled to the cap
const DONE_MIN = 20;                        // Done unlocks here
const ATTEMPT_MS = 130;                     // pause between gate attempts
const KEEP_GAP_MS = 300;                    // min spacing between keeps
const PASS_WORDS = [
  "Walk slowly around it — stay level with the object.",
  "Second pass — a little higher this time, looking down a bit."
];
// v3.2 (native v1.3 parity): the optional second pass (Flip /
// Scan Bottom, theory §13) gets its own small frame budget, and
// the scale-lock loop takes at most LOCK_MAX_ATTEMPTS extra stills
// after the scale tap's own still (native LOCK_MAX_ATTEMPTS = 8).
const SECOND_PASS_TARGET = 8, SECOND_PASS_CAP = 12;
const LOCK_MAX_ATTEMPTS = 8;
const COACHING = {
  "blurry": "Slow down — hold steady for a breath.",
  "highlights clipped": "Too much glare — shift a little so light isn't bouncing straight back.",
  "too bright": "Very bright here — a touch of shade would help.",
  "too dark": "Too dark — more light will help a lot.",
  "shadows clipped": "Deep shadows — more even light will help."
};
let stream = null, capturing = false, attemptBusy = false,
    kept = [], frameMeta = [], currentCode = null, captureMeta = null,
    passIndex = 0, keptThisPass = 0, attempts = 0, gateDropped = 0,
    passStats = [], recentResults = [], lastKeepAt = 0, lastScore = null,
    transitionUntil = 0, turnDeg = 0, lastMotionTs = null;
// v3 audio state
let audioSession = null, audioUnavailableReason = null,
    scanPhase = "starting", scanCancelled = false,
    bundleAudioBlob = null, captureStartedAt = 0;
// v3.2 initiation state (reset per scan in startScan). v3.4:
// the initiation is aim-dot (native v1.3.6) — aimLanded marks
// that a Set target press has landed (the Start scan door
// appears), aimNote carries the aim phase's transient words
// (working / failed), pendingInitiation is a failed still
// chain's recorded aim-point initiation that Start scan
// promotes, and aimChainSeq retires a superseded still chain.
// undersideAnnounced latches the flip-visibility announcement.
let scanGen = 0, loopGen = 0,
    initiationData = null, scaleMachine = null, denomId = "quarter-us",
    scalePointNorm = null, initiationBlob = null, scaleBlob = null,
    coverageTracker = null, pacingSkipped = 0,
    tapBusy = false, lockAttempts = 0,
    aimLanded = false, aimNote = null, pendingInitiation = null,
    aimChainSeq = 0, undersideAnnounced = false,
    secondPassKind = null, choiceShown = false, finalizeStarted = false;
// v3.3 pre-send pruning state: everything finalizeCapture computed,
// held while the user reviews the frames; the manifest + bundle
// are built only when the review is confirmed.
let packCtx = null, reviewExcluded = new Set(), reviewUrls = [];
const gateCanvas = document.createElement("canvas");
const gctx = gateCanvas.getContext("2d", { willReadFrequently: true });

// IMU metadata only: rotation rate + gravity direction per grab, and
// a rough cumulative turn estimate (a diversity proxy for the
// manifest — it never gates, never implies size). Feature-detected:
// absent sensors change nothing.
let latestMotion = null;
window.addEventListener("devicemotion", e => {
  const r = e.rotationRate || {};
  const g = e.accelerationIncludingGravity || {};
  latestMotion = {
    rot_rate: Math.round(Math.hypot(r.alpha || 0, r.beta || 0, r.gamma || 0) * 10) / 10,
    gravity: [g.x, g.y, g.z].map(v => v == null ? null : Math.round(v * 100) / 100)
  };
  if (capturing && e.timeStamp != null) {
    if (lastMotionTs != null) {
      const dt = Math.min(0.25, Math.max(0, (e.timeStamp - lastMotionTs) / 1000));
      turnDeg += Math.hypot(r.alpha || 0, r.beta || 0) * dt;
    }
    lastMotionTs = e.timeStamp;
  }
});

function qualityState() {
  if (!recentResults.length) return { word: "Getting ready…", color: "var(--muted)" };
  const passes = recentResults.filter(Boolean).length;
  if (passes >= Math.ceil(recentResults.length / 2))
    return { word: "Good — keep going slowly", color: "var(--good)" };
  return { word: "Slow down — hold steady", color: "var(--warn)" };
}

function dominantFailure() {
  const counts = {};
  recentFailures.forEach(f => f.forEach(name => { counts[name] = (counts[name] || 0) + 1; }));
  let best = null, bestN = 0;
  for (const k in counts) if (counts[k] > bestN) { best = k; bestN = counts[k]; }
  return best;
}
let recentFailures = [];

function audioLineText() {
  if (!audioSession) return "Sound: starting…";
  if (audioSession.baselineFailure) return "Sound: the checks didn't work — photos still count";
  if (!audioSession.recording) return "Sound: unavailable — " + (audioUnavailableReason || "photos only this scan");
  if (scanSoundMode === "silent") return "Sound: off for this scan — chirps off at setup; nothing is being recorded";
  return "Sound: recording — a soft chirp for every photo kept";
}

function setPrompt() {
  const now = performance.now();
  let text;
  if (scanPhase === "baseline") {
    text = "Sound checks — hold the phone over the EMPTY table where the object will sit, and keep still. Volume up, silent switch off.";
  } else if (scanPhase === "baseline-failed") {
    text = (audioSession && audioSession.baselineFailure) || "The sound checks didn't work — the photos still count.";
  } else if (scanPhase === "baseline-done") {
    text = "Sound checks done — now put the object down and line the dot up on it.";
  } else if (scanPhase === "initiation") {
    // v3.4: aim-dot targeting — the dot is the target; a
    // transient aim note (working / that-didn't-take) outranks
    // the standing instruction while it stands.
    text = aimNote || "Line the dot up on the object, then press Set target.";
  } else if (scanPhase === "scale") {
    text = "Tap anything whose size you know — a coin works. Or skip: shape only is fine.";
  } else if (scanPhase === "locking") {
    text = scaleMachine ? scaleMachine.userWord() : "Keep adjusting — the size hasn't locked yet.";
  } else if (scanPhase === "choice") {
    text = "That's the object from up here. Want the bottom too?";
  } else if (scanPhase === "walk2") {
    text = secondPassKind === "flip"
      ? "Flipped over — now walk around this side the same slow way."
      : "On its side — walk around it slowly. This pass becomes the bottom.";
  } else if (now < transitionUntil) {
    text = passIndex === 1
      ? "That's the first pass. Now the second — a little higher."
      : "That's plenty — all done.";
  } else {
    const fails = recentResults.length >= 3 && recentResults.filter(Boolean).length < 2;
    const df = fails ? dominantFailure() : null;
    text = df && COACHING[df] ? COACHING[df] : PASS_WORDS[passIndex];
    text += " Keep the object filling at least half the screen.";
  }
  $("prompt").textContent = text;
  $("pass-label").textContent =
    (scanPhase === "baseline" || scanPhase === "baseline-failed" || scanPhase === "baseline-done")
      ? "Sound checks"
      : scanPhase === "initiation" ? "Getting started"
      : (scanPhase === "scale" || scanPhase === "locking") ? "Size — optional"
      : scanPhase === "choice" ? "Almost done"
      : scanPhase === "walk2" ? "Second pass"
      : "Pass " + (passIndex + 1) + " of 2";
  const aline = $("audio-line");
  if (aline) aline.textContent = audioLineText();
  const total = kept.length;
  $("kept-count").textContent = total + (total === 1 ? " photo" : " photos");
  // v3.5 (W-WALK-1/2): the count-target fill bar and the
  // sharpness meter are GONE — the coverage segments (banked
  // counts, below) are the walk's only progress display, and
  // sharpness is a word here, never a gauge.
  const q = qualityState();
  $("quality-dot").style.color = q.color;
  $("quality-word").textContent = q.word;
  const val = $("sharp-value");
  if (lastScore == null) { val.textContent = "Sharpness: —"; }
  else {
    const ok = lastScore >= GATE_LIMITS.SHARPNESS_MIN;
    val.textContent = "Sharpness: " + Math.round(lastScore) + (ok ? " — good" : " — too soft, not kept");
    val.style.color = ok ? "var(--good)" : "var(--warn)";
  }
  const doneBtn = $("btn-done");
  if (scanPhase === "choice" || scanPhase === "initiation") {
    // The second-pass card carries the choice phase's actions;
    // the aim controls carry the initiation phase's (v3.4).
    doneBtn.classList.add("hidden");
  } else if (scanPhase === "scale" || scanPhase === "locking") {
    doneBtn.classList.remove("hidden");
    doneBtn.disabled = true;
    doneBtn.textContent = "Done";
  } else {
    doneBtn.classList.remove("hidden");
    doneBtn.disabled = total < DONE_MIN;
    doneBtn.textContent = total >= DONE_MIN
      ? "Done — " + total + " photos"
      : "Done (need " + DONE_MIN + " photos)";
  }
  renderScaleUi();
  renderAimUi();
  updateGuidance();
  renderCoverageSegments();
  renderProblemLine();
  renderUndersideNote();
  const aimDot = $("aim-dot");
  // v3.4: the dot is up in the aim phase too — it IS the target
  // there, not decoration (in the walk it stays the v3.3 banked-
  // frame indicator).
  if (aimDot) aimDot.classList.toggle("hidden",
    !(scanPhase === "initiation" || scanPhase === "walk" || scanPhase === "walk2"));
}

// v3.4: the aim controls (native v1.3.6's aim phase furniture):
// Set target is the phase's one action; Start scan appears once
// a press has landed — the never-strand door that promotes the
// aim point alone when the still chain failed or is still
// running. Outside the aim phase both are gone.
function renderAimUi() {
  const ui = $("aim-ui");
  if (!ui) return;
  const inAim = scanPhase === "initiation";
  ui.classList.toggle("hidden", !inAim);
  if (!inAim) return;
  $("btn-start-aim").classList.toggle("hidden", !aimLanded);
}

// v3.4: flip visibility — the moment primary coverage completes
// during the walk, the underside choice is announced in words
// on the walk screen (before that, a user mid-scan had no way
// to know the flip existed until the end-of-walk card). The
// latch is set here, off the tracker's own completeness, so
// the announcement can only ever follow real banked coverage;
// the end-of-walk card itself is unchanged.
const UNDERSIDE_ANNOUNCE_WORD =
  "Top and sides done — next you can flip it over or scan the bottom, or finish";
function renderUndersideNote() {
  const el = $("underside-note");
  if (!el) return;
  if (scanPhase === "walk" && coverageTracker && coverageTracker.isComplete())
    undersideAnnounced = true;
  const showIt = scanPhase === "walk" && undersideAnnounced;
  el.textContent = showIt ? UNDERSIDE_ANNOUNCE_WORD : "";
  el.classList.toggle("hidden", !showIt);
}

// v3.3: the coverage segment display (Qlone's clearable form,
// bound to the real ledger — see coverageDisplayWeb). Shown
// while walking; the choice screen keeps it visible so the
// completeness the card talks about is on screen.
function renderCoverageSegments() {
  const el = $("coverage-segments");
  if (!el) return;
  const showIt = coverageTracker &&
    (scanPhase === "walk" || scanPhase === "walk2" || scanPhase === "choice");
  if (!showIt) { el.classList.add("hidden"); el.innerHTML = ""; return; }
  const d = coverageDisplayWeb(coverageTracker.snapshot());
  el.innerHTML = "";
  d.parts.forEach(p => {
    const row = document.createElement("div");
    row.className = "cov-row";
    const lab = document.createElement("span");
    lab.className = "cov-label"; lab.textContent = p.label;
    const segs = document.createElement("span");
    segs.className = "cov-segs";
    p.segments.forEach(s => {
      const i = document.createElement("i");
      i.className = "cov-seg " + s;
      segs.appendChild(i);
    });
    const word = document.createElement("span");
    word.className = "cov-word"; word.textContent = p.word;
    row.appendChild(lab); row.appendChild(segs); row.appendChild(word);
    el.appendChild(row);
  });
  el.classList.remove("hidden");
}

// v3.3: the named-problem line (Apple's vocabulary over our
// gate's words). A problem is NAMED only when it recurs in the
// recent-failure window (native dominantProblem, min 2) — one
// stray soft frame never raises a state. Name + the gate's own
// coaching + what the app does while it lasts.
function renderProblemLine() {
  const el = $("problem-line");
  if (!el) return;
  const walking = scanPhase === "walk" || scanPhase === "walk2";
  const prob = walking ? dominantProblemWeb(recentFailures) : null;
  if (!prob) { el.classList.add("hidden"); el.textContent = ""; return; }
  const advice = COACHING[prob.id] || "";
  el.textContent = prob.name + (advice ? " — " + advice : "") + " " + prob.behaviour;
  el.classList.remove("hidden");
}

function encodeCanvas(canvas) {
  return new Promise(res => canvas.toBlob(res, "image/jpeg", 0.92));
}

const bundleCanvas = document.createElement("canvas");
// The bundled photo for a kept frame (v3.1 bundle diet): the gated
// canvas itself when it already fits the cap, otherwise a downscaled
// copy — aspect preserved, never upscaled — at JPEG 0.85. Gating
// always runs on the native-resolution canvas; only the shipped copy
// is capped. Returns the blob and the BUNDLED frame's dimensions,
// which are what frames_meta records.
function encodeBundleFrame(src) {
  const w = src.width, h = src.height, longEdge = Math.max(w, h);
  if (longEdge <= FRAME_MAX_PX) {
    return encodeCanvas(src).then(blob => ({ blob, width: w, height: h }));
  }
  const s = FRAME_MAX_PX / longEdge;
  const bw = Math.round(w * s), bh = Math.round(h * s);
  bundleCanvas.width = bw; bundleCanvas.height = bh;
  bundleCanvas.getContext("2d").drawImage(src, 0, 0, bw, bh);
  return new Promise(res => bundleCanvas.toBlob(res, "image/jpeg", FRAME_JPEG_QUALITY_CAPPED))
    .then(blob => ({ blob, width: bw, height: bh }));
}

// ---------- v3.3 banked-frame feedback effects ----------
// The effect sink for planBankedFeedbackWeb's decisions. Every
// call is guarded — a failed buzz or tick can never fail a
// capture (native feedbackEffects discipline). PLATFORM TRUTH:
// iOS Safari exposes no Vibration API, so on Alan's iPhone the
// haptic half is a no-op BY PLATFORM, not by omission — the gold
// pulse and (chirps off) the tick still land. navigator.vibrate
// fires where it exists (Android Chrome).
let tickCtx = null;
function playUiTick(milestone) {
  try {
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return;
    // Prefer the session's own context (already running, already
    // the user's gesture chain); fall back to a lazily-made one.
    let ctx = (audioSession && audioSession.ctx && audioSession.ctx.state === "running")
      ? audioSession.ctx : null;
    if (!ctx) {
      if (!tickCtx) tickCtx = new AC();
      ctx = tickCtx;
      if (ctx.state === "suspended") { void ctx.resume().catch(() => {}); }
      if (ctx.state !== "running") return;
    }
    const t0 = ctx.currentTime + 0.01;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = "sine";
    osc.frequency.value = milestone ? 1320 : 1760;
    gain.gain.setValueAtTime(0.0001, t0);
    gain.gain.exponentialRampToValueAtTime(milestone ? 0.12 : 0.07, t0 + 0.012);
    gain.gain.exponentialRampToValueAtTime(0.0001, t0 + (milestone ? 0.14 : 0.08));
    osc.connect(gain); gain.connect(ctx.destination);
    osc.start(t0); osc.stop(t0 + (milestone ? 0.16 : 0.1));
  } catch (e) { /* feedback must never fail a capture */ }
}
function fireBankedFeedback(kind) {
  const plan = planBankedFeedbackWeb(kind, scanSoundMode === "chirp");
  if (plan.pulse) {
    const dot = $("aim-dot");
    if (dot) {
      dot.classList.remove("pulse");
      void dot.offsetWidth; // restart the animation on rapid banks
      dot.classList.add("pulse");
    }
  }
  if (plan.haptic !== "none" && navigator.vibrate) {
    try { navigator.vibrate(plan.haptic === "milestone" ? [25, 40, 25] : 15); } catch (e) { /* guarded */ }
  }
  if (plan.tone === "ui-tick") playUiTick(kind !== "banked");
  // plan.tone === 'pose-chirp': the frame's own chirp already
  // sounded for this exact moment — no second sound, ever.
}

// One gate attempt: draw the live frame once, score THOSE pixels
// with the server's gate, and — only if they pass — encode that same
// canvas as the bundled photo.
async function captureAttempt() {
  if (!capturing || attemptBusy) return;
  const video = $("cam");
  if (!video.videoWidth) return;
  attemptBusy = true;
  try {
    if (gateCanvas.width !== video.videoWidth) {
      gateCanvas.width = video.videoWidth; gateCanvas.height = video.videoHeight;
    }
    gctx.drawImage(video, 0, 0);
    // v3: the frame's timestamps, taken at the grab — epoch for the
    // record, audio-context clock for the pairing with its chirp.
    const grabEpoch = Date.now();
    const grabCtx = (audioSession && audioSession.ctx && audioSession.recording)
      ? audioSession.ctx.currentTime : null;
    const img = gctx.getImageData(0, 0, gateCanvas.width, gateCanvas.height);
    const res = gateFromGray(grayFromRGBA(img.data), gateCanvas.width, gateCanvas.height);
    attempts++;
    lastScore = res.sharpness;
    recentResults.push(res.passed); if (recentResults.length > 5) recentResults.shift();
    if (!res.passed) {
      gateDropped++;
      // v3.2: the intake ledger accounts for gate failures too —
      // primary walk only; the second pass keeps its own
      // accounting in second_passes/passStats. v3.3: the trace
      // carries the failure NAMES (native traceGateFailure) —
      // nothing is dropped silently, every refusal has its
      // reason on the record.
      if (coverageTracker && passIndex < 2) coverageTracker.traceGateFailure(res.failures);
      recentFailures.push(res.failures); if (recentFailures.length > 5) recentFailures.shift();
      setPrompt();
      return;
    }
    recentFailures = [];
    const now = performance.now();
    if (now - lastKeepAt < KEEP_GAP_MS) { pacingSkipped++; setPrompt(); return; }
    if (kept.length >= (passIndex === 2 ? CAP_TOTAL + SECOND_PASS_CAP : CAP_TOTAL)) { setPrompt(); return; }
    const enc = await encodeBundleFrame(gateCanvas);
    const blob = enc.blob;
    if (!blob || !capturing) return;
    lastKeepAt = now;
    kept.push(blob);
    frameMeta.push({
      frame: kept.length - 1, pass: passIndex + 1,
      file: null,
      sharpness: Math.round(res.sharpness * 10) / 10,
      exposure_mean: Math.round(res.exposureMean * 10) / 10,
      highlight_frac: Math.round(res.highlightFrac * 10000) / 10000,
      shadow_frac: Math.round(res.shadowFrac * 10000) / 10000,
      width: enc.width, height: enc.height,
      captured_at_epoch_ms: grabEpoch,
      audio_offset_ms: grabCtx != null ? audioSession.offsetMs(grabCtx) : null,
      pose_chirp_audio_offset_ms: null,
      turn_deg_est: Math.round(turnDeg),
      motion: latestMotion
    });
    // v3: ONE chirp for this kept frame's pose (the aperture look).
    // Fire-and-forget on the audio clock; a failure is recorded in
    // the session and the photo is kept regardless (native parity).
    // Sound off: never called — no emission, no recording.
    // v3.2: the second pass (passIndex 2) emits NO pose chirps —
    // like native's second pass it is a separate act whose frames
    // carry no chirp pairing in the primary session record; the
    // frames' audio offsets stay real (one recording, chirps on).
    if (audioSession && scanSoundMode === "chirp" && passIndex < 2) {
      const rec = audioSession.emitPoseChirp(kept.length - 1, passIndex + 1, grabEpoch);
      const fm = frameMeta[frameMeta.length - 1];
      fm.pose_chirp_audio_offset_ms = rec.status === "emitted" ? rec.audio_offset_ms : null;
    }
    keptThisPass++;
    // v3.2: bank the view into the coverage model (guided sectors).
    // v3.3: a banked view is CONFIRMED — gold pulse on the aim
    // dot, a haptic where the platform has one, and the tone the
    // plan rules (the pose chirp itself when chirps are on; a
    // soft tick when they're off). Milestones (a part completing,
    // coverage completing) get the stronger form.
    if (coverageTracker && passIndex < 2) {
      const entry = coverageTracker.offer(passIndex);
      let kind = "banked";
      if (entry && entry.outcome === "banked" && entry.part) {
        if (coverageTracker.isComplete()) kind = "coverage-complete";
        else if (coverageTracker.banked[entry.part] >= VIEWS_PER_PART) kind = "part-complete";
      }
      fireBankedFeedback(kind);
    }
    setPrompt();
    const pTarget = passIndex === 2 ? SECOND_PASS_TARGET : PASS_TARGET;
    const pCap = passIndex === 2 ? SECOND_PASS_CAP : PASS_CAP;
    if (keptThisPass >= pTarget || keptThisPass >= pCap) {
      if (passIndex === 0 && kept.length < CAP_TOTAL) {
        passStats.push({ pass: 1, kept: keptThisPass });
        passIndex = 1; keptThisPass = 0;
        transitionUntil = performance.now() + 2200;
        setPrompt();
      } else {
        void finishCapture();
      }
    } else if (kept.length >= (passIndex === 2 ? CAP_TOTAL + SECOND_PASS_CAP : CAP_TOTAL)) void finishCapture();
  } finally { attemptBusy = false; }
}

async function captureLoop(gen) {
  while (capturing && gen === loopGen) {
    await captureAttempt();
    await new Promise(r => setTimeout(r, ATTEMPT_MS));
  }
}
// The loop restarts per walk (main walk, then the second pass);
// the generation stamp retires any loop still sleeping from an
// earlier phase so two loops never run at once.
function startCaptureLoop() { loopGen++; captureLoop(loopGen); }

// Attempt focus / exposure / white-balance locks through track
// constraints, then read the settings back: the manifest records
// what truly engaged, "unsupported" where the browser offers nothing.
async function tryCameraLocks(track) {
  const locks = { focus: "unsupported", exposure: "unsupported", white_balance: "unsupported" };
  let settings = {};
  try { settings = (track.getSettings && track.getSettings()) || {}; } catch (e) { /* keep {} */ }
  try {
    const caps = (track.getCapabilities && track.getCapabilities()) || {};
    const adv = [], asked = {};
    const pick = (list) => list.includes("continuous") ? "continuous" : list[0];
    if (Array.isArray(caps.focusMode) && caps.focusMode.length) {
      asked.focus = pick(caps.focusMode); adv.push({ focusMode: asked.focus });
    }
    if (Array.isArray(caps.exposureMode) && caps.exposureMode.length) {
      asked.exposure = pick(caps.exposureMode); adv.push({ exposureMode: asked.exposure });
    }
    if (Array.isArray(caps.whiteBalanceMode) && caps.whiteBalanceMode.length) {
      asked.white_balance = pick(caps.whiteBalanceMode); adv.push({ whiteBalanceMode: asked.white_balance });
    }
    if (adv.length) {
      await track.applyConstraints({ advanced: adv });
      const after = (track.getSettings && track.getSettings()) || {};
      if (after.focusMode) locks.focus = "engaged (" + after.focusMode + ")";
      else if (asked.focus) locks.focus = "requested (" + asked.focus + "), not confirmed by the browser";
      if (after.exposureMode) locks.exposure = "engaged (" + after.exposureMode + ")";
      else if (asked.exposure) locks.exposure = "requested (" + asked.exposure + "), not confirmed by the browser";
      if (after.whiteBalanceMode) locks.white_balance = "engaged (" + after.whiteBalanceMode + ")";
      else if (asked.white_balance) locks.white_balance = "requested (" + asked.white_balance + "), not confirmed by the browser";
      settings = after;
    }
  } catch (e) {
    for (const k in locks) if (locks[k] === "unsupported") locks[k] = "request failed";
  }
  const streamInfo = {
    width: settings.width || null, height: settings.height || null,
    frame_rate: settings.frameRate || null, facing: settings.facingMode || null
  };
  return { locks, streamInfo };
}

let streamInfo = null, lockInfo = null;
async function startScan() {
  kept = []; frameMeta = []; passStats = []; recentResults = []; recentFailures = [];
  passIndex = 0; keptThisPass = 0; attempts = 0; gateDropped = 0;
  lastScore = null; lastKeepAt = 0; turnDeg = 0; lastMotionTs = null;
  captureMeta = null; currentCode = newCode();
  audioSession = null; audioUnavailableReason = null; bundleAudioBlob = null;
  scanPhase = "starting"; scanCancelled = false; captureStartedAt = Date.now();
  // v3.2 per-scan reset: initiation, scale lock, coverage,
  // second pass — and the capture screen's furniture. v3.4:
  // the aim state resets with it, and any still chain from a
  // previous scan is retired by the chain stamp.
  scanGen++;
  initiationData = null; scaleMachine = null; scalePointNorm = null;
  initiationBlob = null; scaleBlob = null;
  coverageTracker = new CoverageTracker();
  pacingSkipped = 0; tapBusy = false; lockAttempts = 0;
  aimLanded = false; aimNote = null; pendingInitiation = null;
  aimChainSeq++; undersideAnnounced = false;
  secondPassKind = null; choiceShown = false; finalizeStarted = false;
  packCtx = null; reviewExcluded = new Set(); reviewUrls = [];
  denomId = "quarter-us";
  document.querySelectorAll("#denom-seg .seg-btn").forEach(x =>
    x.classList.toggle("active", x.dataset.denom === denomId));
  $("second-pass-ui").classList.add("hidden");
  $("tap-dot").classList.add("hidden");
  show("capture"); setPrompt();
  // iOS motion permission must come from a user gesture; ask, never block on it.
  try {
    if (window.DeviceMotionEvent && typeof DeviceMotionEvent.requestPermission === "function")
      await DeviceMotionEvent.requestPermission().catch(() => {});
  } catch (e) { /* metadata is optional */ }
  // Camera + microphone in ONE request, so the audio route the
  // session locks is the route the whole capture uses (protocol
  // §1). Mic constraints ask for the browser's processing OFF —
  // what was actually granted is read back and recorded.
  // v3.5 (sound-off truth): when chirps are OFF the microphone
  // is never requested at all — the stream asks for video only,
  // so a sound-off scan cannot record even ambient room sound.
  const videoReq = { facingMode: "environment", width: { ideal: 2560 } };
  if (scanSoundMode === "silent") {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: videoReq, audio: false });
    } catch (e) {
      stream = null;
    }
  } else {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: videoReq, audio: AUDIO_CONSTRAINTS_ASKED });
    } catch (e) {
      audioUnavailableReason = "the microphone was not allowed — photos only this scan";
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: videoReq, audio: false });
      } catch (e2) {
        stream = null;
      }
    }
  }
  if (!stream) {
    $("prompt").textContent = "The camera didn't start. Allow camera access and try again.";
    return;
  }
  if (scanSoundMode !== "silent" && !stream.getAudioTracks().length && !audioUnavailableReason) {
    audioUnavailableReason = "the browser gave no microphone — photos only this scan";
  }
  try {
    // Ask for the most the camera offers (2560 wide ideal — phones
    // grant what they can; iOS Safari commonly grants 1920x1080).
    // Whatever arrives is what's gated, bundled and recorded.
    $("cam").srcObject = stream;
    const track = stream.getVideoTracks()[0];
    if (track) {
      const got = await tryCameraLocks(track);
      lockInfo = got.locks; streamInfo = got.streamInfo;
      if (!streamInfo.width && $("cam").videoWidth) {
        streamInfo.width = $("cam").videoWidth; streamInfo.height = $("cam").videoHeight;
      }
    }
    // v3: the chirp session — playback context up, recorder
    // running, THEN the baseline phase, all before the first photo.
    persistLabel();
    audioSession = new WebChirpSession();
    audioSession.soundMode = scanSoundMode;
    if (scanSoundMode !== "silent") await audioSession.prepare(stream);
    if (scanCancelled) return;
    if (scanSoundMode !== "silent" && !audioSession.recording && !audioUnavailableReason) {
      audioUnavailableReason = audioSession.recFailure || "sound recording did not start — photos only this scan";
    }
    if (scanSoundMode === "silent") {
      // Sound OFF (Alan's ruling): no baseline, no chirps, and
      // NO recording — the session was never prepared, the
      // microphone was never requested. The initiation comes
      // first, then the walk (v3.4: aim-dot).
      enterInitiation();
      return;
    }
    scanPhase = "baseline";
    setPrompt();
    const bf = await audioSession.runBaseline();
    if (scanCancelled) return;
    if (bf) {
      scanPhase = "baseline-failed";
      setPrompt();
      await sleepMs(2400);
      if (scanCancelled) return;
    } else {
      scanPhase = "baseline-done";
      setPrompt();
      await sleepMs(1600);
      if (scanCancelled) return;
    }
    enterInitiation();
  } catch (e) {
    $("prompt").textContent = "The camera didn't start. Allow camera access and try again.";
  }
}
function stopCapture() { capturing = false; loopGen++; }
function endCamera() {
  scanCancelled = true;
  stopCapture();
  if (audioSession) { const s = audioSession; audioSession = null; s.stopRecording(); s.close(); }
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  $("cam").srcObject = null;
}

// ---------- initiation, scale lock, second pass ----------
// The flow (theory §§2-5, 13): baseline (or silent) → set the
// target → tap a known-size thing (or skip) → the lock loop
// reads stills until three agree or the attempts run out →
// the two-pass walk, steered by the coverage guidance → the
// second-pass choice (Flip / Scan Bottom / finish) → pack.
// v3.2 shipped the initiation as native v1.3's tap; v3.4 ports
// native v1.3.6's AIM-DOT targeting (Alan's 2026-11-10
// redesign): there is no tap layer on the camera in the aim
// phase — the centre aim dot is the target, and the labelled
// Set target button is the press. The scale stage keeps its
// tap (the coin finder stays unported — COORDINATOR-REPORT-
// WEB-3.3.md gives the four structural reasons).

function recordPassStat() {
  const p = passIndex + 1;
  if (passStats.some(s => s.pass === p)) return;
  passStats.push(p === 3
    ? { pass: 3, kept: keptThisPass, kind: secondPassKind }
    : { pass: p, kept: keptThisPass });
}

function updateGuidance() {
  const el = $("guidance-line");
  if (!el) return;
  if (coverageTracker && scanPhase === "walk") {
    el.textContent = coverageTracker.guidance();
    el.classList.remove("hidden");
  } else {
    el.classList.add("hidden");
  }
}

function renderScaleUi() {
  const ui = $("scale-ui");
  if (!ui) return;
  const inScale = scanPhase === "scale", inLock = scanPhase === "locking";
  ui.classList.toggle("hidden", !(inScale || inLock));
  $("denom-seg").classList.toggle("hidden", !inScale);
  $("scale-ui-label").classList.toggle("hidden", !inScale);
  $("btn-skip-scale").classList.toggle("hidden", !inScale);
  $("lock-word").classList.toggle("hidden", !inLock);
  $("lock-progress").classList.toggle("hidden", !inLock);
  $("btn-skip-locking").classList.toggle("hidden", !inLock);
  if (inLock && scaleMachine) {
    $("lock-word").textContent = scaleMachine.userWord();
    $("lock-progress").textContent =
      "Reads so far: " + scaleMachine.readCount +
      " · steady streak: " + scaleMachine.streakCount + " of " + LOCK_READS_REQUIRED;
  }
  // v3.4: the tap layer exists ONLY for the scale tap now —
  // the aim phase has no tap layer at all (the dot is the
  // target and Set target is a button, not a touch on glass).
  $("tap-layer").classList.toggle("hidden", !inScale);
}

function enterInitiation() {
  if (scanCancelled) return;
  scanPhase = "initiation";
  setPrompt();
}

// Map a display point on the preview to normalised photo
// coordinates. The preview element uses object-fit: cover, so
// the displayed frame is a centre crop of the stream — the
// mapping undoes exactly that crop (native's viewToPhoto,
// cover-fit form).
function displayToPhoto(clientX, clientY) {
  const video = $("cam");
  if (!video.videoWidth) return null;
  const rect = video.getBoundingClientRect();
  const vw = video.videoWidth, vh = video.videoHeight;
  const scale = Math.max(rect.width / vw, rect.height / vh);
  const ox = (rect.width - vw * scale) / 2, oy = (rect.height - vh * scale) / 2;
  const cx = clientX - rect.left, cy = clientY - rect.top;
  const clamp01 = v => Math.max(0, Math.min(1, v));
  return {
    nx: clamp01((cx - ox) / scale / vw), ny: clamp01((cy - oy) / scale / vh),
    dx: cx / rect.width, dy: cy / rect.height
  };
}

function videoTapPoint(ev) { return displayToPhoto(ev.clientX, ev.clientY); }

// v3.4: the aim seed — the centre aim dot's position mapped
// into the photo through the SAME cover-fit mapping (native
// v1.3.6's viewToPhoto at the aim point (0.5, 0.5)). The crop
// is centred, so the seed is the photo's centre by
// construction; it is computed, not assumed, so the mapping
// stays the single source of the conversion.
function aimSeedPoint() {
  const video = $("cam");
  if (!video.videoWidth) return null;
  const rect = video.getBoundingClientRect();
  const p = displayToPhoto(rect.left + rect.width / 2, rect.top + rect.height / 2);
  return p ? { x: p.nx, y: p.ny } : null;
}

function showTapDot(p, isScale) {
  const dot = $("tap-dot");
  dot.style.left = (p.dx * 100) + "%";
  dot.style.top = (p.dy * 100) + "%";
  dot.style.borderColor = isScale ? "#7fd08c" : "#ffffff";
  dot.classList.remove("hidden");
}

// A fresh canvas holding the current preview frame at stream
// resolution (fresh per still: the bundled blob encodes
// asynchronously and must never race the next grab).
function grabStillCanvas() {
  const video = $("cam");
  if (!video.videoWidth) return null;
  const c = document.createElement("canvas");
  c.width = video.videoWidth; c.height = video.videoHeight;
  c.getContext("2d").drawImage(video, 0, 0);
  return c;
}

// Analyse a still: grayscale analysis frame ≤512 px (native
// analyseStill), returned with the full still's dimensions.
function analyseStill(canvas) {
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const img = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const small = downscaleGray(img.data, canvas.width, canvas.height, 512);
  return { small, fullW: canvas.width, fullH: canvas.height };
}

// One scale read from a still at the stored scale point (native
// scaleReadFrom): grow the region, take its ellipse axes in
// full-still pixels, feed the machine. True when a read was fed.
function scaleReadFrom(canvas) {
  if (!scalePointNorm || !scaleMachine) return false;
  const st = analyseStill(canvas);
  const f = st.small.factor;
  const region = growRegion(st.small.gray, st.small.w, st.small.h,
    scalePointNorm.x * st.fullW / f, scalePointNorm.y * st.fullH / f);
  const axes = region.areaPx > 20 ? ellipseAxes(region.mask, st.small.w, st.small.h) : null;
  if (!axes) { scaleMachine.feed(null); return false; }
  scaleMachine.feed({ majorPx: axes.majorPx * f, minorPx: axes.minorPx * f, atEpochMs: Date.now() });
  return true;
}

// v3.5 method gating (native parity — Alan's 2026-11-10
// ruling): the method composes the flow. The phone method's
// flow contains NO coin stage at all; the coin stage exists
// only inside the coin method (print-sheet and cal-block keep
// the tap-scale stage they share with native's tap flows).
function methodHasScaleStage() {
  return scanMethodId === "coin" || scanMethodId === "print-sheet" ||
    scanMethodId === "cal-block";
}
function scaleStageWord() {
  return methodHasScaleStage() ? "scale-tap" : "not-part-of-this-method";
}

// The initiation stands (origin set, the still recorded
// either way) — route to the flow's next stage, composed from
// the method: methods with a scale stage go to the tap; the
// phone method goes straight to the walk — no coin stage
// exists in its flow to land on.
function advanceAfterInitiation() {
  if (!methodHasScaleStage()) {
    scanPhase = "walk";
    capturing = true;
    setPrompt();
    startCaptureLoop();
    return;
  }
  scaleMachine = new ScaleLockMachine(denominationById(denomId));
  scanPhase = "scale";
  setPrompt();
}

// v3.4 — Set target (native v1.3.6's onAimTarget, ported): the
// user holds the centre aiming dot on the object and presses
// the labelled button — a single press; aim, press, done. The
// initiation still is taken at the press and the origin seed
// is the dot's position, mapped into the still through the
// cover-fit mapping (aimSeedPoint). The press is acknowledged
// FIRST, before any async work. The never-strand machinery is
// native's: the still chain is BOUNDED (the v1.3.2 lesson — a
// stage that hangs must fail with a named reason, not silence),
// and a failed chain records a pending initiation on the aim
// point alone that Start scan promotes. "Aim again" remains,
// but it is never the only door.
const AIM_STILL_TIMEOUT_MS = 10000;
async function onSetTarget() {
  if (scanPhase !== "initiation") return;
  if (tapBusy) {
    // Even a swallowed press answers — never silence (native).
    aimNote = "Still working on the last try — one moment…";
    setPrompt();
    return;
  }
  const seed = aimSeedPoint();
  if (!seed) {
    aimLanded = true;
    pendingInitiation = {
      origin: { x: 0.5, y: 0.5 }, stillFile: null,
      boundaryNote: "The initiation still could not be taken — the camera frame wasn't ready. " +
        "Recorded as failed; the scan is scoped by the aim point alone."
    };
    aimNote = "That didn't take — the camera frame wasn't ready. Aim again, or press Start scan to go with just the aim point.";
    setPrompt();
    return;
  }
  tapBusy = true;
  aimLanded = true;
  aimChainSeq++;
  const chain = aimChainSeq, gen = scanGen;
  // Acknowledge FIRST: the ring pulses on the dot and the
  // status line says what's happening.
  const dot = $("aim-dot");
  if (dot) { dot.classList.remove("pulse"); void dot.offsetWidth; dot.classList.add("pulse"); }
  aimNote = "Got it — taking the still…";
  setPrompt();
  const stale = () => scanCancelled || gen !== scanGen || chain !== aimChainSeq || scanPhase !== "initiation";
  try {
    const work = (async () => {
      const canvas = grabStillCanvas();
      if (!canvas) throw new Error("the camera frame wasn't ready");
      const enc = await encodeBundleFrame(canvas);
      if (!enc.blob) throw new Error("the still could not be turned into a photo");
      return { canvas, blob: enc.blob };
    })();
    const timeout = sleepMs(AIM_STILL_TIMEOUT_MS)
      .then(() => { throw new Error("the still took too long"); });
    const got = await Promise.race([work, timeout]);
    if (stale()) return; // Start scan superseded this chain
    initiationBlob = got.blob;
    const st = analyseStill(got.canvas);
    const region = growRegion(st.small.gray, st.small.w, st.small.h,
      seed.x * st.small.w, seed.y * st.small.h);
    const frac = region.areaPx / (st.small.w * st.small.h);
    const boundary = region.bbox && frac > 0.002 && frac < 0.7
      ? { area_px: region.areaPx, bbox: region.bbox } : null;
    initiationData = {
      origin: { x: seed.x, y: seed.y },
      boundary,
      boundaryNote: boundary
        ? "Boundary grown from the aim point on the initiation still (seeded region grow, analysis frame ≤512 px). The initiation still is the evidence."
        : "The region grow from the aim point did not settle on a plausible boundary on the initiation still — recorded as failed; the scan is scoped by the aim point alone.",
      scaleTap: null,
      scaleStage: scaleStageWord(),
      stillFile: "initiation.jpg",
      sceneLiveUntil: "no-scale-tap"
    };
    pendingInitiation = null;
    aimNote = null;
    advanceAfterInitiation();
  } catch (e) {
    if (stale()) return;
    // The aim DID land: the origin is the aim point, whatever
    // the still chain did. A photo captured before the failure
    // (initiationBlob set) is preserved as bundle evidence and
    // the record says so; with no photo, still_file is null.
    const reason = e && e.message ? e.message : "the still failed";
    const keptStill = initiationBlob != null;
    pendingInitiation = {
      origin: { x: seed.x, y: seed.y },
      stillFile: keptStill ? "initiation.jpg" : null,
      boundaryNote: "The initiation still could not be processed — " + reason + ". " +
        (keptStill
          ? "The still itself is kept in the bundle as evidence; no boundary was grown from it. "
          : "") +
        "Recorded as failed; the scan is scoped by the aim point alone."
    };
    aimNote = "That didn't take — " + reason +
      ". Aim again, or press Start scan to go with just the aim point.";
    setPrompt();
  } finally { tapBusy = false; }
}

// Start scan from the aim phase (native v1.3.3's never-strand
// door, carried into the aim flow). Promotes the aim point to
// the initiation — carrying the recorded still failure when
// the chain already failed, or an honest started-early record
// when it is still in flight (the in-flight chain stands down
// via the chain stamp) — and routes to the scale stage like
// any completed initiation.
function startScanFromAim() {
  if (scanPhase !== "initiation") return;
  aimChainSeq++; // an in-flight chain, if any, stands down
  tapBusy = false;
  const pending = pendingInitiation;
  const seed = aimSeedPoint() || { x: 0.5, y: 0.5 };
  initiationData = {
    origin: pending ? pending.origin : { x: seed.x, y: seed.y },
    boundary: null,
    boundaryNote: pending
      ? pending.boundaryNote
      : "You started the scan before the initiation still finished — no boundary was grown from it. The scan is scoped by the aim point alone.",
    scaleTap: null,
    scaleStage: scaleStageWord(),
    stillFile: pending ? pending.stillFile : null,
    sceneLiveUntil: "no-scale-tap"
  };
  pendingInitiation = null;
  aimNote = null;
  advanceAfterInitiation();
}

async function onScaleTap(ev) {
  if (scanPhase !== "scale" || tapBusy || !initiationData) return;
  const p = videoTapPoint(ev);
  if (!p) return;
  tapBusy = true;
  try {
    // The denomination may have been changed on the chips after
    // the machine was created — rebuild it so the DECLARED size
    // is the one the user settled on (native parity).
    scaleMachine = new ScaleLockMachine(denominationById(denomId));
    const canvas = grabStillCanvas();
    if (!canvas) return;
    const enc = await encodeBundleFrame(canvas);
    if (!enc.blob) return;
    scaleBlob = enc.blob;
    initiationData.scaleTap = { x: p.nx, y: p.ny };
    scalePointNorm = { x: p.nx, y: p.ny };
    scaleReadFrom(canvas);
    initiationData.sceneLiveUntil = "scale-read-taken";
    showTapDot(p, true);
    scanPhase = "locking";
    lockAttempts = 0;
    setPrompt();
    void lockLoop();
  } finally { tapBusy = false; }
}

function skipScale() {
  if (!initiationData || !scaleMachine) return;
  scaleMachine.giveUp("You skipped the size tap — shape only this scan.");
  initiationData.sceneLiveUntil = "scale-refused";
  startWalk();
}

// The locking loop (native parity): while the user adjusts,
// stills repeat every ~1.2 s and feed the machine until it locks
// or LOCK_MAX_ATTEMPTS extra stills run out.
async function lockLoop() {
  const gen = scanGen;
  while (!scanCancelled && gen === scanGen && scanPhase === "locking" &&
         scaleMachine && scaleMachine.currentState === "seeking" &&
         lockAttempts < LOCK_MAX_ATTEMPTS) {
    await sleepMs(1200);
    if (scanCancelled || gen !== scanGen || scanPhase !== "locking") return;
    const canvas = grabStillCanvas();
    if (canvas) scaleReadFrom(canvas); else scaleMachine.feed(null);
    lockAttempts++;
    setPrompt();
  }
  if (scanCancelled || gen !== scanGen || scanPhase !== "locking" || !scaleMachine) return;
  if (scaleMachine.currentState === "seeking") {
    scaleMachine.giveUp("The size never locked after " + (lockAttempts + 1) +
      " reads — shape only this scan. The reads are in the scan's record.");
  }
  setPrompt();
  await sleepMs(900); // let the lock word land before the walk starts
  if (scanCancelled || gen !== scanGen || scanPhase !== "locking") return;
  startWalk();
}

function skipLocking() {
  if (!scaleMachine) return;
  scaleMachine.giveUp("You stopped the size read early — shape only this scan.");
  startWalk();
}

function startWalk() {
  if (scanCancelled) return;
  scanPhase = "walk";
  capturing = true;
  setPrompt();
  startCaptureLoop();
}

// The main walk is done: stop capturing but keep the camera and
// the recording alive — the second-pass choice comes next, and a
// second pass continues the same session (one WAV, one clock).
function endMainWalk() {
  stopCapture();
  recordPassStat();
  choiceShown = true;
  scanPhase = "choice";
  const snap = coverageTracker ? coverageTracker.snapshot() : null;
  $("coverage-word").textContent = snap && snap.declared_complete
    ? "Every side is banked — the model has the whole object from up here."
    : "Some sides are still thin — the scan keeps what it has, and the record says which sides are missing.";
  $("scale-word").textContent = scaleMachine ? scaleMachine.userWord() : "";
  $("second-pass-ui").classList.remove("hidden");
  setPrompt();
}

function beginSecondPass(kind) {
  secondPassKind = kind;
  $("second-pass-ui").classList.add("hidden");
  scanPhase = "walk2";
  passIndex = 2; keptThisPass = 0; lastKeepAt = 0;
  capturing = true;
  setPrompt();
  startCaptureLoop();
}

// finishCapture is the one finish entry (Done button + natural
// pass completion): finishing the MAIN walk opens the second-pass
// choice; finishing the second pass (or choosing "No") packs.
async function finishCapture() {
  if (scanPhase === "walk" && !choiceShown) { endMainWalk(); return; }
  await finalizeCapture();
}

// Finish: thin an over-full scan to the target budget (sharpest per
// slice of the walk), remap the per-pose chirps onto the final frame
// numbering (native captureEngine behaviour), close the audio
// session into a WAV, build the v3 manifest, hand over to Send.
// v3.2: this is the PACK step — it runs only after the second-pass
// choice (or the second pass itself). Thinning is per group: the
// main two passes thin to TARGET_TOTAL as before; second-pass
// frames thin separately to SECOND_PASS_TARGET so the flip/bottom
// views are never eaten by the main budget.
async function finalizeCapture() {
  if (finalizeStarted) return;
  finalizeStarted = true;
  stopCapture();
  recordPassStat();
  if (streamInfo && !streamInfo.width && gateCanvas.width) {
    streamInfo.width = gateCanvas.width; streamInfo.height = gateCanvas.height;
  }
  const secondKeptRaw = frameMeta.filter(m => m.pass === 3).length;
  let selectionDropped = 0;
  const oldToNew = new Map();
  {
    const mainIdx = [], secIdx = [];
    frameMeta.forEach((m, i) => (m.pass === 3 ? secIdx : mainIdx).push(i));
    let keepMain = mainIdx, keepSec = secIdx;
    if (mainIdx.length > TARGET_TOTAL) {
      keepMain = selectKeptIndices(mainIdx.map(i => frameMeta[i]), TARGET_TOTAL)
        .map(j => mainIdx[j]);
    }
    if (secIdx.length > SECOND_PASS_TARGET) {
      keepSec = selectKeptIndices(secIdx.map(i => frameMeta[i]), SECOND_PASS_TARGET)
        .map(j => secIdx[j]);
    }
    const keepAll = keepMain.concat(keepSec).sort((a, b) => a - b);
    selectionDropped = kept.length - keepAll.length;
    // Thinned primary-walk views are traced in the intake ledger
    // before the coverage snapshot is taken below.
    if (coverageTracker) {
      mainIdx.filter(i => !keepMain.includes(i))
        .forEach(() => coverageTracker.traceOutside("selection-thinned"));
    }
    keepAll.forEach((old, n) => oldToNew.set(old, n));
    kept = keepAll.map(i => kept[i]);
    frameMeta = keepAll.map((old, n) => Object.assign({}, frameMeta[old], { frame: n }));
  }
  frameMeta.forEach((m, i) => { m.file = "frame_" + String(i).padStart(3, "0") + ".jpg"; });

  // v3.2 records (native v1.3 shapes): the scale lock's own
  // record, the initiation record, the coverage snapshot, the
  // second-pass list and the region provenance. The underside is
  // recorded FILLED whenever no second pass banked views — the
  // runner-side trophy closes it algorithmically, and the record
  // keeps filled-vs-measured distinct forever (theory §13).
  const secondPassesArr = secondPassKind
    ? [{ kind: secondPassKind, completed: secondKeptRaw > 0 }] : [];
  const v13 = {
    scaleLock: scaleMachine ? scaleMachine.record() : null,
    intakeEntries: coverageTracker ? coverageTracker.intakeEntries() : null,
    regionProvenance: buildRegionProvenance({
      secondPass: secondPassesArr.length ? secondPassesArr[0] : null,
      applyFill: true
    })
  };
  const initiationRecord = initiationData ? buildInitiationRecord(initiationData) : null;
  const coverageSnap = coverageTracker ? coverageTracker.snapshot() : null;

  // Audio close-out: stop the recorder, encode the session WAV,
  // read its rate back from its own header (never the request).
  let audioRecord, audioSessionRecord = null, ledger = null, droppedChirps = 0;
  if (audioSession) {
    const sess = audioSession;
    if (selectionDropped > 0) droppedChirps = sess.remapPoseChirps(oldToNew);
    else {
      // frame numbers unchanged, but keep the pairing fields exact
      sess.poseChirps.forEach(c => { const n = oldToNew.get(c.frame_index); if (n != null) c.frame_index = n; });
    }
    const byFrame = new Map(sess.poseChirps.map(c => [c.frame_index, c]));
    frameMeta.forEach(m => {
      const c = byFrame.get(m.frame);
      m.pose_chirp_audio_offset_ms = c && c.status === "emitted" ? c.audio_offset_ms : null;
    });
    const wav = sess.stopRecording();
    if (wav) {
      bundleAudioBlob = encodeWav16Blob(wav.samples, wav.sampleRate);
      const actual = await wavRateFromBlob(bundleAudioBlob);
      audioRecord = {
        file: "audio.wav",
        sample_rate_requested_hz: AUDIO_REQUESTED_RATE_HZ,
        sample_rate_actual_hz: actual,
        sample_rate_hz: actual,
        channels: 1,
        duration_s: Math.round((wav.samples.length / wav.sampleRate) * 1000) / 1000,
        started_at_epoch_ms: sess.recStartEpochMs,
        unavailable_reason: null
      };
    } else {
      audioRecord = {
        file: null,
        sample_rate_requested_hz: AUDIO_REQUESTED_RATE_HZ,
        sample_rate_actual_hz: null,
        sample_rate_hz: null,
        channels: null,
        duration_s: null,
        started_at_epoch_ms: sess.recStartEpochMs,
        unavailable_reason: sess.broken
          ? (sess.baselineFailure || "the browser would not start sound playback")
          : (sess.recFailure || audioUnavailableReason || "the recording produced no sound samples")
      };
    }
    audioSessionRecord = sess.record(droppedChirps);
    ledger = buildEvidenceLedgerWeb(audioSessionRecord, frameMeta, v13);
    await sess.close();
    if (audioSession === sess) audioSession = null;
  } else {
    audioRecord = {
      file: null, sample_rate_requested_hz: AUDIO_REQUESTED_RATE_HZ,
      sample_rate_actual_hz: null, sample_rate_hz: null, channels: null,
      duration_s: null, started_at_epoch_ms: null,
      unavailable_reason: audioUnavailableReason || "no audio session ran"
    };
    ledger = buildEvidenceLedgerWeb(null, frameMeta, v13);
  }

  // v3.3: the manifest + bundle are NOT built yet. The scan goes
  // to the pre-send review first (KIRI's pattern, native U6):
  // the user sees every photo that would ship and can leave the
  // bad ones out; applyPruneAndPack builds the manifest (with
  // the pruning recorded, or the exact v3.2 shape when nothing
  // was left out) and packs. The evidence ledger is likewise
  // built at pack time, over the FINAL frame list.
  packCtx = {
    v13,
    audioSessionRecord,
    buildArgs: {
      code: currentCode, methodId: scanMethodId, device: browserDeviceBlock(),
      soundMode: scanSoundMode, label: currentLabel(),
      stream: streamInfo, locks: lockInfo, passes: passStats,
      attempts, gateDropped, pacingSkipped, selectionDropped,
      targetFrames: TARGET_TOTAL, capFrames: CAP_TOTAL, frameMaxPx: FRAME_MAX_PX,
      startedAtEpochMs: captureStartedAt, finishedAtEpochMs: Date.now(),
      audio: audioRecord,
      initiation: initiationRecord, coverage: coverageSnap, secondPasses: secondPassesArr,
      // v3.5: the Ground Truth act's measured dimension (native
      // shape { kind, valueMm }), set by showGroundTruth when
      // the user enters one; null = they didn't measure.
      measuredDimension: null
    }
  };
  // camera tracks stop here (the session's mic track rides the same
  // stream); endCamera would also tear down the session, which is
  // already closed — stop the tracks directly.
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  $("cam").srcObject = null;
  showReview();
}

// ---------- v3.3 pre-send review + pruning (KIRI's pattern) ----------
// Every kept frame as a thumbnail with its number, pass and
// sharpness; tap to leave a frame out of the bundle. The capture
// record is never rewritten by the review — pruning is a shipment
// decision, recorded in the manifest's capture.user_pruned fields
// (native v1.4.0 shape) and nowhere when nothing was left out.
const REVIEW_PASS_WORDS = { 1: "Pass 1 · level", 2: "Pass 2 · higher", 3: "Second pass" };
function showReview() {
  reviewExcluded = new Set();
  reviewUrls.forEach(u => { try { URL.revokeObjectURL(u); } catch (e) { /* already gone */ } });
  reviewUrls = [];
  const grid = $("review-grid");
  grid.innerHTML = "";
  kept.forEach((blob, i) => {
    const m = frameMeta[i];
    const item = document.createElement("button");
    item.type = "button";
    item.className = "review-item";
    const url = URL.createObjectURL(blob);
    reviewUrls.push(url);
    const img = document.createElement("img");
    img.src = url; img.alt = "Photo " + (i + 1);
    const cap = document.createElement("span");
    cap.className = "review-cap";
    cap.textContent = "Photo " + (i + 1) + " · " + (REVIEW_PASS_WORDS[m.pass] || ("Pass " + m.pass)) +
      " · sharpness " + Math.round(m.sharpness);
    const badge = document.createElement("span");
    badge.className = "review-badge hidden";
    badge.textContent = "Left out";
    item.appendChild(img); item.appendChild(cap); item.appendChild(badge);
    item.onclick = () => {
      if (reviewExcluded.has(i)) reviewExcluded.delete(i); else reviewExcluded.add(i);
      item.classList.toggle("excluded", reviewExcluded.has(i));
      badge.classList.toggle("hidden", !reviewExcluded.has(i));
      updateReviewLine();
    };
    grid.appendChild(item);
  });
  updateReviewLine();
  show("review");
}
function updateReviewLine() {
  const total = kept.length;
  const send = total - reviewExcluded.size;
  $("review-count").textContent = send === total
    ? "All " + total + " photos will be sent."
    : send + " of " + total + " photos will be sent — " + reviewExcluded.size + " left out.";
  const warn = $("review-warn");
  if (send === 0) {
    warn.textContent = "An empty scan can't pack — put at least one photo back in.";
    warn.classList.remove("hidden");
  } else if (send < DONE_MIN) {
    warn.textContent = "Fewer than " + DONE_MIN + " photos — measuring gets harder with so few. It will still pack.";
    warn.classList.remove("hidden");
  } else {
    warn.classList.add("hidden");
    warn.textContent = "";
  }
  $("btn-pack").disabled = send === 0;
  $("btn-pack").textContent = "Next — label it";
}
// ---------- v3.5 Ground Truth act (native parity) ----------
// Native's post-walk screen, in its own words: "Did you measure
// it?" — the one user-supplied witness the accuracy program
// leans on (the MZEN08 fix). It sits between the photo check
// and the pack: the review's drops are final when this screen
// shows; confirming here records the witness into packCtx and
// runs the pack exactly as the review's button used to.
const GT_DIM_KINDS_WEB = ["length", "height", "diameter", "width"];
let gtKindWeb = "length";
function renderGtKinds() {
  document.querySelectorAll("#gt-kind-seg .seg-btn").forEach(b => {
    b.classList.toggle("active", b.dataset.kind === gtKindWeb);
    b.onclick = () => { gtKindWeb = b.dataset.kind; renderGtKinds(); };
  });
}
function showGroundTruth() {
  if (!packCtx) return;
  gtKindWeb = "length";
  renderGtKinds();
  const inp = $("gt-mm-input");
  if (inp) inp.value = "";
  const keepN = kept.length - reviewExcluded.size;
  const line = $("gt-status");
  if (line) line.textContent = keepN + (keepN === 1 ? " photo" : " photos") +
    " will be packed · " + (SCAN_METHODS_WEB[scanMethodId] || {}).name +
    (scanSoundMode === "silent" ? " · sound off — nothing recorded" : " · sound recorded");
  show("groundtruth");
}
function confirmGroundTruth() {
  if (!packCtx) return;
  const raw = ($("gt-mm-input").value || "").trim().replace(",", ".");
  const mm = parseFloat(raw);
  packCtx.buildArgs.measuredDimension =
    Number.isFinite(mm) && mm > 0
      ? { kind: gtKindWeb, valueMm: Math.round(mm * 10) / 10 }
      : null;
  applyPruneAndPack();
}
// Confirm the review: apply the pruning (frames renumber, the
// dropped frames' pose chirps un-pair and count, the act is
// recorded), then build the manifest over the final frame list
// and pack — the v3.2 path from here, unchanged.
function applyPruneAndPack() {
  if (!packCtx) return;
  let userPruned = null;
  let audioSessionRecord = packCtx.audioSessionRecord;
  if (reviewExcluded.size > 0) {
    const { keptIdx, pruned } = applyPruningWeb(frameMeta, reviewExcluded);
    const oldToNew = new Map();
    keptIdx.forEach((old, n) => oldToNew.set(old, n));
    let poseChirpsDropped = 0;
    if (audioSessionRecord && audioSessionRecord.pose_chirps) {
      const remapped = remapPoseChirpRecordsWeb(audioSessionRecord.pose_chirps, oldToNew);
      poseChirpsDropped = remapped.dropped;
      audioSessionRecord = Object.assign({}, audioSessionRecord, {
        pose_chirps: remapped.records,
        aperture: audioSessionRecord.aperture ? Object.assign({}, audioSessionRecord.aperture, {
          pose_chirps_emitted: remapped.records.filter(c => c.status === "emitted").length,
          pose_chirps_failed: remapped.records.filter(c => c.status === "failed").length,
          selection_dropped_chirps: (audioSessionRecord.aperture.selection_dropped_chirps || 0) + remapped.dropped
        }) : audioSessionRecord.aperture
      });
    }
    kept = keptIdx.map(i => kept[i]);
    frameMeta = keptIdx.map((old, n) => Object.assign({}, frameMeta[old], { frame: n }));
    userPruned = { count: pruned.length, frames: pruned, poseChirpsDropped };
  }
  const ledger = buildEvidenceLedgerWeb(audioSessionRecord, frameMeta, packCtx.v13);
  captureMeta = buildManifestV3(Object.assign({}, packCtx.buildArgs, {
    frames: frameMeta,
    audioSession: audioSessionRecord,
    ledger,
    userPruned
  }));
  packCtx = null;
  reviewUrls.forEach(u => { try { URL.revokeObjectURL(u); } catch (e) { /* already gone */ } });
  reviewUrls = [];
  buildBundle();
}
$("btn-start").onclick = startScan;
$("btn-cancel").onclick = () => { endCamera(); show("home"); };
$("btn-done").onclick = finishCapture;
// v3.2 wiring: the tap layer, the denomination chips (DECLARED,
// never detected), the scale skips and the second-pass choice.
// v3.4: the tap layer serves ONLY the scale tap (the aim phase
// has no tap layer — Set target is a button press), and the
// aim controls get their own wiring.
$("tap-layer").addEventListener("pointerdown", ev => {
  if (scanPhase === "scale") void onScaleTap(ev);
});
$("btn-set-target").onclick = () => { void onSetTarget(); };
$("btn-start-aim").onclick = startScanFromAim;
document.querySelectorAll("#denom-seg .seg-btn").forEach(b => {
  b.onclick = () => {
    denomId = b.dataset.denom;
    document.querySelectorAll("#denom-seg .seg-btn").forEach(x =>
      x.classList.toggle("active", x === b));
  };
});
$("btn-skip-scale").onclick = skipScale;
$("btn-skip-locking").onclick = skipLocking;
$("btn-pack").onclick = showGroundTruth;
$("btn-gt-confirm").onclick = confirmGroundTruth;
$("btn-flip").onclick = () => beginSecondPass("flip");
$("btn-bottom").onclick = () => beginSecondPass("bottom");
$("btn-nobottom").onclick = () => { void finishCapture(); };

// ---------- bundle (zip — JPEG frames stored; WAV + JSON deflated, v3.1) ----------
// v3.1 bundle diet: JPEG frames are already compressed, so they stay
// STORED (method 0) — that part of the old comment was right. But the
// session WAV is raw PCM and the JSON docs are text; in v3 both were
// stored raw too, and a 150 s WAV alone is ~14 MB. Those entries are
// now DEFLATE-compressed (method 8), losslessly: extraction restores
// every byte bit-identically (the proof harness hash-checks the WAV).
const CRC_TABLE = (() => {
  const t = new Int32Array(256);
  for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1; t[n] = c; }
  return t;
})();
function crc32(bytes) {
  let c = 0xFFFFFFFF;
  for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xFF] ^ (c >>> 8);
  return (c ^ 0xFFFFFFFF) >>> 0;
}

// ----- raw DEFLATE (RFC 1951), platform first -----
async function deflateRawPlatform(bytes) {
  const cs = new CompressionStream("deflate-raw");
  const writer = cs.writable.getWriter();
  const reader = cs.readable.getReader();
  const parts = [];
  const pump = (async () => {
    for (;;) { const r = await reader.read(); if (r.done) break; parts.push(r.value); }
  })();
  await writer.write(bytes);
  await writer.close();
  await pump;
  let total = 0; for (const p of parts) total += p.length;
  const out = new Uint8Array(total);
  let o = 0; for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

// ----- vendored raw-DEFLATE encoder (fallback compressor) -----
// A compact fixed-Huffman DEFLATE encoder (RFC 1951), written for
// this app, used only when the platform has no CompressionStream.
// Greedy LZ77 (32 KB window, hash chains) + fixed Huffman codes.
// Its output is a standard raw deflate stream; the v3.1 proof
// round-trips it through zlib and python zipfile bit-identically.
const DEFLATE_LEN_BASE = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
const DEFLATE_LEN_EXT = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const DEFLATE_DIST_BASE = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577];
const DEFLATE_DIST_EXT = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
function deflateRawVendored(bytes) {
  const n = bytes.length;
  const out = [];
  let acc = 0, accBits = 0;
  const putBits = (val, len) => {           // stream bits, LSB-first
    acc |= val << accBits; accBits += len;
    while (accBits >= 8) { out.push(acc & 0xFF); acc >>>= 8; accBits -= 8; }
  };
  const putCode = (code, len) => {          // Huffman codes go MSB-first
    let rev = 0;
    for (let i = 0; i < len; i++) rev = (rev << 1) | ((code >>> i) & 1);
    putBits(rev, len);
  };
  const putLit = sym => {                   // fixed-Huffman literal/length codes
    if (sym < 144) putCode(0x30 + sym, 8);
    else if (sym < 256) putCode(0x190 + sym - 144, 9);
    else if (sym < 280) putCode(sym - 256, 7);
    else putCode(0xC0 + sym - 280, 8);
  };
  const putMatch = (len, dist) => {
    let j;
    if (len === 258) j = 28;
    else { j = 0; while (j < 27 && DEFLATE_LEN_BASE[j + 1] <= len) j++; }
    putLit(257 + j);
    if (DEFLATE_LEN_EXT[j]) putBits(len - DEFLATE_LEN_BASE[j], DEFLATE_LEN_EXT[j]);
    let d = 0; while (d < 29 && DEFLATE_DIST_BASE[d + 1] <= dist) d++;
    putCode(d, 5);                          // fixed distance codes: 5 bits, code = symbol
    if (DEFLATE_DIST_EXT[d]) putBits(dist - DEFLATE_DIST_BASE[d], DEFLATE_DIST_EXT[d]);
  };
  let tokens = [], tokenCount = 0;          // literal: sym >= 0; match: -len, dist
  const flushBlock = final => {
    putBits(final ? 1 : 0, 1); putBits(1, 2);   // BFINAL, BTYPE = fixed Huffman
    for (let k = 0; k < tokens.length; k++) {
      const t = tokens[k];
      if (t >= 0) putLit(t); else putMatch(-t, tokens[++k]);
    }
    putLit(256);                              // end of block
    tokens = []; tokenCount = 0;
  };
  const HASH_SIZE = 1 << 15;
  const head = new Int32Array(HASH_SIZE).fill(-1);
  const prev = new Int32Array(Math.max(1, n)).fill(-1);
  const hashAt = i => Math.imul((bytes[i] << 16) ^ (bytes[i + 1] << 8) ^ bytes[i + 2], 0x9E3779B1) >>> 17;
  const insert = pos => {
    if (pos + 2 < n) { const h = hashAt(pos); prev[pos] = head[h]; head[h] = pos; }
  };
  let i = 0;
  while (i < n) {
    let matchLen = 0, matchDist = 0;
    if (i + 2 < n) {
      let cand = head[hashAt(i)];
      const maxLen = Math.min(258, n - i);
      let tries = 0;
      while (cand >= 0 && i - cand <= 32768 && tries++ < 32) {
        if (matchLen === 0 || bytes[cand + matchLen] === bytes[i + matchLen]) {
          let l = 0;
          while (l < maxLen && bytes[cand + l] === bytes[i + l]) l++;
          if (l >= 3 && l > matchLen) {
            matchLen = l; matchDist = i - cand;
            if (l >= maxLen) break;
          }
        }
        cand = prev[cand];
      }
    }
    if (matchLen >= 3) {
      tokens.push(-matchLen, matchDist); tokenCount++;
      const end = i + matchLen;
      while (i < end) { insert(i); i++; }
    } else {
      tokens.push(bytes[i]); tokenCount++;
      insert(i); i++;
    }
    if (tokenCount >= 16384) flushBlock(i >= n);
  }
  if (tokens.length || !out.length) flushBlock(true);
  if (accBits) { out.push(acc & 0xFF); acc = 0; accBits = 0; }
  return new Uint8Array(out);
}

// Per-blob compression results are cached: buildBundle packs twice
// (the manifest's bundle_compression note is written between the two
// packs) and the WAV must not be compressed twice.
const entryCompressionCache = new WeakMap();
function compressEntry(blob) {
  let p = entryCompressionCache.get(blob);
  if (!p) {
    p = (async () => {
      const bytes = new Uint8Array(await blob.arrayBuffer());
      if (typeof CompressionStream !== "undefined") {
        try {
          const data = await deflateRawPlatform(bytes);
          return { data, method: 8, compressor: "CompressionStream" };
        } catch (e) { /* fall through to the vendored encoder */ }
      }
      try {
        const data = deflateRawVendored(bytes);
        return { data, method: 8, compressor: "vendored" };
      } catch (e) {
        return { data: bytes, method: 0, compressor: null };
      }
    })();
    entryCompressionCache.set(blob, p);
  }
  return p;
}

// Set by every makeZip call: the ACTUAL per-entry methods used and
// the compressor that produced them. buildBundle turns this into the
// manifest's bundle_compression note; it is never guessed in advance.
let lastZipReport = null;

async function makeZip(files) { // files: [{name, blob}]
  const enc = new TextEncoder();
  const chunks = [], central = [];
  const entries = [];
  let compressor = null, offset = 0;
  for (const f of files) {
    const raw = new Uint8Array(await f.blob.arrayBuffer());
    const nameBytes = enc.encode(f.name);
    const crc = crc32(raw);
    let data = raw, method = 0;
    // v3.1: the compressible entries (the session WAV, the JSON docs)
    // are DEFLATE-compressed, losslessly — and only when the deflated
    // form is actually smaller. JPEG frames stay stored.
    if (f.name.endsWith(".wav") || f.name.endsWith(".json")) {
      const r = await compressEntry(f.blob);
      if (r.compressor) compressor = r.compressor;
      if (r.method === 8 && r.data.length < raw.length) { data = r.data; method = 8; }
    }
    entries.push({ name: f.name, method });
    const local = new DataView(new ArrayBuffer(30));
    local.setUint32(0, 0x04034b50, true); local.setUint16(4, 20, true);
    local.setUint16(8, method, true);
    local.setUint32(14, crc, true); local.setUint32(18, data.length, true); local.setUint32(22, raw.length, true);
    local.setUint16(26, nameBytes.length, true);
    chunks.push(new Uint8Array(local.buffer), nameBytes, data);
    const cen = new DataView(new ArrayBuffer(46));
    cen.setUint32(0, 0x02014b50, true); cen.setUint16(4, 20, true); cen.setUint16(6, 20, true);
    cen.setUint16(10, method, true);
    cen.setUint32(16, crc, true); cen.setUint32(20, data.length, true); cen.setUint32(24, raw.length, true);
    cen.setUint16(28, nameBytes.length, true); cen.setUint32(42, offset, true);
    central.push(new Uint8Array(cen.buffer), nameBytes);
    offset += 30 + nameBytes.length + data.length;
  }
  lastZipReport = { compressor, entries };
  const centralSize = central.reduce((a, c) => a + c.length, 0);
  const end = new DataView(new ArrayBuffer(22));
  end.setUint32(0, 0x06054b50, true);
  end.setUint16(8, files.length, true); end.setUint16(10, files.length, true);
  end.setUint32(12, centralSize, true); end.setUint32(16, offset, true);
  return new Blob([...chunks, ...central, new Uint8Array(end.buffer)], { type: "application/zip" });
}

// The manifest's packing note, built from a finished pack's ACTUAL
// per-entry methods (from lastZipReport) — never from intentions.
function bundleCompressionNote(report, hasAudio) {
  const byName = {};
  report.entries.forEach(e => { byName[e.name] = e.method; });
  const how = name => byName[name] === 8 ? "deflated (lossless)" : "stored";
  return {
    audio_wav: hasAudio ? how("audio.wav") : null,
    json: how("scan.json"),
    frames_jpg: "stored",
    compressor: report.compressor,
    note: "Since v3.1 the session WAV and the JSON docs are DEFLATE-compressed inside the zip when a compressor is available and the deflated form is smaller; otherwise the entry is stored. Extraction always restores the original bytes bit-identically. JPEG frames are stored as-is (they do not compress further). A stored audio_wav means this browser had no working compressor, or deflate did not shrink the recording."
  };
}

let bundleBlob = null, bundleName = "";
async function buildBundle() {
  show("send");
  $("big-code").textContent = currentCode;
  $("btn-share").textContent = "Packing your scan…";
  $("btn-share").disabled = true;
  const files = kept.map((b, i) => ({ name: "frame_" + String(i).padStart(3, "0") + ".jpg", blob: b }));
  // v3.2: the stills ride in the bundle under their native
  // names — the initiation still is the scan's scope evidence,
  // the scale still is the lock's first-read evidence.
  if (initiationBlob) files.push({ name: "initiation.jpg", blob: initiationBlob });
  if (scaleBlob) files.push({ name: "scale-still.jpg", blob: scaleBlob });
  // v3: the session sound recording rides in the bundle as
  // audio.wav — the same bundle shape the native app produces.
  if (bundleAudioBlob) files.push({ name: "audio.wav", blob: bundleAudioBlob });
  const meta = captureMeta || { code: currentCode, paper: "LETTER", n_frames: kept.length,
                 pipeline: "browser-web", frames_meta: frameMeta,
                 made_with: "Scan the World test app v3.5 (web capture)" };
  const doc = JSON.stringify(meta, null, 2);
  files.push({ name: "scan.json", blob: new Blob([doc], { type: "application/json" }) });
  files.push({ name: "manifest.json", blob: new Blob([doc], { type: "application/json" }) });
  try {
    bundleBlob = await makeZip(files);
    // The manifest states how this bundle was actually packed
    // (bundle_compression) — knowable only after a first pack, so
    // pack, record the note in the manifest, and pack again with the
    // note inside. The second pack reuses the cached WAV compression
    // and is cheap. The methods the note reports are the methods the
    // first pack used; the second pack takes the identical code path
    // on identical entries (only the JSON text gained the note).
    if (lastZipReport) {
      meta.bundle_compression = bundleCompressionNote(lastZipReport, !!bundleAudioBlob);
      const doc2 = JSON.stringify(meta, null, 2);
      files[files.length - 2] = { name: "scan.json", blob: new Blob([doc2], { type: "application/json" }) };
      files[files.length - 1] = { name: "manifest.json", blob: new Blob([doc2], { type: "application/json" }) };
      bundleBlob = await makeZip(files);
    }
  } catch (e) {
    // Fail loudly, never a dead button: the scan is still on this
    // screen and the failure says what happened.
    $("send-note").textContent = "Packing the scan file failed (" + errMsg(e) + "). Your photos are still taken — go back and try the scan again; if it repeats, tell Homie.";
    $("btn-share").textContent = "Packing failed";
    return;
  }
  bundleName = "scan-" + currentCode + ".zip";
  const mb = (bundleBlob.size / 1048576).toFixed(1);
  $("bundle-size").textContent = "One file, about " + mb + " MB" +
    (bundleAudioBlob ? " (photos + sound recording)" : " (photos only — no sound recording this time)") +
    ". Fine to send on mobile data.";
  const scans = loadScans(); scans.push({ code: currentCode, ts: Date.now() }); saveScans(scans); renderScanList();
  $("btn-share").disabled = false;
  $("btn-share").textContent = "Share the scan file";
  const link = $("download-link");
  link.href = URL.createObjectURL(bundleBlob); link.download = bundleName;
  link.classList.remove("hidden");
}
$("btn-share").onclick = async () => {
  if (!bundleBlob) return;
  const file = new File([bundleBlob], bundleName, { type: "application/zip" });
  if (navigator.canShare && navigator.canShare({ files: [file] })) {
    try { await navigator.share({ files: [file], title: "Scan " + currentCode }); } catch (e) { /* user cancelled */ }
  } else {
    $("download-link").click();
  }
};
$("btn-send-home").onclick = () => show("home");

// ---------- result ----------
const LABEL_WORDS = {
  "uncalibrated": ["Shape captured — no size yet", "", "No measuring reference was found in these photos, so this scan has shape but no size. Scan on the printed mat to get measurements."],
  "ar-estimated": ["Rough size estimate", "warn", "The size comes from the phone's own motion sensing. Treat it as a rough guide."],
  "reference-weak": ["Size measured — treat as approximate", "warn", "A reference was found, but the readings were noisy. The size below is a best effort, not a promise."],
  "reference-single": ["Measured from one reference", "", "One measuring reference was read clearly."],
  "reference-validated": ["Measured and cross-checked", "good", "Two independent references agreed on the size."],
  "independently-verified": ["Independently verified", "good", "Checked against a source beyond these photos."]
};
const WITNESS_NAMES = {
  "mat-sheet": "Printed mat", "ref-card": "Pocket card", "cal-block": "Measuring block",
  "coin": "Coin", "depth": "Phone depth sensor", "post-hoc": "Maker's stated size"
};
function witnessName(p) { return WITNESS_NAMES[p] || p; }

async function openResult(code) {
  code = code.toUpperCase();
  show("result");
  $("result-title").textContent = "Scan " + code;
  const body = $("result-body");
  body.innerHTML = '<p class="muted">Loading…</p>';
  // Simple Scan results (tier "simple") have their own rendering:
  // shape only, no measurements, and NO dimension fields anywhere.
  let simple = null;
  try {
    const sr = await fetch("results/" + encodeURIComponent(code) + "/simple.json", { cache: "no-store" });
    if (sr.ok) simple = await sr.json();
  } catch (e) { /* not a simple scan */ }
  if (simple && simple.tier === "simple") { renderSimpleResult(code, simple, body); return; }
  let ev = null;
  try {
    const res = await fetch("results/" + encodeURIComponent(code) + "/evidence.json", { cache: "no-store" });
    if (res.ok) ev = await res.json();
  } catch (e) { /* offline etc. */ }
  if (!ev) {
    body.innerHTML = "";
    const card = document.createElement("div"); card.className = "card";
    card.innerHTML = "<h2>Still being measured</h2><p>No result for this code yet. If you just sent the scan file, give it about 20 minutes, then check again.</p>";
    body.appendChild(card);
    const b = document.createElement("button"); b.className = "primary"; b.textContent = "Check again";
    b.onclick = () => openResult(code);
    body.appendChild(b);
    return;
  }
  body.innerHTML = "";

  // digital trophy — a viewable 3D keepsake, shown above the evidence
  // when this scan has a model. The evidence below stays the record:
  // the label and measured size are the measurement, the model is
  // for looking at. No model -> the page is exactly as before.
  const modelUrl = "results/" + encodeURIComponent(code) + "/model.glb";
  let hasModel = false;
  try {
    const mr = await fetch(modelUrl, { method: "HEAD", cache: "no-store" });
    hasModel = mr.ok;
  } catch (e) { /* no model / offline — evidence only */ }
  if (hasModel) {
    const trophy = document.createElement("div"); trophy.className = "card";
    trophy.innerHTML = "<h2>Your digital trophy</h2>";
    const viewer = document.createElement("model-viewer");
    viewer.className = "trophy";
    viewer.setAttribute("src", modelUrl);
    viewer.setAttribute("alt", "3D model of your scanned object");
    viewer.setAttribute("camera-controls", "");
    viewer.setAttribute("touch-action", "pan-y");
    viewer.setAttribute("shadow-intensity", "0.6");
    const fallback = document.createElement("p");
    fallback.className = "muted trophy-note";
    fallback.textContent = "Drag to turn it, pinch to zoom. This model is a keepsake — the measured size below is the record.";
    viewer.addEventListener("error", () => {
      viewer.remove();
      fallback.textContent = "The 3D model couldn't be shown on this phone. Your measured result below is complete without it.";
    });
    trophy.appendChild(viewer);
    trophy.appendChild(fallback);
    body.appendChild(trophy);
  }

  const label = (ev.label && ev.label.label) || "uncalibrated";
  const words = LABEL_WORDS[label] || LABEL_WORDS["uncalibrated"];
  let html = '<div class="card"><div class="label-big ' + (words[1] ? "label-" + words[1] : "") + '"></div><p class="label-note"></p></div>';
  body.insertAdjacentHTML("beforeend", html);
  body.querySelector(".label-big").textContent = words[0];
  body.querySelector(".label-note").textContent = words[2];

  // dimensions — only when a scaled measurement exists
  const bbox = ev.object_bbox_mm;
  const dims = bbox && (bbox.dims_mm || bbox.dims || (Array.isArray(bbox) ? bbox : null));
  if (dims && label !== "uncalibrated") {
    const sorted = dims.slice().sort((a, b) => b - a).map(v => Math.round(v));
    const card = document.createElement("div"); card.className = "card";
    card.innerHTML = '<h2>Measured size</h2><p class="dims"></p><p class="muted">Longest, middle and shortest sides of the object.</p>';
    card.querySelector(".dims").textContent = "About " + sorted.join(" × ") + " mm";
    body.appendChild(card);
  }

  // witnesses
  const fusion = ev.fusion || {};
  const agreeBy = {}; (fusion.agreements || []).forEach(a => { agreeBy[a.a + "|" + a.b] = a.agree; });
  const card = document.createElement("div"); card.className = "card";
  card.innerHTML = "<h2>What was measured against</h2>";
  (ev.evidences || []).forEach(e => {
    const row = document.createElement("div"); row.className = "witness";
    const nm = document.createElement("span"); nm.textContent = witnessName(e.provider);
    const st = document.createElement("span"); st.className = "w-status";
    st.textContent = e.scale_mm_per_unit != null ? "gave a reading" : "no reading";
    row.appendChild(nm); row.appendChild(st); card.appendChild(row);
  });
  (fusion.agreements || []).filter(a => a.agree).forEach(a => {
    const p = document.createElement("p"); p.className = "muted";
    p.textContent = witnessName(a.a) + " and " + witnessName(a.b).toLowerCase() + " agree.";
    card.appendChild(p);
  });
  (fusion.disagreements || []).forEach(d => {
    const p = document.createElement("p"); p.className = "muted";
    p.textContent = witnessName(d.a) + " and " + witnessName(d.b).toLowerCase() + " didn't agree — that's recorded, not hidden.";
    card.appendChild(p);
  });
  body.appendChild(card);

  const facts = document.createElement("div"); facts.className = "card";
  let factText = (ev.n_frames_kept_after_gates ?? "?") + " of " + (ev.n_frames ?? "?") + " photos were clear enough to use.";
  if (ev.owner_id) factText += " The mat used belongs to registered owner " + ev.owner_id + ".";
  facts.innerHTML = "<h2>Scan facts</h2><p></p>";
  facts.querySelector("p").textContent = factText;
  body.appendChild(facts);

  const link = document.createElement("a");
  link.className = "inline"; link.href = "results/" + encodeURIComponent(code) + "/evidence.json";
  link.textContent = "See the full evidence (JSON)";
  body.appendChild(link);
}
// ---------- simple-scan result ----------
// Renders ONLY shape information for a Simple Scan. There are no
// dimension fields in this path by design: Simple Scan never claims
// measurements. An "insufficient" verdict is coaching, not an error.
function renderSimpleResult(code, r, body) {
  body.innerHTML = "";
  const head = document.createElement("div"); head.className = "card";
  head.innerHTML = "<h2></h2><p></p>";
  if (r.verdict === "model") {
    head.querySelector("h2").textContent = "Your 3D shape";
    head.querySelector("p").textContent =
      "Made from " + r.registered_frames + " of " + r.total_frames + " photos.";
    body.appendChild(head);
    const modelUrl = "results/" + encodeURIComponent(code) + "/model.glb";
    const card = document.createElement("div"); card.className = "card";
    const viewer = document.createElement("model-viewer");
    viewer.className = "trophy";
    viewer.setAttribute("src", modelUrl);
    viewer.setAttribute("alt", "3D shape of your scanned object");
    viewer.setAttribute("camera-controls", "");
    viewer.setAttribute("touch-action", "pan-y");
    viewer.setAttribute("shadow-intensity", "0.6");
    const note = document.createElement("p");
    note.className = "muted trophy-note";
    note.textContent = "Drag to turn it, pinch to zoom.";
    viewer.addEventListener("error", () => {
      viewer.remove();
      note.textContent = "The 3D shape couldn't be shown on this phone.";
    });
    card.appendChild(viewer); card.appendChild(note);
    body.appendChild(card);
    const link = document.createElement("a");
    link.className = "inline"; link.href = modelUrl; link.download = "scan-" + code + "-shape.glb";
    link.textContent = "Download the 3D shape (GLB)";
    body.appendChild(link);
  } else {
    head.querySelector("h2").textContent = "These photos can't make a model yet";
    head.querySelector("p").textContent =
      "That's fixable — it's about how the photos were taken, not about you doing something wrong. Try again with:";
    body.appendChild(head);
    const tips = document.createElement("div"); tips.className = "card";
    const ul = document.createElement("ul"); ul.className = "coach";
    (r.reasons && r.reasons.length ? r.reasons : []).forEach(reason => {
      const li = document.createElement("li"); li.textContent = reason; ul.appendChild(li);
    });
    ["Go slower, and pause for a heartbeat at each step.",
     "Get closer, so the object fills at least half the screen.",
     "Use more, even light — shade any harsh glare.",
     "Put the object on a matte, textured surface (a patterned cloth or wooden table beats shiny metal)."
    ].forEach(t => { const li = document.createElement("li"); li.textContent = t; ul.appendChild(li); });
    tips.appendChild(ul);
    body.appendChild(tips);
    const again = document.createElement("button");
    again.className = "primary"; again.textContent = "Scan again";
    again.onclick = () => show("home");
    body.appendChild(again);
  }
  const shape = document.createElement("div"); shape.className = "card";
  shape.innerHTML = "<h2>Shape only — no measurements</h2><p class=\"muted\">A Simple Scan captures the shape of an object, not its size. For measurements, scan on the printed mat instead.</p>";
  body.appendChild(shape);
}

$("btn-back").onclick = () => { renderScanList(); show("home"); };

// ---------- boot ----------
renderScanList();
renderMethodSeg();
renderSoundSeg();
initLabelField();
if ("serviceWorker" in navigator && location.protocol === "https:") {
  navigator.serviceWorker.register("sw.js").catch(() => {});
}
