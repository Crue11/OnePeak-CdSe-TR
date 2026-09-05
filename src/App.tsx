import { useEffect, useId, useState } from "react";

type Mode = "forward" | "reverse";

type CurveData = {
  pos_min: number;
  pos_max: number;
  positions: number[];
  temperatures: number[];
};

type ForwardResult =
  | { kind: "empty" }
  | { kind: "loading" }
  | { kind: "success"; temp: number; peakPos: number }
  | { kind: "ood" }
  | { kind: "error" };

type ReverseResult =
  | { kind: "empty" }
  | { kind: "loading" }
  | { kind: "success"; physicsPos: number; modelPos: number }
  | { kind: "ood" }
  | { kind: "error" };

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function predictTemperature(peakPos: number): Promise<ForwardResult> {
  try {
    const res = await fetch(`${API_BASE}/api/predict/temperature`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ peak_position: peakPos }),
    });
    if (!res.ok) return { kind: "error" };
    const data = await res.json();
    if (data.ood) return { kind: "ood" };
    return { kind: "success", temp: Math.round(data.temperature), peakPos };
  } catch {
    return { kind: "error" };
  }
}

async function predictSignature(temp: number): Promise<ReverseResult> {
  try {
    const res = await fetch(`${API_BASE}/api/predict/signature`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ temperature: temp }),
    });
    if (!res.ok) return { kind: "error" };
    const data = await res.json();
    if (data.ood) return { kind: "ood" };
    return {
      kind: "success",
      physicsPos: Math.round(data.physics_position * 10000) / 10000,
      modelPos: Math.round(data.model_position * 10000) / 10000,
    };
  } catch {
    return { kind: "error" };
  }
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center gap-4 py-12 text-center">
      <div className="w-16 h-16 rounded-full bg-gray-100 flex items-center justify-center">
        <svg width="32" height="32" viewBox="0 0 32 32" fill="none" aria-hidden="true">
          <circle cx="16" cy="16" r="12" stroke="#D1D5DB" strokeWidth="2" />
          <path d="M16 10v6M16 20v2" stroke="#D1D5DB" strokeWidth="2" strokeLinecap="round" />
        </svg>
      </div>
      <p className="text-gray-400 text-base">Your result will appear here</p>
    </div>
  );
}

function ErrorState() {
  return (
    <div className="flex flex-col items-center justify-center gap-4 py-10 text-center px-6">
      <div className="w-14 h-14 rounded-full bg-red-50 flex items-center justify-center">
        <svg width="28" height="28" viewBox="0 0 28 28" fill="none" aria-hidden="true">
          <circle cx="14" cy="14" r="11" stroke="#DC2626" strokeWidth="1.5" />
          <path d="M14 9v5.5M14 17.5v2" stroke="#DC2626" strokeWidth="2" strokeLinecap="round" />
        </svg>
      </div>
      <p className="text-gray-500 text-base leading-relaxed max-w-xs mx-auto">
        Something went wrong reaching the prediction service. Please try again.
      </p>
    </div>
  );
}

function LoadingState() {
  return (
    <div className="flex flex-col items-center justify-center gap-4 py-12 text-center">
      <div className="w-8 h-8 rounded-full border-2 border-gray-200 animate-spin" style={{ borderTopColor: "#1B5E20" }} />
      <p className="text-gray-400 text-base">Predicting…</p>
    </div>
  );
}

// Muted diverging cool-to-warm scale, fading through the page background in the middle.
const HEATMAP_STOPS: [number, [number, number, number]][] = [
  [0, [96, 141, 186]],
  [0.5, [244, 243, 238]],
  [1, [196, 104, 56]],
];

function heatmapColor(norm: number): [number, number, number] {
  const t = Math.min(1, Math.max(0, norm));
  let lo = HEATMAP_STOPS[0];
  let hi = HEATMAP_STOPS[HEATMAP_STOPS.length - 1];
  for (let i = 0; i < HEATMAP_STOPS.length - 1; i++) {
    if (t >= HEATMAP_STOPS[i][0] && t <= HEATMAP_STOPS[i + 1][0]) {
      lo = HEATMAP_STOPS[i];
      hi = HEATMAP_STOPS[i + 1];
      break;
    }
  }
  const span = hi[0] - lo[0] || 1;
  const f = (t - lo[0]) / span;
  return [
    Math.round(lo[1][0] + (hi[1][0] - lo[1][0]) * f),
    Math.round(lo[1][1] + (hi[1][1] - lo[1][1]) * f),
    Math.round(lo[1][2] + (hi[1][2] - lo[1][2]) * f),
  ];
}

function interpolateTemp(curve: CurveData, pos: number): number {
  const { positions, temperatures } = curve;
  const p = Math.min(curve.pos_max, Math.max(curve.pos_min, pos));
  for (let i = 0; i < positions.length - 1; i++) {
    if (p >= positions[i] && p <= positions[i + 1]) {
      const span = positions[i + 1] - positions[i] || 1;
      const f = (p - positions[i]) / span;
      return temperatures[i] + f * (temperatures[i + 1] - temperatures[i]);
    }
  }
  return temperatures[temperatures.length - 1];
}

function CurveVisual({ curve, markerPos, caption }: { curve: CurveData; markerPos: number; caption: string }) {
  const width = 380;
  const height = 90;
  const pad = 12;
  const gradientId = useId();

  const tMin = Math.min(...curve.temperatures);
  const tMax = Math.max(...curve.temperatures);
  const tRange = tMax - tMin || 1;
  const posRange = curve.pos_max - curve.pos_min || 1;

  const xFor = (pos: number) => pad + ((pos - curve.pos_min) / posRange) * (width - pad * 2);
  const yFor = (t: number) => height - pad - ((t - tMin) / tRange) * (height - pad * 2);

  const points = curve.positions.map((p, i) => [xFor(p), yFor(curve.temperatures[i])] as const);
  const pathD = points.map(([x, y], i) => `${i === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`).join(" ");

  const clampedMarkerPos = Math.min(curve.pos_max, Math.max(curve.pos_min, markerPos));
  const markerX = xFor(clampedMarkerPos);
  const markerY = yFor(interpolateTemp(curve, clampedMarkerPos));

  return (
    <div className="mt-6 pt-5 border-t border-gray-100 flex flex-col items-center gap-2.5 w-full">
      <div className="relative w-full max-w-[380px]" style={{ height }}>
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-full" preserveAspectRatio="none">
          <defs>
            <linearGradient id={gradientId} x1="0" y1="0" x2="1" y2="0">
              {points.map(([x], i) => {
                const norm = (curve.temperatures[i] - tMin) / tRange;
                const [r, g, b] = heatmapColor(norm);
                return <stop key={i} offset={`${((x / width) * 100).toFixed(2)}%`} stopColor={`rgb(${r},${g},${b})`} />;
              })}
            </linearGradient>
          </defs>
          <path d={pathD} fill="none" stroke={`url(#${gradientId})`} strokeWidth={2.5} strokeLinecap="round" />
          <circle cx={markerX} cy={markerY} r={5} fill="white" stroke="#1B5E20" strokeWidth={2.5} />
        </svg>
      </div>

      <div className="w-full max-w-[380px] flex items-center justify-between text-xs text-gray-400 px-0.5">
        <span>{curve.pos_min.toFixed(2)}°</span>
        <span>Peak Position (2θ) →</span>
        <span>{curve.pos_max.toFixed(2)}°</span>
      </div>

      <p className="text-sm text-gray-500 text-center">{caption}</p>

      <div className="flex items-center gap-2 text-xs text-gray-400">
        <span>Cooler</span>
        <span
          className="h-1.5 w-16 rounded-full"
          style={{ background: "linear-gradient(to right, rgb(96,141,186), rgb(244,243,238), rgb(196,104,56))" }}
        />
        <span>Warmer</span>
      </div>
    </div>
  );
}

function ForwardResultCard({ result, curve }: { result: ForwardResult; curve: CurveData | null }) {
  if (result.kind === "empty") return <EmptyState />;
  if (result.kind === "loading") return <LoadingState />;
  if (result.kind === "error") return <ErrorState />;

  if (result.kind === "ood") {
    return (
      <div className="flex flex-col items-center justify-center gap-4 py-10 text-center px-6">
        <div className="w-14 h-14 rounded-full bg-amber-50 flex items-center justify-center">
          <svg width="28" height="28" viewBox="0 0 28 28" fill="none" aria-hidden="true">
            <circle cx="14" cy="14" r="11" stroke="#D97706" strokeWidth="1.5" />
            <path d="M14 9v5.5M14 17.5v2" stroke="#D97706" strokeWidth="2" strokeLinecap="round" />
          </svg>
        </div>
        <div>
          <p className="text-xl font-semibold text-gray-700 mb-2" style={{ fontFamily: "var(--font-display)" }}>
            We&apos;re not confident here.
          </p>
          <p className="text-gray-500 text-base leading-relaxed max-w-xs mx-auto">
            This reading doesn&apos;t closely match the patterns we&apos;ve learned from, so we&apos;re holding back a prediction rather than guessing.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center justify-center gap-3 py-10 text-center">
      <p className="text-sm font-medium text-gray-400 uppercase tracking-wider">Estimated Temperature</p>
      <p
        className="text-8xl font-bold leading-none"
        style={{ fontFamily: "var(--font-display)", color: "#1B5E20" }}
      >
        {result.temp}°C
      </p>
      <p className="text-sm text-gray-400 mt-2">Typical accuracy: ±47°C</p>
      {curve && (
        <CurveVisual curve={curve} markerPos={result.peakPos} caption="Your reading, shown against the model's full range." />
      )}
    </div>
  );
}

function ReverseResultCard({ result, curve }: { result: ReverseResult; curve: CurveData | null }) {
  if (result.kind === "empty") return <EmptyState />;
  if (result.kind === "loading") return <LoadingState />;
  if (result.kind === "error") return <ErrorState />;

  if (result.kind === "ood") {
    return (
      <div className="flex flex-col items-center justify-center gap-4 py-10 text-center px-6">
        <div className="w-14 h-14 rounded-full bg-amber-50 flex items-center justify-center">
          <svg width="28" height="28" viewBox="0 0 28 28" fill="none" aria-hidden="true">
            <circle cx="14" cy="14" r="11" stroke="#D97706" strokeWidth="1.5" />
            <path d="M14 9v5.5M14 17.5v2" stroke="#D97706" strokeWidth="2" strokeLinecap="round" />
          </svg>
        </div>
        <div>
          <p className="text-xl font-semibold text-gray-700 mb-2" style={{ fontFamily: "var(--font-display)" }}>
            We&apos;re not confident here.
          </p>
          <p className="text-gray-500 text-base leading-relaxed max-w-xs mx-auto">
            This temperature is outside the range we&apos;ve learned from, so we&apos;re holding back a prediction rather than guessing.
          </p>
        </div>
      </div>
    );
  }

  const avgPos = (result.physicsPos + result.modelPos) / 2;

  return (
    <div className="flex flex-col items-center justify-center gap-3 py-10 text-center">
      <p className="text-sm font-medium text-gray-400 uppercase tracking-wider">Expected Position</p>
      <p
        className="text-8xl font-bold leading-none"
        style={{ fontFamily: "var(--font-display)", color: "#1B5E20" }}
      >
        {avgPos.toFixed(4)}°
      </p>
      <p className="text-sm text-gray-400 mt-2">
        Physics: {result.physicsPos.toFixed(4)}° · Model: {result.modelPos.toFixed(4)}°
      </p>
      {curve && (
        <CurveVisual
          curve={curve}
          markerPos={avgPos}
          caption="The expected reading at this temperature, shown against the model's full range."
        />
      )}
    </div>
  );
}

export default function App() {
  const [mode, setMode] = useState<Mode>("forward");
  const [curve, setCurve] = useState<CurveData | null>(null);

  useEffect(() => {
    fetch(`${API_BASE}/api/curve`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => data && setCurve(data))
      .catch(() => {});
  }, []);

  // Forward mode state
  const [peakPos, setPeakPos] = useState("");
  const [forwardResult, setForwardResult] = useState<ForwardResult>({ kind: "empty" });

  // Reverse mode state
  const [tempInput, setTempInput] = useState("");
  const [reverseResult, setReverseResult] = useState<ReverseResult>({ kind: "empty" });

  async function handleForwardPredict() {
    const pos = parseFloat(peakPos);
    if (isNaN(pos)) return;
    setForwardResult({ kind: "loading" });
    setForwardResult(await predictTemperature(pos));
  }

  async function handleReversePredict() {
    const temp = parseFloat(tempInput);
    if (isNaN(temp)) return;
    setReverseResult({ kind: "loading" });
    setReverseResult(await predictSignature(temp));
  }

  function switchMode(m: Mode) {
    setMode(m);
    setForwardResult({ kind: "empty" });
    setReverseResult({ kind: "empty" });
  }

  const inputClass =
    "w-full rounded-xl border border-gray-200 bg-white px-4 py-3.5 text-xl text-gray-900 placeholder-gray-300 outline-none transition focus:border-[#1B5E20] focus:ring-2 focus:ring-[#1B5E20]/15";

  const labelClass = "block text-sm font-semibold text-gray-600 mb-1.5";

  return (
    <div className="min-h-full py-10 px-6" style={{ backgroundColor: "#F7F7F4" }}>
      {/* Centered column */}
      <div className="mx-auto w-full" style={{ maxWidth: 960 }}>

        {/* Header */}
        <div className="mb-8 text-center">
          <h1
            className="text-3xl font-bold text-gray-900 mb-2"
            style={{ fontFamily: "var(--font-display)" }}
          >
            Non-Contact Temperature Reader
          </h1>
          <p className="text-base text-gray-500 max-w-xl mx-auto">
            Read temperature from an X-ray scan — or see what a scan should look like at a given temperature.
          </p>
        </div>

        {/* Mode switcher */}
        <div className="flex justify-center mb-10">
          <div className="inline-flex rounded-full p-1 bg-white border border-gray-200 shadow-sm">
            <button
              onClick={() => switchMode("forward")}
              className="px-6 py-2 rounded-full text-sm font-semibold transition-all"
              style={
                mode === "forward"
                  ? { backgroundColor: "#1B5E20", color: "#fff" }
                  : { color: "#6B7280" }
              }
            >
              Predict Temperature
            </button>
            <button
              onClick={() => switchMode("reverse")}
              className="px-6 py-2 rounded-full text-sm font-semibold transition-all"
              style={
                mode === "reverse"
                  ? { backgroundColor: "#1B5E20", color: "#fff" }
                  : { color: "#6B7280" }
              }
            >
              Predict Signature
            </button>
          </div>
        </div>

        {/* Main card */}
        <div className="rounded-3xl bg-white shadow-sm border border-gray-100 overflow-hidden">
          {mode === "forward" ? (
            <div className="grid" style={{ gridTemplateColumns: "2fr 3fr" }}>
              {/* Left: Inputs */}
              <div className="p-8 border-r border-gray-100">
                <h2
                  className="text-lg font-semibold text-gray-800 mb-6"
                  style={{ fontFamily: "var(--font-display)" }}
                >
                  Scan Value
                </h2>
                <div className="flex flex-col gap-5">
                  <div>
                    <label className={labelClass}>Peak Position (2θ, degrees)</label>
                    <input
                      type="number"
                      value={peakPos}
                      onChange={(e) => setPeakPos(e.target.value)}
                      placeholder="25.40"
                      className={inputClass}
                      step="0.01"
                    />
                  </div>
                  <button
                    onClick={handleForwardPredict}
                    disabled={!peakPos || forwardResult.kind === "loading"}
                    className="w-full py-3.5 rounded-xl text-base font-semibold text-white transition-all disabled:opacity-40 disabled:cursor-not-allowed mt-1"
                    style={{ backgroundColor: "#1B5E20" }}
                    onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "#155218")}
                    onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "#1B5E20")}
                  >
                    Predict Temperature
                  </button>
                  <p className="text-xs text-gray-400 leading-relaxed">
                    Read this value directly from your plotted scan.
                  </p>
                </div>
              </div>

              {/* Right: Result */}
              <div className="p-8 flex flex-col justify-center">
                <ForwardResultCard result={forwardResult} curve={curve} />
              </div>
            </div>
          ) : (
            <div className="grid" style={{ gridTemplateColumns: "2fr 3fr" }}>
              {/* Left: Input */}
              <div className="p-8 border-r border-gray-100">
                <h2
                  className="text-lg font-semibold text-gray-800 mb-6"
                  style={{ fontFamily: "var(--font-display)" }}
                >
                  Target Temperature
                </h2>
                <div className="flex flex-col gap-5">
                  <div>
                    <label className={labelClass}>Temperature (°C)</label>
                    <input
                      type="number"
                      value={tempInput}
                      onChange={(e) => setTempInput(e.target.value)}
                      placeholder="200"
                      className={inputClass}
                      step="1"
                    />
                  </div>
                  <button
                    onClick={handleReversePredict}
                    disabled={!tempInput || reverseResult.kind === "loading"}
                    className="w-full py-3.5 rounded-xl text-base font-semibold text-white transition-all disabled:opacity-40 disabled:cursor-not-allowed mt-1"
                    style={{ backgroundColor: "#1B5E20" }}
                    onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "#155218")}
                    onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "#1B5E20")}
                  >
                    Predict Signature
                  </button>
                  <p className="text-xs text-gray-400 leading-relaxed">
                    See the peak position we&apos;d expect to see at this temperature.
                  </p>
                </div>
              </div>

              {/* Right: Result */}
              <div className="p-8 flex flex-col justify-center">
                <ReverseResultCard result={reverseResult} curve={curve} />
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <p className="text-center text-xs text-gray-400 mt-8">
          Predictions are estimates based on learned patterns and should not replace laboratory measurement.
        </p>
      </div>
    </div>
  );
}
