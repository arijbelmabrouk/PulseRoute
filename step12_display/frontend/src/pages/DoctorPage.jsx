import { useWebSocket } from '../hooks/useWebSocket'
import {
  LineChart, Line, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid, ReferenceLine
} from 'recharts'
import { useState, useEffect } from 'react'

const C = {
  blue:      '#1a5dab',
  blueLight: '#e8f1fb',
  blueMid:   '#2d79d6',
  navy:      '#1a2b4a',
  navyMid:   '#2c3e5a',
  bodyText:  '#3d4f6b',
  muted:     '#6b7a91',
  border:    '#dce4ef',
  borderLt:  '#edf2f8',
  bg:        '#f5f7fa',
  white:     '#ffffff',
  green:     '#1a9e6e',
  greenBg:   '#eaf7f1',
  amber:     '#c47a10',
  amberBg:   '#fdf3e1',
  rose:      '#c0392b',
  roseBg:    '#fcecea',
  sans:      "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
  mono:      "'JetBrains Mono', 'Fira Mono', monospace",
}

const RANGES = {
  hr:         { lo: 60,   hi: 100,  min: 30,  max: 180 },
  rr:         { lo: 12,   hi: 20,   min: 4,   max: 40  },
  rmssd:      { lo: 20,   hi: 80,   min: 0,   max: 150 },
  snr:        { lo: 0.45, hi: 1.0,  min: 0,   max: 1.0 },
  confidence: { lo: 0.5,  hi: 1.0,  min: 0,   max: 1.0 },
}

function getStatus(val, key) {
  const r = RANGES[key]
  if (!r || val == null) return null
  const v = parseFloat(val)
  if (v < r.lo) return 'low'
  if (v > r.hi) return 'high'
  return 'ok'
}

function getRangePct(val, key) {
  const r = RANGES[key]
  if (!r || val == null) return 0
  return Math.max(0, Math.min(98, ((parseFloat(val) - r.min) / (r.max - r.min)) * 100))
}

function statusColor(s) {
  if (s === 'ok')   return C.green
  if (s === 'low' || s === 'high') return C.amber
  return C.muted
}

function MetricCard({ label, value, unit, sub, rangeKey, flagged, flagText, dimmed }) {
  const status = rangeKey ? getStatus(value, rangeKey) : null
  const pct    = rangeKey ? getRangePct(value, rangeKey) : null
  const sColor = statusColor(status)
  const hasVal = value != null

  return (
    <div style={{
      background: C.white,
      border: `1px solid ${flagged ? C.amber : C.border}`,
      borderRadius: 12,
      padding: '1.1rem 1.2rem',
      opacity: dimmed ? 0.45 : 1,
      transition: 'opacity 0.3s, border-color 0.3s',
      borderTop: `3px solid ${hasVal ? (flagged ? C.amber : sColor || C.blue) : C.border}`,
    }}>
      <p style={{
        fontSize: 11, fontWeight: 600, letterSpacing: '0.07em',
        textTransform: 'uppercase', color: C.muted,
        marginBottom: '0.55rem', fontFamily: C.sans,
      }}>{label}</p>

      <div style={{ display: 'flex', alignItems: 'baseline', gap: 5, flexWrap: 'wrap' }}>
        <span style={{
          fontSize: '1.9rem', fontWeight: 700, lineHeight: 1,
          color: hasVal ? (flagged ? C.amber : sColor || C.navy) : C.border,
          fontFamily: C.sans, transition: 'color 0.3s',
          letterSpacing: '-0.02em',
        }}>{value ?? '—'}</span>
        {hasVal && unit && (
          <span style={{ fontSize: 12, color: C.muted, fontFamily: C.sans }}>{unit}</span>
        )}
        {flagged && (
          <span style={{
            fontSize: 10, fontWeight: 600, letterSpacing: '0.06em',
            padding: '2px 8px', borderRadius: 20,
            background: C.amberBg, color: C.amber,
            textTransform: 'uppercase', fontFamily: C.sans,
          }}>{flagText || 'flag'}</span>
        )}
      </div>

      {sub && (
        <p style={{ fontSize: 11, color: C.muted, marginTop: 4, fontFamily: C.sans }}>{sub}</p>
      )}

      {rangeKey && hasVal && (
        <div style={{ marginTop: '0.85rem' }}>
          <div style={{
            height: 4, background: C.borderLt, borderRadius: 4,
            position: 'relative', overflow: 'visible',
          }}>
            <div style={{
              position: 'absolute', height: '100%', borderRadius: 4,
              left: `${getRangePct(RANGES[rangeKey].lo, rangeKey)}%`,
              width: `${getRangePct(RANGES[rangeKey].hi, rangeKey) - getRangePct(RANGES[rangeKey].lo, rangeKey)}%`,
              background: C.blueLight,
            }} />
            <div style={{
              position: 'absolute', top: -4, width: 12, height: 12,
              borderRadius: '50%', background: sColor || C.blue,
              left: `${pct}%`, transform: 'translateX(-50%)',
              border: `2px solid ${C.white}`,
              boxShadow: `0 0 0 2px ${sColor || C.blue}44`,
              transition: 'left 0.5s ease',
            }} />
          </div>
          <div style={{
            display: 'flex', justifyContent: 'space-between',
            marginTop: 6, fontSize: 10, color: C.muted, fontFamily: C.sans,
          }}>
            <span>{RANGES[rangeKey].min}</span>
            <span style={{ color: C.blue, fontWeight: 600 }}>
              normal {RANGES[rangeKey].lo}–{RANGES[rangeKey].hi}
            </span>
            <span>{RANGES[rangeKey].max}</span>
          </div>
        </div>
      )}
    </div>
  )
}

function InfoRow({ label, value }) {
  return (
    <div style={{
      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      padding: '7px 0', borderBottom: `1px solid ${C.borderLt}`,
      fontSize: 13, fontFamily: C.sans,
    }}>
      <span style={{ color: C.muted }}>{label}</span>
      <span style={{ color: value ? C.navy : C.border, fontWeight: value ? 500 : 400 }}>
        {value || '—'}
      </span>
    </div>
  )
}

function StepPip({ n, done }) {
  return (
    <div style={{
      width: 30, height: 30, borderRadius: 8, fontSize: 11,
      fontFamily: C.mono, fontWeight: 500,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      background: done ? C.blue : C.bg,
      color: done ? C.white : C.muted,
      border: `1px solid ${done ? C.blue : C.border}`,
      transition: 'all 0.3s',
    }}>{String(n).padStart(2, '0')}</div>
  )
}

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: C.white, border: `1px solid ${C.border}`,
      borderRadius: 8, padding: '6px 12px',
      fontSize: 12, color: C.navy, fontFamily: C.sans,
      boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
    }}>
      SNR {payload[0].value?.toFixed(3)}
    </div>
  )
}

export default function DoctorPage() {
  const { state, connected, snrHistory } = useWebSocket('/ws/doctor')

  const final  = state.final || {}
  const steps  = state.steps || {}
  const s2  = steps['2']  || {}
  const s3  = steps['3']  || {}
  const s6  = steps['6']  || {}
  const s7  = steps['7']  || {}
  const s9  = steps['9']  || {}
  const s10 = steps['10'] || {}

  const isFailed   = state.status === 'failed'
  const isIdle     = state.status === 'idle'
  const isRunning  = state.status === 'running' || state.status === 'starting'
  const isComplete = state.status === 'complete'
  const modality   = state.modality || 'face'

  const hrVal      = isComplete ? final.hr_bpm    : s9.hr_bpm
  const rrVal      = isComplete ? final.rr_bpm    : s10.rr_bpm
  const rmssdVal   = isComplete ? final.rmssd     : s9.rmssd
  const snrVal     = isComplete ? final.snr_score : state.snr_score
  const confVal    = isComplete ? final.confidence: s9.confidence
  const hrReliable = s9.hr_reliable !== false
  const motionPct  = s3.motion_pct

  const handleReset = async () => {
    try { await fetch('http://localhost:8000/api/reset', { method: 'POST' }) }
    catch(e) {}
  }

  return (
    <div style={{
      minHeight: '100vh', background: C.bg,
      fontFamily: C.sans, color: C.navy,
    }}>

      {/* ── Nav ── */}
      <nav style={{
        background: C.white,
        borderBottom: `1px solid ${C.border}`,
        padding: '0 2rem',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        height: 56, position: 'sticky', top: 0, zIndex: 20,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{
            width: 32, height: 32, borderRadius: '50%',
            border: `2.5px solid ${C.blue}`,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <div style={{
              width: 16, height: 16, borderRadius: '50%',
              border: `2px solid ${C.blue}`,
            }} />
          </div>
          <div>
            <span style={{ fontWeight: 700, fontSize: 15, color: C.navy }}>PulseRoute</span>
            <span style={{
              marginLeft: 10, fontSize: 12, color: C.muted,
              borderLeft: `1px solid ${C.border}`, paddingLeft: 10,
            }}>Clinical Dashboard</span>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{
            fontSize: 12, fontWeight: 600, padding: '4px 14px', borderRadius: 20,
            background: modality === 'palm' ? C.amberBg : C.blueLight,
            color: modality === 'palm' ? C.amber : C.blue,
            textTransform: 'uppercase', letterSpacing: '0.05em',
          }}>{modality}</span>

          <div style={{
            display: 'flex', alignItems: 'center', gap: 6,
            fontSize: 13, fontWeight: 500,
            color: isFailed ? C.rose : isComplete ? C.green : isRunning ? C.blue : C.muted,
          }}>
            <span style={{
              width: 7, height: 7, borderRadius: '50%',
              background: isFailed ? C.rose : isComplete ? C.green : isRunning ? C.blue : C.muted,
              animation: isRunning ? 'blink 1.4s infinite' : 'none',
            }} />
            {isFailed ? 'Failed' : isComplete ? 'Complete' : isRunning ? 'Measuring' : 'Idle'}
          </div>

          <span style={{
            fontSize: 11, fontWeight: 600, color: connected ? C.green : C.muted,
            letterSpacing: '0.06em', textTransform: 'uppercase',
          }}>{connected ? 'Live' : 'Reconnecting'}</span>

          {(isComplete || isFailed) && (
            <button onClick={handleReset} style={{
              fontSize: 13, fontWeight: 600, color: C.blue,
              background: C.blueLight, border: 'none',
              borderRadius: 20, padding: '6px 18px', cursor: 'pointer',
              transition: 'background 0.2s',
            }}
            onMouseEnter={e => e.currentTarget.style.background = '#d0e4f7'}
            onMouseLeave={e => e.currentTarget.style.background = C.blueLight}>
              Re-measure
            </button>
          )}
        </div>
      </nav>

      {/* Progress bar */}
      <div style={{ height: 3, background: C.borderLt }}>
        <div style={{
          height: '100%', width: `${state.progress || 0}%`,
          background: isFailed ? C.rose : isComplete ? C.green : C.blue,
          transition: 'width 0.5s ease',
          borderRadius: '0 3px 3px 0',
        }} />
      </div>

      <div style={{ padding: '1.5rem 2rem', maxWidth: 1280, margin: '0 auto' }}>

        {/* ── Alert banners ── */}
        {isFailed && (
          <div style={{
            padding: '12px 16px', marginBottom: '1.25rem', borderRadius: 10,
            background: C.roseBg, border: `1px solid ${C.rose}44`,
            fontSize: 13, color: C.rose, fontWeight: 500,
          }}>
            Measurement failed — {state.message || 'signal too weak for reliable results'}. Ask the patient to improve lighting and re-measure.
          </div>
        )}
        {!isFailed && !hrReliable && s9.hr_agreement && (
          <div style={{
            padding: '10px 16px', marginBottom: '1rem', borderRadius: 10,
            background: C.amberBg, border: `1px solid ${C.amber}44`,
            fontSize: 13, color: C.amber, fontWeight: 500,
          }}>
            HR low confidence — FFT and peak detection disagree by {s9.hr_agreement} BPM. Interpret with caution.
          </div>
        )}
        {isRunning && modality === 'palm' && (
          <div style={{
            padding: '10px 16px', marginBottom: '1rem', borderRadius: 10,
            background: C.blueLight, border: `1px solid ${C.blue}44`,
            fontSize: 13, color: C.blue, fontWeight: 500,
            display: 'flex', alignItems: 'center', gap: 8,
          }}>
            <span style={{
              width: 6, height: 6, borderRadius: '50%', background: C.blue,
              animation: 'blink 1.4s infinite', flexShrink: 0,
            }} />
            Palm mode active — patient is showing palm to camera
          </div>
        )}

        {/* ── Main grid ── */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 280px', gap: 16 }}>

          <div>
            {/* Primary vitals */}
            <p style={{
              fontSize: 11, fontWeight: 700, letterSpacing: '0.08em',
              textTransform: 'uppercase', color: C.muted,
              marginBottom: '0.75rem',
            }}>Primary vitals</p>
            <div style={{
              display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)',
              gap: 12, marginBottom: 16,
            }}>
              <MetricCard label="Heart rate" value={hrVal} unit="BPM"
                rangeKey="hr" dimmed={isIdle}
                flagged={!hrReliable} flagText="low conf" />
              <MetricCard label="Respiratory" value={rrVal} unit="BrPM"
                rangeKey="rr" dimmed={isIdle} />
              <MetricCard label="RMSSD" value={rmssdVal} unit="ms"
                rangeKey="rmssd" dimmed={isIdle}
                sub={!s9.hrv_available ? s9.hrv_fps_message : null} />
              <MetricCard label="HRV assessment"
                value={isComplete ? final.hrv_overall : s9.hrv_overall}
                dimmed={isIdle} />
            </div>

            {/* Signal quality */}
            <p style={{
              fontSize: 11, fontWeight: 700, letterSpacing: '0.08em',
              textTransform: 'uppercase', color: C.muted,
              marginBottom: '0.75rem',
            }}>Signal quality</p>
            <div style={{
              display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)',
              gap: 12, marginBottom: 16,
            }}>
              <MetricCard label="SNR score"
                value={snrVal != null ? snrVal.toFixed(3) : null}
                rangeKey="snr" dimmed={isIdle}
                sub={isComplete ? final.quality_level?.toUpperCase() : null} />
              <MetricCard label="HR confidence"
                value={confVal} rangeKey="confidence" dimmed={isIdle} />
              <MetricCard label="Motion rejected"
                value={motionPct != null ? `${motionPct}%` : null}
                dimmed={isIdle}
                flagged={motionPct > 30} flagText="high"
                sub={s3.frames_rejected != null ? `${s3.frames_rejected} frames` : null} />
              <MetricCard label="FFT SNR ratio"
                value={s7.snr_ratio != null ? `${s7.snr_ratio}x` : null}
                dimmed={isIdle} />
            </div>

            {/* SNR chart */}
            <div style={{
              background: C.white, border: `1px solid ${C.border}`,
              borderRadius: 12, padding: '1.25rem',
            }}>
              <div style={{
                display: 'flex', justifyContent: 'space-between',
                alignItems: 'center', marginBottom: '1rem',
              }}>
                <p style={{
                  fontSize: 11, fontWeight: 700, letterSpacing: '0.08em',
                  textTransform: 'uppercase', color: C.muted,
                }}>SNR score — session trace</p>
                <div style={{ display: 'flex', gap: 16, fontSize: 11, color: C.muted }}>
                  {[[C.green, 'High', '≥0.70'], [C.amber, 'Medium', '≥0.45'], [C.rose, 'Low', '<0.45']].map(([c, l, r]) => (
                    <div key={l} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                      <div style={{ width: 14, height: 2, background: c, borderRadius: 2 }} />
                      {l} {r}
                    </div>
                  ))}
                </div>
              </div>

              {snrHistory?.length > 1 ? (
                <ResponsiveContainer width="100%" height={150}>
                  <LineChart data={snrHistory} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="4 4" stroke={C.borderLt} />
                    <XAxis dataKey="t" hide />
                    <YAxis domain={[0, 1]}
                      tick={{ fontSize: 11, fill: C.muted, fontFamily: C.sans }} />
                    <ReferenceLine y={0.70} stroke={C.green} strokeDasharray="5 3" strokeWidth={1} />
                    <ReferenceLine y={0.45} stroke={C.amber} strokeDasharray="5 3" strokeWidth={1} />
                    <Tooltip content={<CustomTooltip />} />
                    <Line type="monotone" dataKey="snr" stroke={C.blue}
                      strokeWidth={2} dot={false} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div style={{
                  height: 150, display: 'flex', alignItems: 'center',
                  justifyContent: 'center', fontSize: 13, color: C.muted,
                }}>
                  {isIdle ? 'Awaiting pipeline start' : 'Collecting trace data…'}
                </div>
              )}
            </div>
          </div>

          {/* ── Subject profile panel ── */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div style={{
              background: C.white, border: `1px solid ${C.border}`,
              borderRadius: 12, padding: '1.1rem',
            }}>
              <p style={{
                fontSize: 11, fontWeight: 700, letterSpacing: '0.08em',
                textTransform: 'uppercase', color: C.muted, marginBottom: '0.75rem',
              }}>Subject profile</p>
              <InfoRow label="ITA"         value={s2.ita != null ? String(s2.ita) : null} />
              <InfoRow label="Fitzpatrick" value={s2.fitzpatrick} />
              <InfoRow label="Profile"     value={s2.profile_valid === true ? 'Valid' : s2.profile_valid === false ? 'Fallback' : null} />
              <InfoRow label="Calib HR"    value={s2.hr_estimate ? `${s2.hr_estimate} BPM` : null} />
              <InfoRow label="Signal std"  value={s6.filtered_std != null ? s6.filtered_std.toFixed(5) : null} />
              <InfoRow label="BP cutoff"   value={s6.bandpass_low_hz != null ? `${s6.bandpass_low_hz} Hz` : null} />
              <InfoRow label="FPS"         value={s3.fps ? `${s3.fps}` : null} />
            </div>

            <div style={{
              background: C.white, border: `1px solid ${C.border}`,
              borderRadius: 12, padding: '1.1rem',
            }}>
              <p style={{
                fontSize: 11, fontWeight: 700, letterSpacing: '0.08em',
                textTransform: 'uppercase', color: C.muted, marginBottom: '0.75rem',
              }}>Pipeline steps</p>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                {[1,2,3,4,5,6,7,8,9,10,11].map(n => (
                  <StepPip key={n} n={n} done={!!steps[String(n)]} />
                ))}
              </div>
            </div>

            {isComplete && (
              <div style={{
                background: C.white, border: `1px solid ${C.border}`,
                borderRadius: 12, padding: '1.1rem',
              }}>
                <p style={{
                  fontSize: 11, fontWeight: 700, letterSpacing: '0.08em',
                  textTransform: 'uppercase', color: C.muted, marginBottom: '0.75rem',
                }}>Routing decision</p>
                <div style={{
                  padding: '8px 14px', borderRadius: 20, textAlign: 'center',
                  fontSize: 13, fontWeight: 600,
                  background: final.route_palm ? C.amberBg : C.greenBg,
                  color: final.route_palm ? C.amber : C.green,
                }}>
                  {final.route_palm ? 'Palm (auto-routed)' : 'Face accepted'}
                </div>
                {final.routing_reason && (
                  <p style={{
                    fontSize: 11, color: C.muted, marginTop: 8,
                    lineHeight: 1.5, textAlign: 'center',
                  }}>{final.routing_reason}</p>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div style={{
          marginTop: '1.5rem', paddingTop: '1rem',
          borderTop: `1px solid ${C.border}`,
          display: 'flex', justifyContent: 'space-between',
          fontSize: 11, color: C.muted,
        }}>
          <span>PulseRoute · rPPG Teleconsultation · Investigational use only</span>
          <span>Steps 1–11 · {modality} modality</span>
        </div>
      </div>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
        @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.25} }
        * { box-sizing: border-box; margin: 0; padding: 0; }
      `}</style>
    </div>
  )
}