import { useWebSocket } from '../hooks/useWebSocket'
import {
  LineChart, Line, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid, ReferenceLine
} from 'recharts'

// ── Clinical normal ranges ────────────────────────────────
const RANGES = {
  hr: {
    low: 40, lowNormal: 60, highNormal: 100, high: 150,
    labels: ['Bradycardia', 'Below normal', 'Normal', 'Elevated', 'Tachycardia'],
  },
  rr: {
    low: 8, lowNormal: 12, highNormal: 20, high: 30,
    labels: ['Dangerously slow', 'Below normal', 'Normal', 'Elevated', 'Tachypnea'],
  },
  rmssd: {
    low: 5, lowNormal: 20, highNormal: 80, high: 200,
    labels: ['Very low HRV', 'Below normal', 'Normal', 'Elevated', 'Very high HRV'],
  },
  snr: {
    low: 0, lowNormal: 0.45, highNormal: 1.0, high: 1.0,
    labels: ['Poor', 'Medium', 'High', 'High', 'High'],
    noHigh: true,
  },
  confidence: {
    low: 0, lowNormal: 0.5, highNormal: 0.8, high: 1.0,
    labels: ['Very low', 'Low', 'Good', 'High', 'High'],
    noHigh: true,
  },
}

function getRangeStatus(value, range) {
  if (value === undefined || value === null || range === undefined) return null
  const v = parseFloat(value)
  if (isNaN(v)) return null
  const { low, lowNormal, highNormal, high, labels, noHigh } = range
  if (v < low)               return { zone: 'danger',  label: labels[0], pct: 0 }
  if (v < lowNormal)         return { zone: 'warning', label: labels[1], pct: Math.round((v - low) / (lowNormal - low) * 25) }
  if (v <= highNormal)       return { zone: 'success', label: labels[2], pct: Math.round(25 + (v - lowNormal) / (highNormal - lowNormal) * 50) }
  if (!noHigh && v <= high)  return { zone: 'warning', label: labels[3], pct: Math.round(75 + (v - highNormal) / (high - highNormal) * 20) }
  if (!noHigh && v > high)   return { zone: 'danger',  label: labels[4], pct: 100 }
  return { zone: 'success', label: labels[2], pct: Math.min(100, Math.round(25 + (v - lowNormal) / (highNormal - lowNormal) * 50)) }
}

// ── Reusable components ───────────────────────────────────

function RangeBar({ value, rangeKey, lowConfidence }) {
  const range = RANGES[rangeKey]
  if (!range) return null
  const status = getRangeStatus(value, range)
  if (!status) return null

  const zoneColor = { success: 'var(--success)', warning: 'var(--warning)', danger: 'var(--danger)' }[status.zone]
  const zoneBg    = { success: 'var(--success-bg)', warning: 'var(--warning-bg)', danger: 'var(--danger-bg)' }[status.zone]
  const zoneText  = { success: 'var(--success)', warning: 'var(--warning)', danger: 'var(--danger)' }[status.zone]
  const effectiveLabel = lowConfidence ? `${status.label} · low confidence` : status.label

  return (
    <div style={{ marginTop: '8px' }}>
      <div style={{ height: 4, borderRadius: 999, background: 'var(--surface2)', position: 'relative', overflow: 'hidden' }}>
        <div style={{
          position: 'absolute', left: 0, top: 0, height: '100%',
          width: `${status.pct}%`, background: zoneColor, borderRadius: 999,
          transition: 'width 0.5s ease', opacity: lowConfidence ? 0.45 : 1,
        }} />
        <div style={{ position: 'absolute', left: '25%', top: 0, height: '100%', width: '50%', background: 'rgba(16,185,129,0.12)' }} />
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: '5px', marginTop: '5px' }}>
        <span style={{
          fontSize: '10px', fontWeight: 500, padding: '2px 7px', borderRadius: 999,
          background: zoneBg, color: zoneText, border: `1px solid ${zoneColor}22`,
          opacity: lowConfidence ? 0.7 : 1, fontFamily: 'var(--font-mono)', letterSpacing: '0.04em',
        }}>{effectiveLabel}</span>
      </div>
    </div>
  )
}

function Metric({ label, value, unit, sub, accent, badge, warn, muted, rangeKey, lowConfidence }) {
  const color = accent ? 'var(--accent)' : warn ? 'var(--warning)' : 'var(--text)'
  return (
    <div style={{
      background: 'var(--surface)', border: '1px solid var(--border)',
      borderRadius: 'var(--radius-md)', padding: '1rem 1.1rem',
      boxShadow: 'var(--shadow-sm)', opacity: muted ? 0.45 : 1, transition: 'opacity 0.3s',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.4rem' }}>
        <p style={{ fontSize: '11px', fontWeight: 500, color: 'var(--text3)', letterSpacing: '0.07em', textTransform: 'uppercase' }}>{label}</p>
        {badge && (
          <span style={{
            fontSize: '10px', fontWeight: 500, padding: '2px 7px', borderRadius: 999,
            background: badge === 'LOW CONF' ? 'var(--danger-bg)' : 'var(--warning-bg)',
            color: badge === 'LOW CONF' ? 'var(--danger)' : 'var(--warning)',
            border: `1px solid ${badge === 'LOW CONF' ? '#fca5a5' : '#fcd34d'}`,
          }}>{badge}</span>
        )}
      </div>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: '4px' }}>
        <span style={{ fontSize: '1.75rem', fontWeight: 600, color, lineHeight: 1 }}>
          {value ?? <span style={{ color: 'var(--text3)', fontSize: '1.4rem' }}>—</span>}
        </span>
        {value && unit && (
          <span style={{ fontSize: '12px', color: 'var(--text3)', fontFamily: 'var(--font-mono)' }}>{unit}</span>
        )}
      </div>
      {sub && <p style={{ fontSize: '11px', color: 'var(--text3)', marginTop: '3px' }}>{sub}</p>}
      {value !== undefined && value !== null && rangeKey && (
        <RangeBar value={value} rangeKey={rangeKey} lowConfidence={lowConfidence} />
      )}
    </div>
  )
}

function SectionLabel({ children }) {
  return (
    <p style={{
      fontSize: '11px', fontWeight: 500, color: 'var(--text3)',
      letterSpacing: '0.08em', textTransform: 'uppercase',
      marginBottom: '0.75rem', marginTop: '1.5rem',
    }}>{children}</p>
  )
}

function InfoRow({ label, value, mono }) {
  return (
    <div style={{
      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      padding: '7px 0', borderBottom: '1px solid var(--border)', fontSize: '13px',
    }}>
      <span style={{ color: 'var(--text2)' }}>{label}</span>
      <span style={{
        color: 'var(--text)', fontWeight: 500,
        fontFamily: mono ? 'var(--font-mono)' : undefined,
        fontSize: mono ? '12px' : undefined,
      }}>{value || '—'}</span>
    </div>
  )
}

function StepBadge({ n, done }) {
  return (
    <div title={`Step ${n}`} style={{
      width: 26, height: 26, borderRadius: 'var(--radius-sm)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontSize: '11px', fontWeight: 500, fontFamily: 'var(--font-mono)',
      background: done ? 'var(--accent-bg)' : 'var(--surface2)',
      color: done ? 'var(--accent)' : 'var(--text3)',
      border: `1px solid ${done ? '#bfdbfe' : 'var(--border)'}`,
      transition: 'all 0.3s',
    }}>{n}</div>
  )
}

function QualityBar({ score }) {
  if (score === undefined) return null
  const pct = Math.round(score * 100)
  const color = pct >= 70 ? 'var(--success)' : pct >= 45 ? 'var(--warning)' : 'var(--danger)'
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
        <span style={{ fontSize: '12px', color: 'var(--text2)' }}>Signal quality</span>
        <span style={{ fontSize: '12px', fontWeight: 600, color, fontFamily: 'var(--font-mono)' }}>{pct}%</span>
      </div>
      <div style={{ height: 6, background: 'var(--surface2)', borderRadius: 999, overflow: 'hidden' }}>
        <div style={{ height: '100%', width: `${pct}%`, background: color, borderRadius: 999, transition: 'width 0.6s ease' }} />
      </div>
    </div>
  )
}

const SNR_TOOLTIP = ({ active, payload }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: 'var(--surface)', border: '1px solid var(--border)',
      borderRadius: 'var(--radius-sm)', padding: '6px 10px',
      fontSize: '12px', boxShadow: 'var(--shadow-sm)', fontFamily: 'var(--font-mono)',
    }}>
      SNR {payload[0].value?.toFixed(3)}
    </div>
  )
}

// ── Doctor page ───────────────────────────────────────────
export default function DoctorPage() {
  const { state, connected, snrHistory } = useWebSocket('/ws/doctor')

  const final      = state.final   || {}
  const steps      = state.steps   || {}
  const isIdle     = state.status === 'idle'
  const isRunning  = state.status === 'running' || state.status === 'starting'
  const isComplete = state.status === 'complete'

  const s2  = steps['2']  || {}
  const s3  = steps['3']  || {}
  const s6  = steps['6']  || {}
  const s7  = steps['7']  || {}
  const s9  = steps['9']  || {}
  const s10 = steps['10'] || {}

  const snrVal    = isComplete ? final.snr_score : state.snr_score
  const hrVal     = isComplete ? final.hr_bpm    : s9.hr_bpm
  const rrVal     = isComplete ? final.rr_bpm    : s10.rr_bpm
  const rmssdVal  = isComplete ? final.rmssd     : s9.rmssd
  const confVal   = isComplete ? final.confidence : s9.confidence
  const confLevel = isComplete ? final.confidence_level : s9.confidence_level
  const qualLevel = isComplete ? final.quality_level : undefined
  const routePalm = isComplete ? final.route_palm : state.route_palm
  const stdTrig   = isComplete ? final.std_floor_triggered : state.std_floor_triggered
  const motionPct = s3.motion_pct

  // ── Re-measurement trigger ────────────────────────────
  const handleRemeasure = async () => {
    try {
      await fetch('http://localhost:8000/api/reset', { method: 'POST' })
    } catch (e) {
      console.error('Reset failed:', e)
    }
  }

  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg)' }}>

      {/* ── Top nav ── */}
      <div style={{
        background: 'var(--surface)', borderBottom: '1px solid var(--border)',
        padding: '0 2rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        height: 56, position: 'sticky', top: 0, zIndex: 10, boxShadow: 'var(--shadow-sm)',
      }}>
        {/* Logo */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            width: 28, height: 28, borderRadius: 'var(--radius-sm)',
            background: 'var(--accent-bg)', border: '1px solid #bfdbfe',
            display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '14px',
          }}>♥</div>
          <div>
            <span style={{ fontWeight: 600, fontSize: '14px', color: 'var(--text)' }}>PulseRoute</span>
            <span style={{ marginLeft: '8px', fontSize: '12px', color: 'var(--text3)', fontFamily: 'var(--font-mono)' }}>
              Clinical Dashboard
            </span>
          </div>
        </div>

        {/* Right side: re-measure button + status + live indicator */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>

          {/* Re-measurement button — only shown when complete or palm routed */}
          {(isComplete || routePalm) && (
            <button
              onClick={handleRemeasure}
              style={{
                display: 'inline-flex', alignItems: 'center', gap: '6px',
                padding: '5px 14px', borderRadius: 999,
                fontSize: '12px', fontWeight: 500,
                background: 'var(--surface)', color: 'var(--accent)',
                border: '1px solid #bfdbfe', cursor: 'pointer',
                transition: 'background 0.15s',
              }}
              onMouseEnter={e => e.currentTarget.style.background = 'var(--accent-bg)'}
              onMouseLeave={e => e.currentTarget.style.background = 'var(--surface)'}
            >
              <span style={{ fontSize: '14px', lineHeight: 1 }}>↺</span>
              Re-measure
            </button>
          )}

          {/* Session status badge */}
          <div style={{
            display: 'inline-flex', alignItems: 'center', gap: '6px',
            padding: '4px 12px', borderRadius: 999, fontSize: '12px', fontWeight: 500,
            ...(isComplete
              ? { background: 'var(--success-bg)', color: 'var(--success)', border: '1px solid #6ee7b7' }
              : isRunning
                ? { background: 'var(--accent-bg)', color: 'var(--accent)', border: '1px solid #bfdbfe' }
                : { background: 'var(--surface2)', color: 'var(--text3)', border: '1px solid var(--border)' }),
          }}>
            <span className={isRunning ? 'blink' : ''} style={{
              width: 6, height: 6, borderRadius: '50%',
              background: isComplete ? 'var(--success)' : isRunning ? 'var(--accent)' : '#d1d5db',
              display: 'inline-block',
            }} />
            {isComplete ? 'Complete' : isRunning ? 'Measuring…' : 'Idle'}
          </div>

          {/* Live / reconnecting indicator */}
          <div style={{
            display: 'flex', alignItems: 'center', gap: '5px',
            fontSize: '11px', color: connected ? 'var(--success)' : 'var(--text3)',
            fontFamily: 'var(--font-mono)',
          }}>
            <span className={connected ? 'blink' : ''} style={{
              width: 6, height: 6, borderRadius: '50%',
              background: connected ? 'var(--success)' : '#d1d5db', display: 'inline-block',
            }} />
            {connected ? 'LIVE' : 'RECONNECTING'}
          </div>
        </div>
      </div>

      {/* ── Progress bar ── */}
      <div style={{ height: 3, background: '#e5e7eb' }}>
        <div style={{
          height: '100%', width: `${state.progress || 0}%`,
          background: isComplete ? 'var(--success)' : 'var(--accent)',
          transition: 'width 0.6s ease',
        }} />
      </div>

      <div style={{ padding: '1.5rem 2rem', maxWidth: 1200, margin: '0 auto' }}>

        {/* ── Routing banner ── */}
        {routePalm !== undefined && (
          <div className="fade-up" style={{
            padding: '11px 16px', marginBottom: '1.5rem',
            borderRadius: 'var(--radius-md)',
            display: 'flex', alignItems: 'center', gap: '10px',
            fontSize: '13px', fontWeight: 500,
            ...(routePalm
              ? { background: 'var(--warning-bg)', border: '1px solid #fcd34d', color: 'var(--warning)' }
              : { background: 'var(--success-bg)', border: '1px solid #6ee7b7', color: 'var(--success)' }),
          }}>
            <span style={{ fontSize: '16px' }}>{routePalm ? '⚠' : '✓'}</span>
            <span>
              {routePalm
                ? `Palm signal recommended — ${final.routing_reason || state.routing_reason || 'signal below threshold'}`
                : 'Face signal accepted — quality sufficient for all metrics'}
            </span>
          </div>
        )}

        {/* ── Primary vitals ── */}
        <SectionLabel>Primary vitals</SectionLabel>
        <div style={{
          display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))',
          gap: '12px', marginBottom: '0.5rem',
        }}>
          <Metric label="Heart rate" value={hrVal} unit="BPM"
            sub={s7.fft_peak_bpm ? `FFT peak ${s7.fft_peak_bpm} BPM` : undefined}
            accent muted={isIdle} rangeKey="hr" />
          <Metric label="Respiratory" value={rrVal} unit="BrPM"
            muted={isIdle} rangeKey="rr" />
          <Metric label="RMSSD" value={rmssdVal} unit="ms"
            warn={stdTrig} badge={stdTrig ? 'LOW CONF' : undefined}
            muted={isIdle} rangeKey="rmssd" lowConfidence={stdTrig} />
          <Metric label="HRV overall" value={isComplete ? final.hrv_overall : s9.hrv_overall}
            muted={isIdle} />
        </div>

        {/* ── Reference legend ── */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: '16px',
          marginBottom: '0.5rem', padding: '7px 12px',
          background: 'var(--surface)', border: '1px solid var(--border)',
          borderRadius: 'var(--radius-sm)', fontSize: '11px', color: 'var(--text3)',
        }}>
          <span style={{ fontWeight: 500, letterSpacing: '0.05em', textTransform: 'uppercase' }}>Range reference</span>
          {[
            ['Normal', 'var(--success)', 'var(--success-bg)'],
            ['Borderline', 'var(--warning)', 'var(--warning-bg)'],
            ['Abnormal', 'var(--danger)', 'var(--danger-bg)'],
          ].map(([label, color, bg]) => (
            <div key={label} style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
              <div style={{ width: 20, height: 4, borderRadius: 999, background: color }} />
              <span>{label}</span>
            </div>
          ))}
          <span style={{ marginLeft: 'auto', fontFamily: 'var(--font-mono)', fontSize: '10px' }}>
            HR 60–100 · RR 12–20 · RMSSD 20–80 · adult population norms
          </span>
        </div>

        {/* ── Signal quality ── */}
        <SectionLabel>Signal quality</SectionLabel>
        <div style={{
          display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))',
          gap: '12px', marginBottom: '0.5rem',
        }}>
          <Metric label="SNR score"
            value={snrVal !== undefined ? snrVal.toFixed(3) : undefined}
            sub={qualLevel ? qualLevel.toUpperCase() : undefined}
            muted={isIdle} rangeKey="snr" />
          <Metric label="HR confidence"
            value={confVal}
            sub={confLevel ? confLevel.toUpperCase() : undefined}
            muted={isIdle} rangeKey="confidence" />
          <Metric label="Motion rejected"
            value={motionPct !== undefined ? `${motionPct}%` : undefined}
            sub={s3.frames_rejected !== undefined ? `${s3.frames_rejected} / ${s3.frames_total} frames` : undefined}
            warn={motionPct > 30} muted={isIdle} />
          <Metric label="FFT SNR ratio"
            value={s7.snr_ratio !== undefined ? `${s7.snr_ratio}×` : undefined}
            muted={isIdle} />
        </div>

        {/* ── Chart + profile ── */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: '16px', marginTop: '0.5rem' }}>

          {/* SNR chart */}
          <div style={{
            background: 'var(--surface)', border: '1px solid var(--border)',
            borderRadius: 'var(--radius-md)', boxShadow: 'var(--shadow-sm)', padding: '1.25rem',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <p style={{ fontSize: '11px', fontWeight: 500, color: 'var(--text3)', letterSpacing: '0.07em', textTransform: 'uppercase' }}>
                SNR score — session history
              </p>
              <div style={{ display: 'flex', gap: '12px', fontSize: '11px' }}>
                {[['≥0.70', 'var(--success)', 'High'], ['≥0.45', 'var(--warning)', 'Medium'], ['<0.45', 'var(--danger)', 'Low']].map(([range, c, l]) => (
                  <div key={l} style={{ display: 'flex', alignItems: 'center', gap: '5px', color: 'var(--text2)' }}>
                    <div style={{ width: 8, height: 8, borderRadius: '50%', background: c }} />
                    {l} {range}
                  </div>
                ))}
              </div>
            </div>

            {snrHistory.length > 1 ? (
              <ResponsiveContainer width="100%" height={180}>
                <LineChart data={snrHistory} margin={{ top: 4, right: 8, left: -16, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="t" hide />
                  <YAxis domain={[0, 1]} tick={{ fontSize: 11, fill: '#8a95a3', fontFamily: 'DM Mono' }} />
                  <ReferenceLine y={0.7}  stroke="#0d7e5f" strokeDasharray="4 3" strokeWidth={1} />
                  <ReferenceLine y={0.45} stroke="#b45309" strokeDasharray="4 3" strokeWidth={1} />
                  <Tooltip content={<SNR_TOOLTIP />} />
                  <Line type="monotone" dataKey="snr" stroke="#1a56db" strokeWidth={2} dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div style={{ height: 180, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
                {isRunning && (
                  <div className="spin" style={{ width: 20, height: 20, borderRadius: '50%', border: '2px solid #e5e7eb', borderTopColor: 'var(--accent)' }} />
                )}
                <p style={{ fontSize: '13px', color: 'var(--text3)' }}>
                  {isIdle ? 'Waiting for pipeline to start…' : 'Collecting SNR data…'}
                </p>
              </div>
            )}

            {snrVal !== undefined && (
              <div style={{ marginTop: '1rem', paddingTop: '1rem', borderTop: '1px solid var(--border)' }}>
                <QualityBar score={snrVal} />
              </div>
            )}
          </div>

          {/* Subject profile panel */}
          <div style={{
            background: 'var(--surface)', border: '1px solid var(--border)',
            borderRadius: 'var(--radius-md)', boxShadow: 'var(--shadow-sm)', padding: '1.25rem',
          }}>
            <p style={{ fontSize: '11px', fontWeight: 500, color: 'var(--text3)', letterSpacing: '0.07em', textTransform: 'uppercase', marginBottom: '0.75rem' }}>
              Subject profile
            </p>

            <InfoRow label="ITA value"           value={s2.ita !== undefined ? `${s2.ita}` : null} mono />
            <InfoRow label="Fitzpatrick type"    value={s2.fitzpatrick} />
            <InfoRow label="Profile valid"       value={s2.profile_valid === true ? 'Yes' : s2.profile_valid === false ? 'No' : null} />
            <InfoRow label="Calibration HR"      value={s2.hr_estimate ? `${s2.hr_estimate} BPM` : null} mono />
            <InfoRow label="Filtered signal std" value={s6.filtered_std !== undefined ? s6.filtered_std.toFixed(6) : null} mono />
            <InfoRow label="Bandpass low cutoff" value={s6.bandpass_low_hz !== undefined ? `${s6.bandpass_low_hz} Hz` : null} mono />
            <InfoRow label="Measured FPS"        value={s3.fps ? `${s3.fps} fps` : null} mono />

            {/* Step completion grid */}
            <div style={{ marginTop: '1.2rem', paddingTop: '1rem', borderTop: '1px solid var(--border)' }}>
              <p style={{ fontSize: '11px', fontWeight: 500, color: 'var(--text3)', letterSpacing: '0.07em', textTransform: 'uppercase', marginBottom: '0.6rem' }}>
                Pipeline steps
              </p>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                {[1,2,3,4,5,6,7,8,9,10,11].map(n => (
                  <StepBadge key={n} n={n} done={!!steps[String(n)]} />
                ))}
              </div>
            </div>

            {/* Modality badge */}
            <div style={{ marginTop: '1.2rem', paddingTop: '1rem', borderTop: '1px solid var(--border)' }}>
              <p style={{ fontSize: '11px', fontWeight: 500, color: 'var(--text3)', letterSpacing: '0.07em', textTransform: 'uppercase', marginBottom: '0.6rem' }}>
                Modality
              </p>
              <div style={{ display: 'flex', gap: '8px' }}>
                <span style={{
                  padding: '4px 10px', borderRadius: 999, fontSize: '12px', fontWeight: 500,
                  background: !routePalm ? 'var(--accent-bg)' : 'var(--surface2)',
                  color: !routePalm ? 'var(--accent)' : 'var(--text3)',
                  border: `1px solid ${!routePalm ? '#bfdbfe' : 'var(--border)'}`,
                }}>Face</span>
                <span style={{
                  padding: '4px 10px', borderRadius: 999, fontSize: '12px', fontWeight: 500,
                  background: routePalm ? 'var(--warning-bg)' : 'var(--surface2)',
                  color: routePalm ? 'var(--warning)' : 'var(--text3)',
                  border: `1px solid ${routePalm ? '#fcd34d' : 'var(--border)'}`,
                }}>Palm</span>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div style={{
          marginTop: '2rem', paddingTop: '1rem', borderTop: '1px solid var(--border)',
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        }}>
          <p style={{ fontSize: '11px', color: 'var(--text3)', fontFamily: 'var(--font-mono)' }}>
            PulseRoute · rPPG Teleconsultation · For investigational use only
          </p>
          <p style={{ fontSize: '11px', color: 'var(--text3)', fontFamily: 'var(--font-mono)' }}>
            Steps 1–11 · Face modality
          </p>
        </div>
      </div>
    </div>
  )
}
